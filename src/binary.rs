use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::RngCore;
use rand::Rng;
use std::io::{self, Write};
use std::fs::File;

use crate::ert_core::{
    get_input, get_input_usize, get_bool, get_optional_input,
    calculate_n_binary, BinaryERTProcess, z_test_power_binary,
    FutilityMonitor, FutilityConfig,
};
use crate::agnostic::{AgnosticERT, Signal, Arm};

/// Compact trial result - only what we need
struct Trial {
    stop_n: Option<usize>,
    arr_at_stop: f64,
    arr_final: f64,
    min_wealth: f64,
    z_significant: bool,
    futility_ever_recommended: bool,  // Did FutilityMonitor ever recommend stop?
    futility_worst_ratio: Option<f64>, // Worst ratio observed
}

pub fn run() {
    println!("\n==========================================");
    println!("   BINARY e-RT SIMULATION");
    println!("==========================================\n");

    // === USER INPUTS ===
    let p_ctrl = get_input("Control Event Rate (e.g. 0.40): ");
    let p_trt = get_input("Treatment Event Rate (e.g. 0.30): ");
    let design_arr = (p_ctrl - p_trt).abs();

    let (n_patients, target_power) = if get_bool("Calculate Sample Size automatically?") {
        let power: f64 = get_input("Target Power (e.g. 0.80): ");
        let power = power.min(0.99);
        let n = calculate_n_binary(p_ctrl, p_trt, power);
        println!("\nFrequentist N (Power {:.0}%): {}", power * 100.0, n);

        let final_n = if get_bool("Add buffer?") {
            let buf: f64 = get_input("Buffer percentage (e.g. 15): ");
            let buffered = (n as f64 * (1.0 + buf / 100.0)).ceil() as usize;
            println!("Buffered N: {}", buffered);
            buffered
        } else { n };
        (final_n, Some(power))
    } else {
        (get_input_usize("Enter Number of Patients: "), None)
    };

    let n_sims = get_input_usize("Number of simulations (e.g. 2000): ");
    println!("\nSuccess threshold (1/alpha). Default = 20");
    let threshold: f64 = get_input("Success threshold: ");

    // Futility monitoring configuration
    let run_futility = get_bool("Enable futility monitoring?");
    let futility_config = if run_futility {
        println!("\n--- Futility Config (simulation-based, NOT martingale) ---");
        println!("Recovery target: probability of recovery to recommend stop");
        let recovery_target: f64 = get_input("Recovery target (default 0.10): ");
        println!("Stop ratio: recommend stop if required ARR > ratio * design ARR");
        let stop_ratio: f64 = get_input("Stop ratio (default 1.75): ");
        Some(FutilityConfig {
            recovery_target: if recovery_target > 0.0 { recovery_target } else { 0.10 },
            stop_ratio: if stop_ratio > 0.0 { stop_ratio } else { 1.75 },
            ..FutilityConfig::default()
        })
    } else {
        None
    };

    let seed = get_optional_input("Seed (Enter for random): ");

    let (burn_in, ramp) = (50usize, 100usize);

    println!("\n--- Configuration ---");
    println!("p_ctrl={:.1}% p_trt={:.1}% ARR={:.1}% N={} sims={} thresh={} fut={}",
        p_ctrl*100.0, p_trt*100.0, design_arr*100.0, n_patients, n_sims, threshold, run_futility);

    let mut rng: Box<dyn RngCore> = match seed {
        Some(s) => Box::new(StdRng::seed_from_u64(s)),
        None => Box::new(rand::thread_rng()),
    };

    // === PHASE 1: TYPE I ERROR ===
    print!("\nPhase 1: Type I Error... ");
    io::stdout().flush().unwrap();

    let mut null_rej = 0u32;
    for _ in 0..n_sims {
        let mut proc = BinaryERTProcess::new(burn_in, ramp);
        for i in 1..=n_patients {
            proc.update(i, if rng.gen_bool(p_ctrl) { 1.0 } else { 0.0 }, rng.gen_bool(0.5));
            if proc.wealth > threshold { null_rej += 1; break; }
        }
    }
    let type1 = null_rej as f64 / n_sims as f64 * 100.0;
    println!("{:.2}%", type1);

    // === PHASE 2: POWER + FUTILITY ===
    print!("Phase 2: Power");
    if run_futility { print!(" + Futility"); }
    print!("... ");
    io::stdout().flush().unwrap();

    let mut trials: Vec<Trial> = Vec::with_capacity(n_sims);
    let mut sample_trajs: Vec<Vec<f64>> = Vec::with_capacity(30);
    let mut agn_success = 0u32;
    let mut agn_stops: Vec<usize> = Vec::new();

    // Running percentile accumulators (for envelope plot)
    let step = 5;
    let n_points = n_patients / step + 1;
    let mut all_wealth_at_step: Vec<Vec<f64>> = vec![Vec::with_capacity(n_sims); n_points];

    for sim in 0..n_sims {
        if sim % (n_sims / 20).max(1) == 0 { print!("."); io::stdout().flush().unwrap(); }

        let mut proc = BinaryERTProcess::new(burn_in, ramp);
        let mut agn = AgnosticERT::new(burn_in, ramp, threshold);
        let (mut stopped, mut stop_n, mut arr_stop) = (false, None, 0.0);
        let mut agn_stopped = false;
        let mut min_w = 1.0f64;
        let mut traj = if sim < 30 { Vec::with_capacity(n_patients / step + 1) } else { Vec::new() };

        // Create futility monitor for this trial if enabled
        let mut fut_monitor = futility_config.as_ref().map(|cfg| {
            FutilityMonitor::new(cfg.clone(), design_arr, p_ctrl, n_patients, threshold, burn_in, ramp)
        });

        if sim < 30 { traj.push(1.0); }
        all_wealth_at_step[0].push(1.0);

        for i in 1..=n_patients {
            let is_trt = rng.gen_bool(0.5);
            let event = rng.gen_bool(if is_trt { p_trt } else { p_ctrl });
            proc.update(i, if event { 1.0 } else { 0.0 }, is_trt);

            min_w = min_w.min(proc.wealth);
            if i % step == 0 {
                all_wealth_at_step[i / step].push(proc.wealth);
                if sim < 30 { traj.push(proc.wealth); }
            }

            // Agnostic
            if !agn_stopped {
                let sig = Signal { arm: if is_trt { Arm::Treatment } else { Arm::Control }, good: !event };
                if agn.observe(sig) { agn_stopped = true; agn_stops.push(i); }
            }

            // Futility monitoring
            if let Some(ref mut monitor) = fut_monitor {
                if monitor.should_check(i) {
                    monitor.check(i, proc.wealth);
                }
            }

            // Success
            if !stopped && proc.wealth > threshold {
                stopped = true;
                stop_n = Some(i);
                arr_stop = proc.current_risk_diff().abs();
            }
        }

        if agn_stopped { agn_success += 1; }
        if sim < 30 { sample_trajs.push(traj); }

        // Z-test at end
        let z_sig = {
            let (n1, n2) = (proc.n_trt, proc.n_ctrl);
            if n1 > 0.0 && n2 > 0.0 {
                let (p1, p2) = (proc.events_trt / n1, proc.events_ctrl / n2);
                let pp = (proc.events_trt + proc.events_ctrl) / (n1 + n2);
                let se = (pp * (1.0 - pp) * (1.0/n1 + 1.0/n2)).sqrt();
                se > 0.0 && (p1 - p2).abs() / se > 1.96
            } else { false }
        };

        // Extract futility results
        let (fut_recommended, fut_ratio) = if let Some(monitor) = fut_monitor {
            (monitor.ever_recommended_stop(), monitor.worst_ratio())
        } else {
            (false, None)
        };

        trials.push(Trial {
            stop_n,
            arr_at_stop: arr_stop,
            arr_final: proc.current_risk_diff().abs(),
            min_wealth: min_w,
            z_significant: z_sig,
            futility_ever_recommended: fut_recommended,
            futility_worst_ratio: fut_ratio,
        });
    }
    println!(" done");

    // === STATISTICS ===
    let successes: Vec<&Trial> = trials.iter().filter(|t| t.stop_n.is_some()).collect();
    let n_success = successes.len();
    let power = n_success as f64 / n_sims as f64 * 100.0;

    let (avg_stop, avg_arr_stop, avg_arr_end, type_m) = if n_success > 0 {
        let sum_n: f64 = successes.iter().map(|t| t.stop_n.unwrap() as f64).sum();
        let sum_arr_s: f64 = successes.iter().map(|t| t.arr_at_stop).sum();
        let sum_arr_e: f64 = successes.iter().map(|t| t.arr_final).sum();
        let n = n_success as f64;
        (sum_n / n, sum_arr_s / n, sum_arr_e / n, sum_arr_s / sum_arr_e)
    } else { (0.0, 0.0, 0.0, 0.0) };

    let agn_power = agn_success as f64 / n_sims as f64 * 100.0;
    let _agn_avg = if !agn_stops.is_empty() { agn_stops.iter().sum::<usize>() as f64 / agn_stops.len() as f64 } else { 0.0 };
    let z_power = z_test_power_binary(p_ctrl, p_trt, n_patients, 1.0 / threshold) * 100.0;

    // Futility grid (single pass)
    let thresholds = [0.1, 0.2, 0.3, 0.4, 0.5];
    let mut grid: Vec<(f64, usize, usize, usize, f64)> = thresholds.iter()
        .map(|&th| (th, 0usize, 0usize, 0usize, 0.0f64)).collect();

    for t in &trials {
        for (th, n_trig, n_z, n_e, sum_arr) in &mut grid {
            if t.min_wealth < *th {
                *n_trig += 1;
                if t.z_significant { *n_z += 1; }
                if t.stop_n.is_some() { *n_e += 1; }
                *sum_arr += t.arr_final;
            }
        }
    }

    // Futility monitor stats
    let fut_recommended_count = trials.iter().filter(|t| t.futility_ever_recommended).count();
    let fut_ratios: Vec<f64> = trials.iter().filter_map(|t| t.futility_worst_ratio).collect();
    let fut_med_ratio = if !fut_ratios.is_empty() {
        let mut sorted = fut_ratios.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[sorted.len() / 2]
    } else { 0.0 };
    // Recovery rate: of those where stop was recommended, how many would have succeeded?
    let fut_would_recover = trials.iter()
        .filter(|t| t.futility_ever_recommended && t.stop_n.is_some())
        .count();
    let fut_recovery_rate = if fut_recommended_count > 0 {
        fut_would_recover as f64 / fut_recommended_count as f64 * 100.0
    } else { 0.0 };

    // === CONSOLE OUTPUT ===
    let mut out = String::with_capacity(2048);
    out.push_str("==========================================\n");
    out.push_str("   PARAMETERS\n");
    out.push_str("==========================================\n");
    out.push_str(&format!("Control:     {:.1}%\n", p_ctrl * 100.0));
    out.push_str(&format!("Treatment:   {:.1}%\n", p_trt * 100.0));
    out.push_str(&format!("Design ARR:  {:.1}%\n", design_arr * 100.0));
    out.push_str(&format!("N:           {}\n", n_patients));
    out.push_str(&format!("Simulations: {}\n", n_sims));
    out.push_str(&format!("Threshold:   {} (α={:.3})\n", threshold, 1.0/threshold));
    if let Some(ref cfg) = futility_config {
        out.push_str(&format!("Futility:    recovery={:.0}% ratio={:.2}x\n",
            cfg.recovery_target * 100.0, cfg.stop_ratio));
    } else {
        out.push_str("Futility:    disabled\n");
    }
    if let Some(p) = target_power {
        out.push_str(&format!("Target Pwr:  {:.0}%\n", p * 100.0));
    }
    out.push_str(&format!("Seed:        {}\n", seed.map_or("random".into(), |s| s.to_string())));

    out.push_str("\n==========================================\n");
    out.push_str("   RESULTS\n");
    out.push_str("==========================================\n");
    out.push_str(&format!("Type I Error:  {:.2}%\n\n", type1));
    out.push_str(&format!("--- Power at N={} ---\n", n_patients));
    out.push_str(&format!("Z-test:    {:.1}%\n", z_power));
    out.push_str(&format!("e-RT:      {:.1}%\n", power));
    out.push_str(&format!("e-RTu:     {:.1}%\n", agn_power));

    if n_success > 0 {
        out.push_str("\n--- Stopping ---\n");
        out.push_str(&format!("Avg stop:      {:.0} ({:.0}%)\n", avg_stop, avg_stop / n_patients as f64 * 100.0));
        out.push_str(&format!("ARR @ stop:    {:.1}%\n", avg_arr_stop * 100.0));
        out.push_str(&format!("ARR @ end:     {:.1}%\n", avg_arr_end * 100.0));
        out.push_str(&format!("Type M:        {:.2}x\n", type_m));
    }

    if run_futility && fut_recommended_count > 0 {
        out.push_str("\n--- Futility Monitor ---\n");
        out.push_str(&format!("Stop recommended:  {} ({:.1}%)\n",
            fut_recommended_count, fut_recommended_count as f64 / n_sims as f64 * 100.0));
        out.push_str(&format!("Would recover:     {} ({:.1}%)\n",
            fut_would_recover, fut_recovery_rate));
        out.push_str(&format!("Median ratio:      {:.2}x\n", fut_med_ratio));
    }

    print!("\n{}", out);
    println!("Generating report...");

    // === PREPARE PLOT DATA ===
    // Percentiles from accumulated wealth
    let mut x_pts: Vec<usize> = Vec::new();
    let mut y_lo: Vec<f64> = Vec::new();
    let mut y_med: Vec<f64> = Vec::new();
    let mut y_hi: Vec<f64> = Vec::new();

    for (idx, vals) in all_wealth_at_step.iter_mut().enumerate() {
        if vals.is_empty() { continue; }
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = vals.len();
        x_pts.push(idx * step);
        y_lo.push(vals[(n as f64 * 0.025) as usize]);
        y_med.push(vals[n / 2]);
        y_hi.push(vals[(n as f64 * 0.975) as usize]);
    }

    let stop_times: Vec<f64> = successes.iter().map(|t| t.stop_n.unwrap() as f64).collect();

    // === BUILD HTML ===
    let html = build_report(
        &out, threshold, n_patients, run_futility,
        &x_pts, &y_lo, &y_med, &y_hi, &sample_trajs,
        &stop_times, &grid, n_sims,
        fut_recommended_count, fut_med_ratio, &fut_ratios,
    );

    File::create("binary_report.html").unwrap().write_all(html.as_bytes()).unwrap();
    println!("\n>> binary_report.html");
}

fn build_report(
    console: &str, threshold: f64, n_pts: usize, run_futility: bool,
    x: &[usize], y_lo: &[f64], y_med: &[f64], y_hi: &[f64], trajs: &[Vec<f64>],
    stops: &[f64], grid: &[(f64, usize, usize, usize, f64)], n_sims: usize,
    fut_count: usize, fut_ratio: f64, fut_ratios: &[f64],
) -> String {
    let x_js = format!("{:?}", x);

    // Sample trajectories
    let mut traces = String::new();
    for tr in trajs.iter().take(30) {
        traces.push_str(&format!(
            "{{type:'scatter',mode:'lines',x:{},y:{:?},line:{{color:'rgba(100,100,100,0.25)',width:1}},showlegend:false}},",
            x_js, tr
        ));
    }

    // Stop ECDF
    let (stop_x, stop_y) = if !stops.is_empty() {
        let mut s = stops.to_vec();
        s.sort_by(|a,b| a.partial_cmp(b).unwrap());
        let y: Vec<f64> = (1..=s.len()).map(|i| i as f64 / s.len() as f64 * 100.0).collect();
        (format!("{:?}", s), format!("{:?}", y))
    } else { ("[]".into(), "[]".into()) };

    // Grid table + plot data
    let mut grid_tbl = String::from("Thresh | Triggered |  Z-test+  |   e-RT+   | Avg ARR\n");
    grid_tbl.push_str("-------|-----------|-----------|-----------|--------\n");
    let (mut g_th, mut g_z, mut g_e): (Vec<f64>, Vec<f64>, Vec<f64>) = (vec![], vec![], vec![]);
    for &(th, nt, nz, ne, sum) in grid {
        let (pct_t, pct_z, pct_e, avg) = if nt > 0 {
            (nt as f64 / n_sims as f64 * 100.0, nz as f64 / nt as f64 * 100.0,
             ne as f64 / nt as f64 * 100.0, sum / nt as f64 * 100.0)
        } else { (0.0, 0.0, 0.0, 0.0) };
        grid_tbl.push_str(&format!(" {:.1}   |   {:5.1}%  |   {:5.1}%  |   {:5.1}%  |  {:4.1}%\n",
            th, pct_t, pct_z, pct_e, avg));
        g_th.push(th); g_z.push(pct_z); g_e.push(pct_e);
    }

    // Futility ratio ECDF
    let (fut_div, fut_plot) = if run_futility && !fut_ratios.is_empty() {
        let mut s = fut_ratios.to_vec();
        s.sort_by(|a,b| a.partial_cmp(b).unwrap());
        let y: Vec<f64> = (1..=s.len()).map(|i| i as f64 / s.len() as f64).collect();
        (
            "<h3>Recovery Difficulty (Ratio Distribution)</h3><div id=\"p5\" style=\"height:280px\"></div>".into(),
            format!("Plotly.newPlot('p5',[{{type:'scatter',mode:'lines',x:{:?},y:{:?},line:{{color:'coral',width:2}}}}],{{xaxis:{{title:'Required ARR / Design ARR'}},yaxis:{{title:'Cumulative'}},shapes:[{{type:'line',x0:1,x1:1,y0:0,y1:1,line:{{color:'green',dash:'dash'}}}},{{type:'line',x0:1.75,x1:1.75,y0:0,y1:1,line:{{color:'red',dash:'dot'}}}}]}});", s, y)
        )
    } else { (String::new(), String::new()) };

    let fut_txt = if run_futility && fut_count > 0 {
        format!("Futility: {} triggered ({:.1}%), median ratio {:.2}x\n",
            fut_count, fut_count as f64 / n_sims as f64 * 100.0, fut_ratio)
    } else { String::new() };

    format!(r#"<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>e-RT Binary Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
body{{font-family:system-ui,-apple-system,sans-serif;max-width:1400px;margin:0 auto;padding:20px;background:#fafafa}}
h1{{color:#1a1a2e}}h2,h3{{color:#16213e}}
pre{{background:#fff;padding:15px;border-radius:8px;border:1px solid #ddd;overflow-x:auto;font-size:13px}}
.plot-container{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin:20px 0}}
.plot{{background:#fff;border-radius:8px;padding:10px;box-shadow:0 1px 3px rgba(0,0,0,0.1)}}
</style></head><body>
<h1>e-RT Binary Report</h1>
<h2>Console Output</h2>
<pre>{}</pre>
<h2>Visualizations</h2>
<div class="plot-container">
<div class="plot"><div id="p1" style="height:350px"></div></div>
<div class="plot"><div id="p2" style="height:350px"></div></div>
<div class="plot"><div id="p3" style="height:350px"></div></div>
<div class="plot"><div id="p4" style="height:350px"></div></div>
</div>
{}
<h2>Futility Grid</h2>
<pre>{}{}</pre>
<script>
Plotly.newPlot('p1',[
  {{type:'scatter',x:{},y:{:?},line:{{width:0}},showlegend:false}},
  {{type:'scatter',x:{},y:{:?},fill:'tonexty',fillcolor:'rgba(31,119,180,0.2)',line:{{width:0}},showlegend:false}},
  {{type:'scatter',x:{},y:{:?},line:{{color:'#1f77b4',width:2}},name:'Median'}}
],{{yaxis:{{type:'log',title:'e-value'}},xaxis:{{title:'Patients'}},shapes:[{{type:'line',x0:0,x1:1,xref:'paper',y0:{},y1:{},line:{{color:'green',dash:'dash',width:2}}}}]}});
Plotly.newPlot('p2',[{}
  {{type:'scatter',x:[0,{}],y:[{},{}],line:{{color:'green',dash:'dash',width:2}},name:'Threshold'}}
],{{yaxis:{{type:'log',title:'e-value'}},xaxis:{{title:'Patients'}}}});
Plotly.newPlot('p3',[{{type:'scatter',mode:'lines',x:{},y:{},line:{{color:'steelblue',width:2}}}}],{{xaxis:{{title:'Stop (patients)'}},yaxis:{{title:'%',range:[0,100]}}}});
Plotly.newPlot('p4',[
  {{type:'scatter',mode:'lines+markers',x:{:?},y:{:?},line:{{color:'purple'}},name:'Z-test+'}},
  {{type:'scatter',mode:'lines+markers',x:{:?},y:{:?},line:{{color:'blue'}},name:'e-RT+'}}
],{{xaxis:{{title:'Threshold'}},yaxis:{{title:'%',range:[0,100]}}}});
{}
</script></body></html>"#,
        console, fut_div, fut_txt, grid_tbl,
        x_js, y_lo, x_js, y_hi, x_js, y_med, threshold, threshold,
        traces, n_pts, threshold, threshold,
        stop_x, stop_y, g_th, g_z, g_th, g_e, fut_plot
    )
}
