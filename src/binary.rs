use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::RngCore;
use rand::Rng;
use std::io::{self, Write};
use std::fs::File;
use std::time::Instant;

use crate::ert_core::{
    get_input, get_input_usize, get_bool, get_optional_input,
    calculate_n_binary, BinaryERTProcess, z_test_power_binary,
};
use crate::agnostic::{AgnosticERT, Signal, Arm};

/// Compact trial result
struct Trial {
    stop_n: Option<usize>,
    arr_at_stop: f64,
    arr_final: f64,
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

    let seed = get_optional_input("Seed (Enter for random): ");

    let (burn_in, ramp) = (50usize, 100usize);

    println!("\n--- Configuration ---");
    println!("p_ctrl={:.1}% p_trt={:.1}% ARR={:.1}% N={} sims={} thresh={}",
        p_ctrl*100.0, p_trt*100.0, design_arr*100.0, n_patients, n_sims, threshold);

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

    // === PHASE 2: POWER ===
    println!("Phase 2: Power");
    io::stdout().flush().unwrap();

    let mut trials: Vec<Trial> = Vec::with_capacity(n_sims);
    let mut pos_trajs: Vec<Vec<f64>> = Vec::new();  // Crossed threshold
    let mut neg_trajs: Vec<Vec<f64>> = Vec::new();  // Did not cross
    let mut agn_success = 0u32;
    let mut agn_stops: Vec<usize> = Vec::new();

    // Running percentile accumulators (for envelope plot)
    let step = 5;
    let n_points = n_patients / step + 1;
    let mut all_wealth_at_step: Vec<Vec<f64>> = vec![Vec::with_capacity(n_sims); n_points];

    let start_time = Instant::now();
    let progress_interval = 50;

    for sim in 0..n_sims {
        // Progress bar with ETA
        if sim > 0 && sim % progress_interval == 0 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let rate = sim as f64 / elapsed;
            let remaining = (n_sims - sim) as f64 / rate;
            let pct = sim as f64 / n_sims as f64 * 100.0;
            print!("\r  [{:>5.1}%] {}/{} ({:.0}s elapsed, ~{:.0}s remaining)     ",
                   pct, sim, n_sims, elapsed, remaining);
            io::stdout().flush().unwrap();
        }

        let mut proc = BinaryERTProcess::new(burn_in, ramp);
        let mut agn = AgnosticERT::new(burn_in, ramp, threshold);
        let (mut stopped, mut stop_n, mut arr_stop) = (false, None, 0.0);
        let mut agn_stopped = false;
        let need_traj = pos_trajs.len() < 30 || neg_trajs.len() < 30;
        let mut traj = if need_traj { Vec::with_capacity(n_patients / step + 1) } else { Vec::new() };

        if need_traj { traj.push(1.0); }
        all_wealth_at_step[0].push(1.0);

        for i in 1..=n_patients {
            let is_trt = rng.gen_bool(0.5);
            let event = rng.gen_bool(if is_trt { p_trt } else { p_ctrl });
            proc.update(i, if event { 1.0 } else { 0.0 }, is_trt);

            if i % step == 0 {
                all_wealth_at_step[i / step].push(proc.wealth);
                if need_traj { traj.push(proc.wealth); }
            }

            // Agnostic
            if !agn_stopped {
                let sig = Signal { arm: if is_trt { Arm::Treatment } else { Arm::Control }, good: !event };
                if agn.observe(sig) { agn_stopped = true; agn_stops.push(i); }
            }

            // Success
            if !stopped && proc.wealth > threshold {
                stopped = true;
                stop_n = Some(i);
                arr_stop = proc.current_risk_diff().abs();
            }
        }

        if agn_stopped { agn_success += 1; }
        // Collect positive/negative trajectories separately
        if need_traj {
            if stop_n.is_some() {
                if pos_trajs.len() < 30 { pos_trajs.push(traj); }
            } else if neg_trajs.len() < 30 {
                neg_trajs.push(traj);
            }
        }

        trials.push(Trial {
            stop_n,
            arr_at_stop: arr_stop,
            arr_final: proc.current_risk_diff().abs(),
        });
    }
    // Clear progress line and show completion
    let total_time = start_time.elapsed().as_secs_f64();
    print!("\r  [100.0%] {}/{} complete ({:.1}s total)                    \n", n_sims, n_sims, total_time);
    io::stdout().flush().unwrap();

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
    let z_power = z_test_power_binary(p_ctrl, p_trt, n_patients, 1.0 / threshold) * 100.0;

    // Build representative sample: proportion of positive samples matches power
    let n_pos_sample = ((power / 100.0) * 30.0).round() as usize;
    let n_neg_sample = 30 - n_pos_sample;
    let mut sample_trajs: Vec<Vec<f64>> = Vec::new();
    sample_trajs.extend(pos_trajs.into_iter().take(n_pos_sample));
    sample_trajs.extend(neg_trajs.into_iter().take(n_neg_sample));

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
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = vals.len();
        x_pts.push(idx * step);
        y_lo.push(vals[(n as f64 * 0.025) as usize]);
        y_med.push(vals[n / 2]);
        y_hi.push(vals[(n as f64 * 0.975) as usize]);
    }

    let stop_times: Vec<f64> = successes.iter().map(|t| t.stop_n.unwrap() as f64).collect();

    // === BUILD HTML ===
    let html = build_report(
        &out, threshold, n_patients,
        &x_pts, &y_lo, &y_med, &y_hi, &sample_trajs,
        &stop_times,
    );

    match File::create("binary_report.html").and_then(|mut f| f.write_all(html.as_bytes())) {
        Ok(_) => println!("\n>> binary_report.html"),
        Err(e) => eprintln!("\nError saving report: {}", e),
    }
}

fn build_report(
    console: &str, threshold: f64, n_pts: usize,
    x: &[usize], y_lo: &[f64], y_med: &[f64], y_hi: &[f64], trajs: &[Vec<f64>],
    stops: &[f64],
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
        s.sort_by(|a,b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let y: Vec<f64> = (1..=s.len()).map(|i| i as f64 / s.len() as f64 * 100.0).collect();
        (format!("{:?}", s), format!("{:?}", y))
    } else { ("[]".into(), "[]".into()) };

    format!(r#"<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>e-RT Binary Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
body{{font-family:system-ui,-apple-system,sans-serif;max-width:900px;margin:0 auto;padding:20px;background:#fafafa}}
h1{{color:#1a1a2e}}h2,h3{{color:#16213e}}
pre{{background:#fff;padding:15px;border-radius:8px;border:1px solid #ddd;overflow-x:auto;font-size:13px}}
.plot{{background:#fff;border-radius:8px;padding:15px;margin:20px 0;box-shadow:0 1px 3px rgba(0,0,0,0.1)}}
</style></head><body>
<h1>e-RT Binary Report</h1>
<h2>Console Output</h2>
<pre>{}</pre>
<h2>Visualizations</h2>
<div class="plot"><div id="p1" style="height:400px"></div></div>
<div class="plot"><div id="p2" style="height:400px"></div></div>
<div class="plot"><div id="p3" style="height:400px"></div></div>
<script>
Plotly.newPlot('p1',[
  {{type:'scatter',x:{},y:{:?},line:{{width:0}},showlegend:false}},
  {{type:'scatter',x:{},y:{:?},fill:'tonexty',fillcolor:'rgba(31,119,180,0.2)',line:{{width:0}},showlegend:false}},
  {{type:'scatter',x:{},y:{:?},line:{{color:'#1f77b4',width:2}},name:'Median'}}
],{{yaxis:{{type:'log',title:'e-value'}},xaxis:{{title:'Patients'}},shapes:[{{type:'line',x0:0,x1:1,xref:'paper',y0:{},y1:{},line:{{color:'green',dash:'dash',width:2}}}}],title:'e-Value Envelope (2.5-97.5 percentile)'}});
Plotly.newPlot('p2',[{}
  {{type:'scatter',x:[0,{}],y:[{},{}],line:{{color:'green',dash:'dash',width:2}},name:'Threshold'}}
],{{yaxis:{{type:'log',title:'e-value'}},xaxis:{{title:'Patients'}},title:'Sample Trajectories (30 trials)'}});
Plotly.newPlot('p3',[{{type:'scatter',mode:'lines',x:{},y:{},line:{{color:'steelblue',width:2}}}}],{{xaxis:{{title:'Stop (patients)'}},yaxis:{{title:'Cumulative %',range:[0,100]}},title:'Early Stopping ECDF'}});
</script></body></html>"#,
        console,
        x_js, y_lo, x_js, y_hi, x_js, y_med, threshold, threshold,
        traces, n_pts, threshold, threshold,
        stop_x, stop_y
    )
}
