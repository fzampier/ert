use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::RngCore;
use std::fs::File;
use std::io::{self, Write};

use crate::ert_core::{get_input, get_input_usize, get_bool, get_optional_input, chrono_lite, normal_cdf};
use crate::agnostic::{AgnosticERT, Signal, Arm};

// === HELPERS ===

fn calculate_n_survival(target_hr: f64, power: f64) -> usize {
    let z_alpha: f64 = 1.96;
    let z_beta: f64 = if power > 0.85 { 1.28 } else { 0.84 };
    let log_hr = target_hr.ln();
    (4.0 * ((z_alpha + z_beta) / log_hr).powi(2)).ceil() as usize
}

fn log_rank_power(target_hr: f64, n_events: usize, alpha: f64) -> f64 {
    let z_alpha = if (alpha - 0.05).abs() < 0.001 { 1.96 } else { 2.576 };
    let log_hr = target_hr.ln().abs();
    let z_effect = log_hr * (n_events as f64 / 4.0).sqrt();
    normal_cdf(z_effect - z_alpha)
}

// === DATA ===

struct SurvivalData {
    time: Vec<f64>,
    status: Vec<u8>,
    treatment: Vec<u8>,
}

struct Trial {
    stop_n: Option<usize>,
    hr_at_stop: f64,
    hr_final: f64,
    min_wealth: f64,
    agnostic_stopped: bool,
}

// === SIMULATE ===

fn simulate_trial<R: Rng + ?Sized>(
    rng: &mut R, n: usize, hr: f64, shape: f64, scale: f64, cens_prop: f64,
) -> SurvivalData {
    let mut treatment = vec![0u8; n];
    let mut time = vec![0.0; n];
    let mut status = vec![1u8; n];
    let scale_trt = scale / hr.powf(1.0 / shape);

    for i in 0..n {
        treatment[i] = if rng.gen_bool(0.5) { 1 } else { 0 };
        let u: f64 = rng.gen();
        let s = if treatment[i] == 1 { scale_trt } else { scale };
        let true_time = s * (-u.ln()).powf(1.0 / shape);

        if cens_prop > 0.0 {
            let cens_time = rng.gen::<f64>() * 2.0 * scale;
            if cens_time < true_time {
                time[i] = cens_time;
                status[i] = 0;
            } else {
                time[i] = true_time;
            }
        } else {
            time[i] = true_time;
        }
    }
    SurvivalData { time, status, treatment }
}

// === e-SURVIVAL ===

fn compute_e_survival(
    data: &SurvivalData, burn_in: usize, ramp: usize, lambda_max: f64,
) -> Vec<f64> {
    let n = data.time.len();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| data.time[a].partial_cmp(&data.time[b]).unwrap());

    let mut wealth = vec![1.0; n];
    let mut cumulative_z: f64 = 0.0;
    let mut risk_trt: i32 = data.treatment.iter().filter(|&&t| t == 1).count() as i32;
    let mut risk_ctrl: i32 = data.treatment.iter().filter(|&&t| t == 0).count() as i32;

    for (i, &idx) in indices.iter().enumerate() {
        let is_event = data.status[idx] == 1;
        let is_trt = data.treatment[idx] == 1;

        let b_i = if i > burn_in {
            let c_i = (((i - burn_in) as f64) / ramp as f64).clamp(0.0, 1.0);
            let bet_direction = if cumulative_z > 0.0 { 1.0 } else if cumulative_z < 0.0 { -1.0 } else { 0.0 };
            c_i * lambda_max * bet_direction
        } else { 0.0 };

        let total_risk = risk_trt + risk_ctrl;
        let p_null = if total_risk > 0 { risk_trt as f64 / total_risk as f64 } else { 0.5 };

        if is_event {
            let obs = if is_trt { 1.0 } else { 0.0 };
            let u_i = obs - p_null;
            let multiplier = 1.0 + b_i * u_i;
            cumulative_z += u_i;
            wealth[i] = if i > 0 { wealth[i - 1] * multiplier } else { multiplier };
        } else if i > 0 {
            wealth[i] = wealth[i - 1];
        }

        if is_trt { risk_trt = (risk_trt - 1).max(0); }
        else { risk_ctrl = (risk_ctrl - 1).max(0); }
    }
    wealth
}

fn compute_agnostic_survival(
    data: &SurvivalData, burn_in: usize, ramp: usize, threshold: f64,
) -> (bool, Option<usize>) {
    let n = data.time.len();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| data.time[a].partial_cmp(&data.time[b]).unwrap());

    let mut agnostic = AgnosticERT::new(burn_in, ramp, threshold);

    for (i, &idx) in indices.iter().enumerate() {
        let is_event = data.status[idx] == 1;
        let is_trt = data.treatment[idx] == 1;
        let signal = Signal {
            arm: if is_trt { Arm::Treatment } else { Arm::Control },
            good: !is_event,
        };
        if agnostic.observe(signal) { return (true, Some(i + 1)); }
    }
    (false, None)
}

fn calculate_observed_hr(data: &SurvivalData, max_events: Option<usize>) -> f64 {
    let mut indices: Vec<usize> = (0..data.time.len()).collect();
    indices.sort_by(|&a, &b| data.time[a].partial_cmp(&data.time[b]).unwrap());

    let (mut events_trt, mut events_ctrl, mut time_trt, mut time_ctrl) = (0.0, 0.0, 0.0, 0.0);
    let mut event_count = 0;

    for &idx in &indices {
        if data.treatment[idx] == 1 {
            time_trt += data.time[idx];
            if data.status[idx] == 1 { events_trt += 1.0; event_count += 1; }
        } else {
            time_ctrl += data.time[idx];
            if data.status[idx] == 1 { events_ctrl += 1.0; event_count += 1; }
        }
        if let Some(max) = max_events { if event_count >= max { break; } }
    }

    let rate_trt = if time_trt > 0.0 { events_trt / time_trt } else { 0.0 };
    let rate_ctrl = if time_ctrl > 0.0 { events_ctrl / time_ctrl } else { 0.0 };
    if rate_ctrl > 0.0 { rate_trt / rate_ctrl } else { 1.0 }
}

// === HTML REPORT ===

fn build_report(
    console: &str, n_pts: usize,
    threshold: f64, fut_watch: f64,
    x: &[usize], y_lo: &[f64], y_med: &[f64], y_hi: &[f64],
    trajs: &[Vec<f64>], stops: &[f64], min_wealths: &[f64],
    grid: &[(f64, usize, usize, f64)],
) -> String {
    let x_json = format!("{:?}", x);
    let lo_json = format!("{:?}", y_lo);
    let med_json = format!("{:?}", y_med);
    let hi_json = format!("{:?}", y_hi);

    let mut sample_traces = String::new();
    for traj in trajs.iter().take(30) {
        sample_traces.push_str(&format!(
            "{{type:'scatter',mode:'lines',x:{},y:{:?},line:{{color:'rgba(0,100,0,0.3)',width:1}},showlegend:false}},",
            x_json, traj
        ));
    }

    // ECDF for stops
    let mut sorted_stops = stops.to_vec();
    sorted_stops.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let ecdf_x: Vec<f64> = sorted_stops.clone();
    let ecdf_y: Vec<f64> = (1..=sorted_stops.len()).map(|i| i as f64 / sorted_stops.len() as f64).collect();

    // ECDF for min wealth
    let mut sorted_mw = min_wealths.to_vec();
    sorted_mw.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mw_x: Vec<f64> = sorted_mw.clone();
    let mw_y: Vec<f64> = (1..=sorted_mw.len()).map(|i| i as f64 / sorted_mw.len() as f64).collect();

    // Grid line plot
    let grid_th: Vec<f64> = grid.iter().map(|(th, _, _, _)| *th).collect();
    let grid_e: Vec<f64> = grid.iter().map(|(_, n_trig, n_e, _)| if *n_trig > 0 { (*n_e as f64 / *n_trig as f64) * 100.0 } else { 0.0 }).collect();

    format!(r#"<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>e-RTs Survival Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
body{{font-family:system-ui,-apple-system,sans-serif;max-width:1400px;margin:0 auto;padding:20px;background:#fafafa}}
h1{{color:#1a1a2e}}h2,h3{{color:#16213e}}
pre{{background:#fff;padding:15px;border-radius:8px;border:1px solid #ddd;overflow-x:auto;font-size:13px}}
.plot-container{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin:20px 0}}
.plot{{background:#fff;border-radius:8px;padding:10px;box-shadow:0 1px 3px rgba(0,0,0,0.1)}}
</style></head><body>
<h1>e-RTs Survival Report</h1>
<h2>Console Output</h2>
<pre>{}</pre>
<h2>Visualizations</h2>
<div class="plot-container">
<div class="plot"><div id="p1" style="height:350px"></div></div>
<div class="plot"><div id="p2" style="height:350px"></div></div>
<div class="plot"><div id="p3" style="height:350px"></div></div>
<div class="plot"><div id="p4" style="height:350px"></div></div>
<div class="plot"><div id="p5" style="height:350px"></div></div>
</div>
<script>
Plotly.newPlot('p1',[
  {{type:'scatter',x:{},y:{},line:{{width:0}},showlegend:false}},
  {{type:'scatter',x:{},y:{},fill:'tonexty',fillcolor:'rgba(0,100,0,0.3)',line:{{width:0}},showlegend:false}},
  {{type:'scatter',x:{},y:{},line:{{color:'darkgreen',width:2}},name:'Median'}}
],{{title:'Wealth Trajectory (5-95% CI)',yaxis:{{type:'log',title:'e-value'}},xaxis:{{title:'Events'}},
shapes:[{{type:'line',x0:0,x1:1,xref:'paper',y0:{},y1:{},line:{{color:'green',width:2,dash:'dash'}}}}]}});
Plotly.newPlot('p2',[{}
  {{type:'scatter',x:[0,{}],y:[{},{}],line:{{color:'green',width:2,dash:'dash'}},name:'Success'}},
  {{type:'scatter',x:[0,{}],y:[{},{}],line:{{color:'coral',width:2,dash:'dot'}},name:'Futility'}}
],{{title:'Sample Trajectories (n=30)',yaxis:{{type:'log',title:'e-value'}},xaxis:{{title:'Events'}}}});
Plotly.newPlot('p3',[{{type:'scatter',mode:'lines',x:{:?},y:{:?},line:{{color:'steelblue',width:2}}}}],
{{title:'Stopping Time ECDF',xaxis:{{title:'Event #'}},yaxis:{{title:'Cumulative Proportion'}}}});
Plotly.newPlot('p4',[{{type:'scatter',mode:'lines',x:{:?},y:{:?},line:{{color:'coral',width:2}}}}],
{{title:'Minimum Wealth ECDF',xaxis:{{title:'Min Wealth'}},yaxis:{{title:'Cumulative Proportion'}},
shapes:[{{type:'line',x0:{},x1:{},y0:0,y1:1,line:{{color:'red',width:1,dash:'dash'}}}}]}});
Plotly.newPlot('p5',[{{type:'scatter',mode:'lines+markers',x:{:?},y:{:?},name:'e-RTs+',line:{{color:'coral'}}}}],
{{title:'Recovery Rate by Threshold',xaxis:{{title:'Futility Threshold'}},yaxis:{{title:'% Would Succeed'}}}});
</script></body></html>"#,
        console,
        x_json, lo_json, x_json, hi_json, x_json, med_json, threshold, threshold,
        sample_traces, n_pts, threshold, threshold, n_pts, fut_watch, fut_watch,
        ecdf_x, ecdf_y,
        mw_x, mw_y, fut_watch, fut_watch,
        grid_th, grid_e
    )
}

// === MAIN ===

pub fn run() {
    println!("\n==========================================");
    println!("   e-RTs SURVIVAL SIMULATION");
    println!("==========================================\n");

    let target_hr: f64 = get_input("Target HR (e.g., 0.80): ");

    let n_pts = if get_bool("Calculate Sample Size automatically?") {
        let power: f64 = get_input("Target Power (e.g., 0.80): ");
        let freq_n = calculate_n_survival(target_hr, power);
        println!("\nSchoenfeld N (Power {:.0}%, HR={:.2}): {}", power * 100.0, target_hr, freq_n);
        if get_bool("Add buffer?") {
            let buf: f64 = get_input("Buffer percentage (e.g., 15): ");
            let buffered = (freq_n as f64 * (1.0 + buf / 100.0)).ceil() as usize;
            println!("Buffered N: {}", buffered);
            buffered
        } else { freq_n }
    } else {
        get_input_usize("Enter Number of Patients: ")
    };

    let n_sims = get_input_usize("Number of simulations (e.g., 2000): ");
    let threshold: f64 = get_input("Success threshold (default 20): ");
    let fut_watch: f64 = get_input("Futility watch (default 0.2): ");

    println!("\nWeibull parameters:");
    let shape: f64 = get_input("Shape (default 1.2): ");
    let scale: f64 = get_input("Scale (default 10): ");
    let cens_prop: f64 = get_input("Censoring proportion 0-1 (default 0): ");

    let seed = get_optional_input("Seed (Enter for random): ");

    let burn_in: usize = 30;
    let ramp: usize = 50;
    let lambda_max: f64 = 0.25;

    // Console output capture
    let mut console = String::new();
    console.push_str(&format!("{}\n", chrono_lite()));
    console.push_str(&format!("\n==========================================\n"));
    console.push_str(&format!("   PARAMETERS\n"));
    console.push_str(&format!("==========================================\n"));
    console.push_str(&format!("Target HR:   {:.2}\n", target_hr));
    console.push_str(&format!("N:           {}\n", n_pts));
    console.push_str(&format!("Simulations: {}\n", n_sims));
    console.push_str(&format!("Threshold:   {} (α={:.3})\n", threshold, 1.0/threshold));
    console.push_str(&format!("Futility:    {}\n", fut_watch));
    console.push_str(&format!("Weibull:     shape={}, scale={}\n", shape, scale));
    console.push_str(&format!("Censoring:   {:.0}%\n", cens_prop * 100.0));

    let mut rng: Box<dyn RngCore> = match seed {
        Some(s) => { console.push_str(&format!("Seed:        {}\n", s)); Box::new(StdRng::seed_from_u64(s)) }
        None => { console.push_str("Seed:        random\n"); Box::new(rand::thread_rng()) }
    };

    println!("\n==========================================");
    println!("   RUNNING SIMULATIONS");
    println!("==========================================\n");

    // Type I Error
    print!("  Type I Error (HR=1.0)... ");
    io::stdout().flush().unwrap();
    let mut null_rejections = 0;
    for _ in 0..n_sims {
        let trial = simulate_trial(&mut *rng, n_pts, 1.0, shape, scale, cens_prop);
        let wealth = compute_e_survival(&trial, burn_in, ramp, lambda_max);
        if wealth.iter().any(|&w| w > threshold) { null_rejections += 1; }
    }
    let type1 = (null_rejections as f64 / n_sims as f64) * 100.0;
    println!("{:.2}%", type1);

    // Power simulation
    print!("  Power (HR={:.2})... ", target_hr);
    io::stdout().flush().unwrap();

    let sample_indices: Vec<usize> = (0..30.min(n_sims)).collect();
    let step = (n_pts / 100).max(1);
    let steps: Vec<usize> = (1..=n_pts).filter(|&i| i % step == 0 || i == n_pts).collect();

    let mut trials: Vec<Trial> = Vec::with_capacity(n_sims);
    let mut sample_trajs: Vec<Vec<f64>> = vec![Vec::new(); sample_indices.len()];
    let mut step_vals: Vec<Vec<f64>> = vec![Vec::with_capacity(n_sims); steps.len()];

    let pb_interval = (n_sims / 20).max(1);

    for sim in 0..n_sims {
        if sim % pb_interval == 0 { print!("."); io::stdout().flush().unwrap(); }

        let data = simulate_trial(&mut *rng, n_pts, target_hr, shape, scale, cens_prop);
        let wealth = compute_e_survival(&data, burn_in, ramp, lambda_max);

        let (agn_stopped, _) = compute_agnostic_survival(&data, burn_in, ramp, threshold);

        let mut stop_n: Option<usize> = None;
        let mut hr_at_stop = 1.0;
        let mut min_wealth = 1.0f64;

        for (i, &w) in wealth.iter().enumerate() {
            min_wealth = min_wealth.min(w);
            if stop_n.is_none() && w > threshold {
                stop_n = Some(i);
                hr_at_stop = calculate_observed_hr(&data, Some(i + 1));
            }
        }

        // Track at steps
        let is_sample = sample_indices.contains(&sim);
        for (step_idx, &s) in steps.iter().enumerate() {
            if s < wealth.len() {
                step_vals[step_idx].push(wealth[s]);
                if is_sample {
                    let idx = sample_indices.iter().position(|&x| x == sim).unwrap();
                    sample_trajs[idx].push(wealth[s]);
                }
            }
        }

        let hr_final = calculate_observed_hr(&data, None);
        trials.push(Trial { stop_n, hr_at_stop, hr_final, min_wealth, agnostic_stopped: agn_stopped });
    }
    println!(" done");

    // Compute percentiles
    let mut y_lo: Vec<f64> = vec![0.0; steps.len()];
    let mut y_med: Vec<f64> = vec![0.0; steps.len()];
    let mut y_hi: Vec<f64> = vec![0.0; steps.len()];
    for (i, vals) in step_vals.iter_mut().enumerate() {
        if vals.is_empty() { continue; }
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = vals.len();
        y_lo[i] = vals[(n as f64 * 0.05) as usize];
        y_med[i] = vals[n / 2];
        y_hi[i] = vals[((n as f64 * 0.95) as usize).min(n - 1)];
    }

    // Statistics
    let successes: Vec<&Trial> = trials.iter().filter(|t| t.stop_n.is_some()).collect();
    let power = (successes.len() as f64 / n_sims as f64) * 100.0;
    let agn_power = (trials.iter().filter(|t| t.agnostic_stopped).count() as f64 / n_sims as f64) * 100.0;

    let expected_events = (n_pts as f64 * (1.0 - cens_prop)).round() as usize;
    let lr_power = log_rank_power(target_hr, expected_events, 1.0/threshold) * 100.0;

    let (avg_stop, avg_hr_stop, avg_hr_final, type_m) = if !successes.is_empty() {
        let avg_n = successes.iter().map(|t| t.stop_n.unwrap() as f64).sum::<f64>() / successes.len() as f64;
        let avg_s = successes.iter().map(|t| t.hr_at_stop).sum::<f64>() / successes.len() as f64;
        let avg_f = successes.iter().map(|t| t.hr_final).sum::<f64>() / successes.len() as f64;
        (avg_n, avg_s, avg_f, if avg_f.ln().abs() > 0.001 { avg_s.ln() / avg_f.ln() } else { 1.0 })
    } else { (0.0, 1.0, 1.0, 1.0) };

    // Futility grid
    let thresholds = [0.1, 0.2, 0.3, 0.4, 0.5];
    let mut grid: Vec<(f64, usize, usize, f64)> = thresholds.iter()
        .map(|&th| (th, 0usize, 0usize, 0.0f64)).collect();

    for t in &trials {
        for (th, n_trig, n_e, sum_hr) in &mut grid {
            if t.min_wealth < *th {
                *n_trig += 1;
                if t.stop_n.is_some() { *n_e += 1; }
                *sum_hr += t.hr_final;
            }
        }
    }

    // Console output
    console.push_str(&format!("\n==========================================\n"));
    console.push_str(&format!("   RESULTS\n"));
    console.push_str(&format!("==========================================\n"));
    console.push_str(&format!("Type I Error:  {:.2}%\n", type1));
    console.push_str(&format!("\n--- Power at N={} (~{} events) ---\n", n_pts, expected_events));
    console.push_str(&format!("Log-rank:  {:.1}%\n", lr_power));
    console.push_str(&format!("e-RTs:     {:.1}%\n", power));
    console.push_str(&format!("e-RTu:     {:.1}%\n", agn_power));

    if !successes.is_empty() {
        console.push_str(&format!("\n--- Stopping ---\n"));
        console.push_str(&format!("Avg stop:      {:.0} ({:.0}%)\n", avg_stop, avg_stop / n_pts as f64 * 100.0));
        console.push_str(&format!("HR @ stop:     {:.3}\n", avg_hr_stop));
        console.push_str(&format!("HR @ end:      {:.3}\n", avg_hr_final));
        console.push_str(&format!("Type M:        {:.2}x\n", type_m));
    }

    console.push_str(&format!("\n--- Futility Grid ---\n"));
    console.push_str(&format!("{:<10} {:>10} {:>10} {:>10}\n", "Threshold", "Triggered", "e-RTs+", "Avg HR"));
    for (th, n_trig, n_e, sum_hr) in &grid {
        if *n_trig > 0 {
            let trig_pct = (*n_trig as f64 / n_sims as f64) * 100.0;
            let e_pct = (*n_e as f64 / *n_trig as f64) * 100.0;
            let avg_hr = sum_hr / *n_trig as f64;
            console.push_str(&format!("{:<10.1} {:>9.1}% {:>9.1}% {:>10.3}\n", th, trig_pct, e_pct, avg_hr));
        }
    }

    print!("{}", console);

    // Generate report
    println!("\nGenerating report...");
    let stops: Vec<f64> = successes.iter().map(|t| t.stop_n.unwrap() as f64).collect();
    let min_wealths: Vec<f64> = trials.iter().map(|t| t.min_wealth).collect();

    let html = build_report(&console, n_pts, threshold, fut_watch,
        &steps, &y_lo, &y_med, &y_hi, &sample_trajs, &stops, &min_wealths, &grid);

    let mut file = File::create("survival_report.html").unwrap();
    file.write_all(html.as_bytes()).unwrap();
    println!("\n>> survival_report.html");
}
