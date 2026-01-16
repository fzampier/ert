use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::RngCore;
use std::io::{self, Write};
use std::fs::File;

use crate::ert_core::{
    get_input, get_input_usize, get_bool, get_optional_input,
    calculate_n_continuous, chrono_lite, t_test_power_continuous,
    MADProcess,
};
use crate::agnostic::{AgnosticERT, Signal, Arm};

struct Trial {
    stop_n: Option<usize>,
    effect_at_stop: f64,
    effect_final: f64,
    agnostic_stopped: bool,
}

fn sample_normal<R: Rng + ?Sized>(rng: &mut R, mean: f64, sd: f64) -> f64 {
    let u1: f64 = rng.gen();
    let u2: f64 = rng.gen();
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    mean + z * sd
}

// === MAIN ===

pub fn run() {
    println!("\n==========================================");
    println!("   e-RTc SIMULATION (Continuous)");
    println!("==========================================\n");

    let mu_ctrl: f64 = get_input("Control Mean (e.g., 10): ");
    let mu_trt: f64 = get_input("Treatment Mean (e.g., 12): ");
    let sd: f64 = get_input("Standard Deviation (e.g., 8): ");

    let cohen_d = ((mu_trt - mu_ctrl) / sd).abs();

    let n_pts = if get_bool("Calculate Sample Size automatically?") {
        let power: f64 = get_input("Target Power (e.g., 0.80): ");
        let n = calculate_n_continuous(cohen_d, power);
        println!("\nFrequentist N (Power {:.0}%, d={:.2}): {}", power * 100.0, cohen_d, n);
        if get_bool("Add buffer?") {
            let buf: f64 = get_input("Buffer percentage (e.g., 15): ");
            let buffered = (n as f64 * (1.0 + buf / 100.0)).ceil() as usize;
            println!("Buffered N: {}", buffered);
            buffered
        } else { n }
    } else {
        get_input_usize("Enter Number of Patients: ")
    };

    let n_sims = get_input_usize("Number of simulations (e.g., 2000): ");
    println!("\nSuccess threshold (1/alpha). Default = 20");
    let threshold: f64 = get_input("Success threshold: ");
    let seed = get_optional_input("Seed (Enter for random): ");

    let burn_in: usize = 20;
    let ramp: usize = 50;
    let c_max: f64 = 0.6;

    println!("\n--- Configuration ---");
    println!("mu_ctrl={:.2} mu_trt={:.2} SD={:.2} d={:.2} N={} sims={} thresh={}",
        mu_ctrl, mu_trt, sd, cohen_d, n_pts, n_sims, threshold);

    let mut rng: Box<dyn RngCore> = match seed {
        Some(s) => Box::new(StdRng::seed_from_u64(s)),
        None => Box::new(rand::thread_rng()),
    };

    // === PHASE 1: TYPE I ERROR ===
    print!("\nPhase 1: Type I Error... ");
    io::stdout().flush().unwrap();

    let mut null_rej = 0u32;
    for _ in 0..n_sims {
        let mut proc = MADProcess::new(burn_in, ramp, c_max);
        for i in 1..=n_pts {
            let is_trt = rng.gen_bool(0.5);
            let outcome = sample_normal(&mut *rng, mu_ctrl, sd); // Both arms same under null
            proc.update(i, outcome, is_trt);
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
    let alpha = 1.0 / threshold;

    let step = 5;
    let n_points = n_pts / step + 1;
    let mut all_wealth_at_step: Vec<Vec<f64>> = vec![Vec::with_capacity(n_sims); n_points];

    let progress_interval = 50;

    for sim in 0..n_sims {
        if sim > 0 && sim % progress_interval == 0 {
            let pct = sim as f64 / n_sims as f64 * 100.0;
            print!("\r  [{:>5.1}%] {}/{}     ", pct, sim, n_sims);
            io::stdout().flush().unwrap();
        }

        let mut proc = MADProcess::new(burn_in, ramp, c_max);
        let mut agnostic = AgnosticERT::new(burn_in, ramp, threshold);
        let (mut stopped, mut stop_n, mut effect_stop) = (false, None, 0.0);
        let mut agnostic_stopped = false;
        let mut all_outcomes: Vec<f64> = Vec::with_capacity(n_pts);
        let need_traj = pos_trajs.len() < 30 || neg_trajs.len() < 30;
        let mut traj = if need_traj { Vec::with_capacity(n_points) } else { Vec::new() };

        if need_traj { traj.push(1.0); }
        all_wealth_at_step[0].push(1.0);

        for i in 1..=n_pts {
            let is_trt = rng.gen_bool(0.5);
            let mu = if is_trt { mu_trt } else { mu_ctrl };
            let outcome = sample_normal(&mut *rng, mu, sd);
            all_outcomes.push(outcome);

            proc.update(i, outcome, is_trt);

            if i % step == 0 {
                all_wealth_at_step[i / step].push(proc.wealth);
                if need_traj { traj.push(proc.wealth); }
            }

            // Agnostic with running median
            if !agnostic_stopped {
                let running_med = if all_outcomes.len() > 1 {
                    let mut sorted = all_outcomes.clone();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    sorted[sorted.len() / 2]
                } else { outcome };
                let signal = Signal { arm: if is_trt { Arm::Treatment } else { Arm::Control }, good: outcome > running_med };
                if agnostic.observe(signal) { agnostic_stopped = true; }
            }

            if !stopped && proc.wealth > threshold {
                stopped = true;
                stop_n = Some(i);
                effect_stop = proc.current_effect(sd);
            }
        }

        // Collect positive/negative trajectories separately
        if need_traj {
            if stop_n.is_some() {
                if pos_trajs.len() < 30 { pos_trajs.push(traj); }
            } else {
                if neg_trajs.len() < 30 { neg_trajs.push(traj); }
            }
        }

        let effect_final = proc.current_effect(sd);
        trials.push(Trial { stop_n, effect_at_stop: effect_stop, effect_final, agnostic_stopped });
    }
    print!("\r  [100.0%] {}/{} complete          \n", n_sims, n_sims);
    io::stdout().flush().unwrap();

    // === STATISTICS ===
    let successes: Vec<&Trial> = trials.iter().filter(|t| t.stop_n.is_some()).collect();
    let n_success = successes.len();
    let power = n_success as f64 / n_sims as f64 * 100.0;
    let agn_power = trials.iter().filter(|t| t.agnostic_stopped).count() as f64 / n_sims as f64 * 100.0;
    let t_power = t_test_power_continuous(mu_trt - mu_ctrl, sd, n_pts, alpha) * 100.0;

    let (avg_stop, avg_eff_stop, avg_eff_final, type_m) = if n_success > 0 {
        let sum_n: f64 = successes.iter().map(|t| t.stop_n.unwrap() as f64).sum();
        let sum_s: f64 = successes.iter().map(|t| t.effect_at_stop.abs()).sum();
        let sum_f: f64 = successes.iter().map(|t| t.effect_final.abs()).sum();
        let n = n_success as f64;
        (sum_n / n, sum_s / n, sum_f / n, sum_s / sum_f)
    } else { (0.0, 0.0, 0.0, 1.0) };

    // === CONSOLE OUTPUT ===
    let mut out = String::with_capacity(2048);
    out.push_str(&format!("{}\n", chrono_lite()));
    out.push_str("==========================================\n");
    out.push_str("   PARAMETERS\n");
    out.push_str("==========================================\n");
    out.push_str(&format!("Control:     {:.2}\n", mu_ctrl));
    out.push_str(&format!("Treatment:   {:.2}\n", mu_trt));
    out.push_str(&format!("SD:          {:.2}\n", sd));
    out.push_str(&format!("Cohen's d:   {:.2}\n", cohen_d));
    out.push_str(&format!("N:           {}\n", n_pts));
    out.push_str(&format!("Simulations: {}\n", n_sims));
    out.push_str(&format!("Threshold:   {} (α={:.3})\n", threshold, alpha));
    out.push_str(&format!("Seed:        {}\n", seed.map_or("random".into(), |s| s.to_string())));

    out.push_str("\n==========================================\n");
    out.push_str("   RESULTS\n");
    out.push_str("==========================================\n");
    out.push_str(&format!("Type I Error:  {:.2}%\n\n", type1));
    out.push_str(&format!("--- Power at N={} ---\n", n_pts));
    out.push_str(&format!("t-test:    {:.1}%\n", t_power));
    out.push_str(&format!("e-RTc:     {:.1}%\n", power));
    out.push_str(&format!("e-RTu:     {:.1}%\n", agn_power));

    if n_success > 0 {
        out.push_str("\n--- Stopping ---\n");
        out.push_str(&format!("Avg stop:      {:.0} ({:.0}%)\n", avg_stop, avg_stop / n_pts as f64 * 100.0));
        out.push_str(&format!("Effect @ stop: {:.2}\n", avg_eff_stop));
        out.push_str(&format!("Effect @ end:  {:.2}\n", avg_eff_final));
        out.push_str(&format!("Type M:        {:.2}x\n", type_m));
    }

    print!("\n{}", out);
    println!("Generating report...");

    // === PREPARE PLOT DATA ===
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

    // Build representative sample: proportion of positive samples matches power
    let n_pos_sample = ((power / 100.0) * 30.0).round() as usize;
    let n_neg_sample = 30 - n_pos_sample;
    let mut sample_trajs: Vec<Vec<f64>> = Vec::new();
    sample_trajs.extend(pos_trajs.into_iter().take(n_pos_sample));
    sample_trajs.extend(neg_trajs.into_iter().take(n_neg_sample));

    let html = build_report(&out, threshold, n_pts, &x_pts, &y_lo, &y_med, &y_hi, &sample_trajs, &stop_times);

    File::create("continuous_report.html").unwrap().write_all(html.as_bytes()).unwrap();
    println!("\n>> continuous_report.html");
}

fn build_report(
    console: &str, threshold: f64, n_pts: usize,
    x: &[usize], y_lo: &[f64], y_med: &[f64], y_hi: &[f64],
    trajs: &[Vec<f64>], stops: &[f64],
) -> String {
    let x_js = format!("{:?}", x);

    let mut traces = String::new();
    for tr in trajs.iter().take(30) {
        traces.push_str(&format!(
            "{{type:'scatter',mode:'lines',x:{},y:{:?},line:{{color:'rgba(100,100,100,0.25)',width:1}},showlegend:false}},",
            x_js, tr
        ));
    }

    let (stop_x, stop_y) = if !stops.is_empty() {
        let mut s = stops.to_vec();
        s.sort_by(|a,b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let y: Vec<f64> = (1..=s.len()).map(|i| i as f64 / s.len() as f64 * 100.0).collect();
        (format!("{:?}", s), format!("{:?}", y))
    } else { ("[]".into(), "[]".into()) };

    format!(r#"<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>e-RTc Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
body{{font-family:system-ui,-apple-system,sans-serif;max-width:900px;margin:0 auto;padding:20px;background:#fafafa}}
h1{{color:#1a1a2e}}h2,h3{{color:#16213e}}
pre{{background:#fff;padding:15px;border-radius:8px;border:1px solid #ddd;overflow-x:auto;font-size:13px}}
.plot{{background:#fff;border-radius:8px;padding:15px;margin:20px 0;box-shadow:0 1px 3px rgba(0,0,0,0.1)}}
</style></head><body>
<h1>e-RTc Report</h1>
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
