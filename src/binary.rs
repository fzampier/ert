use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::RngCore;
use std::io::{self, Write};
use std::fs::File;

use crate::ert_core::{
    get_input, get_input_usize, get_bool, get_optional_input,
    calculate_n_binary, chrono_lite, BinaryERTProcess, z_test_power_binary,
};
use crate::agnostic::{AgnosticERT, Signal, Arm};

// --- Monte Carlo: Required Effect Size for Recovery ---
fn required_effect_for_success<R: Rng + ?Sized>(
    rng: &mut R,
    current_wealth: f64,
    n_remaining: usize,
    p_ctrl: f64,
    burn_in: usize,
    ramp: usize,
    mc_sims: usize,
) -> f64 {
    if n_remaining == 0 {
        return 1.0;
    }

    let mut low = 0.001;
    let mut high = 0.50;

    for _ in 0..6 {
        let mid = (low + high) / 2.0;
        let p_trt = (p_ctrl - mid).max(0.001);

        let mut successes = 0;
        for _ in 0..mc_sims {
            let mut wealth = current_wealth;
            let mut n_trt = 0.0;
            let mut events_trt = 0.0;
            let mut n_ctrl = 0.0;
            let mut events_ctrl = 0.0;

            for j in 1..=n_remaining {
                let is_trt = rng.gen_bool(0.5);
                let prob = if is_trt { p_trt } else { p_ctrl };
                let outcome = if rng.gen_bool(prob) { 1.0 } else { 0.0 };

                let rate_trt = if n_trt > 0.0 { events_trt / n_trt } else { 0.5 };
                let rate_ctrl = if n_ctrl > 0.0 { events_ctrl / n_ctrl } else { 0.5 };
                let delta_hat = rate_trt - rate_ctrl;

                if is_trt {
                    n_trt += 1.0;
                    if outcome == 1.0 { events_trt += 1.0; }
                } else {
                    n_ctrl += 1.0;
                    if outcome == 1.0 { events_ctrl += 1.0; }
                }

                if j > burn_in {
                    let num = ((j - burn_in) as f64).max(0.0);
                    let c_i = (num / ramp as f64).clamp(0.0, 1.0);

                    let lambda = if outcome == 1.0 {
                        0.5 + 0.5 * c_i * delta_hat
                    } else {
                        0.5 - 0.5 * c_i * delta_hat
                    };
                    let lambda = lambda.clamp(0.001, 0.999);
                    let multiplier = if is_trt { lambda / 0.5 } else { (1.0 - lambda) / 0.5 };
                    wealth *= multiplier;
                }

                if wealth >= 20.0 {
                    successes += 1;
                    break;
                }
            }
        }

        let success_rate = successes as f64 / mc_sims as f64;
        if success_rate < 0.5 {
            low = mid;
        } else {
            high = mid;
        }
    }

    (low + high) / 2.0
}

struct FutilityInfo {
    patient_number: usize,
    _wealth_at_trigger: f64,
    required_arr: f64,
    ratio_to_design: f64,
}

struct TrialResult {
    stopped_at: Option<usize>,
    success: bool,
    risk_diff_at_stop: Option<f64>,
    final_risk_diff: f64,
    futility_info: Option<FutilityInfo>,
}

pub fn run() {
    println!("\n==========================================");
    println!("   BINARY e-RT SIMULATION");
    println!("==========================================\n");

    // --- User Inputs ---
    let p_ctrl = get_input("Control Event Rate (e.g. 0.40): ");
    let p_trt  = get_input("Treatment Event Rate (e.g. 0.30): ");
    let design_arr = (p_ctrl - p_trt).abs();

    // Sample size
    let n_patients = if get_bool("Calculate Sample Size automatically?") {
        let mut power = get_input("Target Power (e.g. 0.80): ");
        if power >= 1.0 {
            println!("Power must be < 1.0. Capping at 0.99.");
            power = 0.99;
        }
        let freq_n = calculate_n_binary(p_ctrl, p_trt, power);
        println!("\nFrequentist N (Power {:.0}%): {}", power * 100.0, freq_n);

        if get_bool("Add buffer? (10-20% increase may improve e-process power)") {
            let buffer_pct = get_input("Buffer percentage (e.g. 15): ");
            let buffered = (freq_n as f64 * (1.0 + buffer_pct / 100.0)).ceil() as usize;
            println!("Buffered N: {}", buffered);
            buffered
        } else {
            freq_n
        }
    } else {
        get_input_usize("Enter Number of Patients: ")
    };

    // Simulation parameters
    let n_sims = get_input_usize("Number of simulations (e.g. 2000): ");

    // Success threshold
    println!("\nSuccess threshold (1/alpha). Default = 20 (alpha=0.05)");
    let success_threshold = get_input("Success threshold (e.g. 20): ");

    // Futility watch
    println!("\nFutility watch threshold. Default = 0.5");
    let futility_watch = get_input("Futility watch threshold (e.g. 0.5): ");

    // Futility analysis
    let run_futility = get_bool("Run futility analysis? (adds computation time)");

    // Seed
    let seed = get_optional_input("Seed for reproducibility (press Enter for random): ");

    // Fixed parameters (vanilla e-RT)
    let burn_in: usize = 50;
    let ramp: usize = 100;

    println!("\n--- Trial Design ---");
    println!("Control Rate:    {:.1}%", p_ctrl * 100.0);
    println!("Treatment Rate:  {:.1}%", p_trt * 100.0);
    println!("Design ARR:      {:.1}%", design_arr * 100.0);
    println!("Total N:         {}", n_patients);
    println!("Simulations:     {}", n_sims);
    println!("Success (1/α):   {}", success_threshold);
    println!("Futility Watch:  {}", futility_watch);
    println!("Burn-In:         {} (default)", burn_in);
    println!("Ramp:            {} (default)", ramp);
    if let Some(s) = seed {
        println!("Seed:            {}", s);
    } else {
        println!("Seed:            random");
    }

    // Initialize RNG
    let mut rng: Box<dyn RngCore> = match seed {
        Some(s) => Box::new(StdRng::seed_from_u64(s)),
        None => Box::new(rand::thread_rng()),
    };

    // === PHASE 1: TYPE I ERROR ===
    print!("\nPhase 1: Type I Error (Null)... ");
    io::stdout().flush().unwrap();
    let mut null_rejections = 0;

    for _ in 0..n_sims {
        let mut proc = BinaryERTProcess::new(burn_in, ramp);
        for i in 1..=n_patients {
            let is_trt = rng.gen_bool(0.5);
            let outcome = if rng.gen_bool(p_ctrl) { 1.0 } else { 0.0 };
            proc.update(i, outcome, is_trt);
            if proc.wealth > success_threshold {
                null_rejections += 1;
                break;
            }
        }
    }
    let type1_error = (null_rejections as f64 / n_sims as f64) * 100.0;
    println!("Done. Type I Error: {:.2}%", type1_error);

    // === PHASE 2: POWER & FUTILITY ANALYSIS (with Agnostic comparison) ===
    print!("Phase 2: Power Analysis");
    if run_futility { print!(" + Futility"); }
    print!(" + e-RTu");
    println!("...");
    io::stdout().flush().unwrap();

    let mut results: Vec<TrialResult> = Vec::with_capacity(n_sims);
    let mut trajectories: Vec<Vec<f64>> = vec![vec![0.0; n_patients + 1]; n_sims];

    // Agnostic e-RT tracking
    let mut agnostic_successes = 0;
    let mut agnostic_stop_times: Vec<usize> = Vec::new();

    // Store 30 sample trajectories for plotting
    let sample_indices: Vec<usize> = (0..30.min(n_sims)).collect();

    let pb_interval = (n_sims / 20).max(1);

    for sim in 0..n_sims {
        if sim % pb_interval == 0 {
            print!(".");
            io::stdout().flush().unwrap();
        }

        let mut proc = BinaryERTProcess::new(burn_in, ramp);
        let mut agnostic = AgnosticERT::new(burn_in, ramp, success_threshold);
        let mut stopped = false;
        let mut stop_step = None;
        let mut stop_diff = None;
        let mut futility_info: Option<FutilityInfo> = None;
        let mut agnostic_stopped = false;
        let mut agnostic_stop_step: Option<usize> = None;

        trajectories[sim][0] = 1.0;

        for i in 1..=n_patients {
            let is_trt = rng.gen_bool(0.5);
            let prob = if is_trt { p_trt } else { p_ctrl };
            let outcome = if rng.gen_bool(prob) { 1.0 } else { 0.0 };

            proc.update(i, outcome, is_trt);
            trajectories[sim][i] = proc.wealth;

            // Agnostic e-RT: translate binary outcome to signal
            // For binary: event (outcome=1) is "bad" (higher is worse)
            // Treatment should have FEWER events, so event=bad
            let signal = Signal {
                arm: if is_trt { Arm::Treatment } else { Arm::Control },
                good: outcome == 0.0, // no event = good
            };
            if !agnostic_stopped && agnostic.observe(signal) {
                agnostic_stopped = true;
                agnostic_stop_step = Some(i);
            }

            // Futility info (if enabled)
            if run_futility && futility_info.is_none() && proc.wealth < futility_watch && i > burn_in {
                let n_remaining = n_patients - i;
                let req_arr = required_effect_for_success(
                    &mut *rng,
                    proc.wealth,
                    n_remaining,
                    p_ctrl,
                    burn_in,
                    ramp,
                    50
                );
                futility_info = Some(FutilityInfo {
                    patient_number: i,
                    _wealth_at_trigger: proc.wealth,
                    required_arr: req_arr,
                    ratio_to_design: req_arr / design_arr,
                });
            }

            // Success (binary e-RT)
            if !stopped && proc.wealth > success_threshold {
                stopped = true;
                stop_step = Some(i);
                stop_diff = Some(proc.current_risk_diff());
            }
        }

        // Track agnostic results
        if agnostic_stopped {
            agnostic_successes += 1;
            if let Some(t) = agnostic_stop_step {
                agnostic_stop_times.push(t);
            }
        }

        results.push(TrialResult {
            stopped_at: stop_step,
            success: stopped,
            risk_diff_at_stop: stop_diff,
            final_risk_diff: proc.current_risk_diff(),
            futility_info,
        });
    }
    println!(" Done.");

    // === COMPUTE STATISTICS ===
    let success_count = results.iter().filter(|r| r.success).count();
    let no_stop_count = n_sims - success_count;

    let (avg_stop_n, avg_diff_stop, avg_diff_final, type_m_error) = if success_count > 0 {
        let successes: Vec<&TrialResult> = results.iter().filter(|r| r.success).collect();
        let avg_n: f64 = successes.iter()
            .map(|r| r.stopped_at.unwrap() as f64).sum::<f64>() / success_count as f64;
        let avg_stop: f64 = successes.iter()
            .map(|r| r.risk_diff_at_stop.unwrap().abs()).sum::<f64>() / success_count as f64;
        let avg_final: f64 = successes.iter()
            .map(|r| r.final_risk_diff.abs()).sum::<f64>() / success_count as f64;
        (avg_n, avg_stop, avg_final, avg_stop / avg_final)
    } else {
        (0.0, 0.0, 0.0, 0.0)
    };

    // Futility stats (if enabled)
    let futility_stats = if run_futility {
        let trials_with_info: Vec<&TrialResult> = results.iter()
            .filter(|r| r.futility_info.is_some()).collect();

        if !trials_with_info.is_empty() {
            let n_triggered = trials_with_info.len();

            let mut trigger_patients: Vec<f64> = trials_with_info.iter()
                .map(|r| r.futility_info.as_ref().unwrap().patient_number as f64).collect();
            trigger_patients.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let mut required_arrs: Vec<f64> = trials_with_info.iter()
                .map(|r| r.futility_info.as_ref().unwrap().required_arr).collect();
            required_arrs.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let mut ratios: Vec<f64> = trials_with_info.iter()
                .map(|r| r.futility_info.as_ref().unwrap().ratio_to_design).collect();
            ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let triggered_success = trials_with_info.iter().filter(|r| r.success).count();

            Some((
                n_triggered,
                trigger_patients[n_triggered / 2],
                required_arrs[n_triggered / 2],
                ratios[n_triggered / 4],
                ratios[n_triggered / 2],
                ratios[(n_triggered * 3) / 4],
                triggered_success,
            ))
        } else {
            None
        }
    } else {
        None
    };

    // === COMPUTE AGNOSTIC STATISTICS ===
    let agnostic_power = (agnostic_successes as f64 / n_sims as f64) * 100.0;
    let agnostic_avg_stop = if !agnostic_stop_times.is_empty() {
        agnostic_stop_times.iter().sum::<usize>() as f64 / agnostic_stop_times.len() as f64
    } else {
        0.0
    };

    // === TRADITIONAL POWER (Z-TEST) ===
    let alpha = 1.0 / success_threshold;
    let z_test_power = z_test_power_binary(p_ctrl, p_trt, n_patients, alpha) * 100.0;

    // === PRINT CONSOLE SUMMARY ===
    println!("\n==========================================");
    println!("   RESULTS");
    println!("==========================================");
    println!("Type I Error:    {:.2}%", type1_error);

    // Three-tier power comparison
    println!("\n--- Power Comparison at N={} ---", n_patients);
    println!("Z-test (fixed):      {:.1}%  <- ceiling (traditional)", z_test_power);
    println!("e-RT binary:         {:.1}%  <- domain-aware sequential", (success_count as f64/n_sims as f64)*100.0);
    println!("e-RTu:               {:.1}%  <- floor (universal)", agnostic_power);

    println!("\nDomain knowledge:    +{:.1}%", (success_count as f64/n_sims as f64)*100.0 - agnostic_power);
    println!("Sequential cost:     -{:.1}%", z_test_power - (success_count as f64/n_sims as f64)*100.0);

    if success_count > 0 {
        println!("\n--- Stopping Analysis ---");
        println!("e-RT Avg Stop:       {:.0} ({:.0}% of N)", avg_stop_n, (avg_stop_n / n_patients as f64) * 100.0);
        if !agnostic_stop_times.is_empty() {
            println!("e-RTu Avg Stop:      {:.0} ({:.0}% of N)", agnostic_avg_stop, (agnostic_avg_stop / n_patients as f64) * 100.0);
        }
        println!("Type M Error:        {:.2}x", type_m_error);
    }

    // === GENERATE HTML REPORT ===
    println!("\nGenerating report...");

    // Prepare trajectory data for plots
    let mut x_axis: Vec<usize> = Vec::new();
    let mut y_median: Vec<f64> = Vec::new();
    let mut y_lower: Vec<f64> = Vec::new();
    let mut y_upper: Vec<f64> = Vec::new();

    for i in 0..=n_patients {
        if i % 5 != 0 && i != n_patients { continue; }
        x_axis.push(i);
        let mut step_vals: Vec<f64> = trajectories.iter().map(|v| v[i]).collect();
        step_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

        y_lower.push(step_vals[(n_sims as f64 * 0.025) as usize]);
        y_median.push(step_vals[(n_sims as f64 * 0.50) as usize]);
        y_upper.push(step_vals[(n_sims as f64 * 0.975) as usize]);
    }

    // Sample trajectories (30 runs)
    let sample_trajectories: Vec<&Vec<f64>> = sample_indices.iter()
        .map(|&i| &trajectories[i]).collect();

    // Stopping times
    let stop_times: Vec<f64> = results.iter()
        .filter(|r| r.success)
        .map(|r| r.stopped_at.unwrap() as f64).collect();

    // Required ARRs (if futility enabled)
    let required_arrs: Vec<f64> = if run_futility {
        results.iter()
            .filter(|r| r.futility_info.is_some())
            .map(|r| r.futility_info.as_ref().unwrap().required_arr * 100.0)
            .collect()
    } else {
        Vec::new()
    };

    // Build HTML
    let html = build_html_report(
        // Parameters
        p_ctrl, p_trt, design_arr, n_patients, n_sims,
        success_threshold, futility_watch, burn_in, ramp, seed,
        // Results
        type1_error, success_count, no_stop_count,
        avg_stop_n, avg_diff_stop, avg_diff_final, type_m_error,
        futility_stats, run_futility,
        // Power comparison
        z_test_power, agnostic_power, agnostic_avg_stop,
        // Plot data
        &x_axis, &y_median, &y_lower, &y_upper,
        &sample_trajectories, &stop_times, &required_arrs,
    );

    let mut file = File::create("binary_report.html").unwrap();
    file.write_all(html.as_bytes()).unwrap();

    println!("\n>> Report saved: binary_report.html");
    println!("==========================================");
}

fn build_html_report(
    _p_ctrl: f64, _p_trt: f64, _design_arr: f64, n_patients: usize, n_sims: usize,
    success_threshold: f64, futility_watch: f64, _burn_in: usize, _ramp: usize, _seed: Option<u64>,
    type1_error: f64, success_count: usize, _no_stop_count: usize,
    _avg_stop_n: f64, _avg_diff_stop: f64, _avg_diff_final: f64, _type_m_error: f64,
    _futility_stats: Option<(usize, f64, f64, f64, f64, f64, usize)>,
    _run_futility: bool,
    z_test_power: f64, agnostic_power: f64, _agnostic_avg_stop: f64,
    x_axis: &[usize], y_median: &[f64], y_lower: &[f64], y_upper: &[f64],
    sample_trajectories: &[&Vec<f64>], stop_times: &[f64], _required_arrs: &[f64],
) -> String {
    let timestamp = chrono_lite();
    let power = (success_count as f64 / n_sims as f64) * 100.0;

    let x_json = format!("{:?}", x_axis);
    let median_json = format!("{:?}", y_median);
    let lower_json = format!("{:?}", y_lower);
    let upper_json = format!("{:?}", y_upper);
    let stop_times_json = format!("{:?}", stop_times);

    let mut sample_traces = String::new();
    for (idx, traj) in sample_trajectories.iter().enumerate() {
        let downsampled: Vec<f64> = traj.iter().enumerate()
            .filter(|(i, _)| i % 5 == 0 || *i == n_patients)
            .map(|(_, v)| *v).collect();
        let color = if idx == 0 { "rgba(100,100,100,0.4)" } else { "rgba(100,100,100,0.2)" };
        sample_traces.push_str(&format!(
            "{{type:'scatter',mode:'lines',x:{},y:{:?},line:{{color:'{}',width:1}},showlegend:false}},",
            x_json, downsampled, color
        ));
    }

    format!(r#"<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>e-RT Binary</title>
<script src="https://cdn.plot.ly/plotly-2.12.1.min.js"></script>
<style>body{{font-family:monospace;max-width:1200px;margin:0 auto;padding:20px}}pre{{background:#f5f5f5;padding:10px}}</style>
</head><body>
<h1>e-RT Binary</h1>
<pre>
{}
Type I: {:.2}%  |  Power: {:.1}%  |  Z-test: {:.1}%  |  e-RTu: {:.1}%
</pre>
<div id="p1" style="height:400px"></div>
<div id="p2" style="height:400px"></div>
<div id="p3" style="height:300px"></div>
<script>
Plotly.newPlot('p1',[
  {{type:'scatter',x:{},y:{},line:{{width:0}},showlegend:false}},
  {{type:'scatter',x:{},y:{},fill:'tonexty',fillcolor:'rgba(31,119,180,0.3)',line:{{width:0}},showlegend:false}},
  {{type:'scatter',x:{},y:{},line:{{color:'blue',width:2}},name:'Median'}}
],{{yaxis:{{type:'log',title:'e-value'}},xaxis:{{title:'Patients'}},
  shapes:[{{type:'line',x0:0,x1:1,xref:'paper',y0:{},y1:{},line:{{color:'green',width:2,dash:'dash'}}}}]}});
Plotly.newPlot('p2',[{}
  {{type:'scatter',x:[0,{}],y:[{},{}],line:{{color:'green',width:2,dash:'dash'}},name:'Threshold'}},
  {{type:'scatter',x:[0,{}],y:[{},{}],line:{{color:'orange',width:1,dash:'dot'}},name:'Futility'}}
],{{yaxis:{{type:'log',title:'e-value'}},xaxis:{{title:'Patients'}}}});
Plotly.newPlot('p3',[{{type:'histogram',x:{},marker:{{color:'steelblue'}}}}],{{xaxis:{{title:'Stop Time'}},yaxis:{{title:'Count'}}}});
</script></body></html>"#,
        timestamp, type1_error, power, z_test_power, agnostic_power,
        x_json, lower_json, x_json, upper_json, x_json, median_json,
        success_threshold, success_threshold,
        sample_traces, n_patients, success_threshold, success_threshold,
        n_patients, futility_watch, futility_watch,
        stop_times_json
    )
}
