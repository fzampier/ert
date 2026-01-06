use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::RngCore;
use std::fs::File;
use std::io::{self, Write};

use crate::ert_core::{get_input, get_input_usize, get_bool, get_optional_input, chrono_lite, normal_cdf};
use crate::agnostic::{AgnosticERT, Signal, Arm};

// === SURVIVAL-SPECIFIC HELPERS ===

fn calculate_n_survival(target_hr: f64, power: f64) -> usize {
    let z_alpha: f64 = 1.96;
    let z_beta: f64 = if power > 0.85 { 1.28 } else { 0.84 };
    let log_hr = target_hr.ln();
    (4.0 * ((z_alpha + z_beta) / log_hr).powi(2)).ceil() as usize
}

/// Calculate log-rank test power given number of events
fn log_rank_power(target_hr: f64, n_events: usize, alpha: f64) -> f64 {
    let z_alpha = if (alpha - 0.05).abs() < 0.001 { 1.96 } else { 2.576 };
    let log_hr = target_hr.ln().abs();
    let z_effect = log_hr * (n_events as f64 / 4.0).sqrt();
    normal_cdf(z_effect - z_alpha)
}

// === DATA STRUCTURES ===

struct SurvivalData {
    time: Vec<f64>,
    status: Vec<u8>,
    treatment: Vec<u8>,
}

struct FutilityInfo {
    event_number: usize,
    _wealth_at_trigger: f64,
    required_hr: f64,
    ratio_to_design: f64,
}

struct TrialResult {
    stopped_at: Option<usize>,
    success: bool,
    hr_at_stop: Option<f64>,
    final_hr: f64,
    futility_info: Option<FutilityInfo>,
}

// === SIMULATE SURVIVAL TRIAL (Weibull) ===

fn simulate_trial<R: Rng + ?Sized>(
    rng: &mut R,
    n: usize,
    hr: f64,
    shape: f64,
    scale: f64,
    cens_prop: f64,
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
                status[i] = 1;
            }
        } else {
            time[i] = true_time;
            status[i] = 1;
        }
    }

    SurvivalData { time, status, treatment }
}

// === COMPUTE e-SURVIVAL ===

fn compute_e_survival(
    data: &SurvivalData,
    burn_in: usize,
    ramp: usize,
    lambda_max: f64,
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
        } else {
            0.0
        };

        let total_risk = risk_trt + risk_ctrl;
        let p_null = if total_risk > 0 {
            risk_trt as f64 / total_risk as f64
        } else {
            0.5
        };

        if is_event {
            let obs = if is_trt { 1.0 } else { 0.0 };
            let u_i = obs - p_null;
            let multiplier = 1.0 + b_i * u_i;
            cumulative_z += u_i;

            if i > 0 {
                wealth[i] = wealth[i - 1] * multiplier;
            } else {
                wealth[i] = multiplier;
            }
        } else {
            if i > 0 {
                wealth[i] = wealth[i - 1];
            }
        }

        if is_trt {
            risk_trt = (risk_trt - 1).max(0);
        } else {
            risk_ctrl = (risk_ctrl - 1).max(0);
        }
    }

    wealth
}

// === COMPUTE AGNOSTIC e-RT FOR SURVIVAL ===

fn compute_agnostic_survival(
    data: &SurvivalData,
    burn_in: usize,
    ramp: usize,
    threshold: f64,
) -> (bool, Option<usize>) {
    let n = data.time.len();

    // Sort by event time
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| data.time[a].partial_cmp(&data.time[b]).unwrap());

    let mut agnostic = AgnosticERT::new(burn_in, ramp, threshold);

    // Process ALL patients (events AND censored) in time order
    // For survival, treat like binary: event=bad, no event=good
    // Same mapping for both arms (symmetric under null)
    // Treatment should have MORE good outcomes (fewer events) under H1
    for (i, &idx) in indices.iter().enumerate() {
        let is_event = data.status[idx] == 1;
        let is_trt = data.treatment[idx] == 1;

        // Symmetric signal: event = bad for both arms (like binary)
        let signal = Signal {
            arm: if is_trt { Arm::Treatment } else { Arm::Control },
            good: !is_event, // no event (censored/survived) = good
        };

        if agnostic.observe(signal) {
            return (true, Some(i + 1));
        }
    }

    (false, None)
}

// === CALCULATE OBSERVED HR ===

fn calculate_observed_hr(data: &SurvivalData, max_events: Option<usize>) -> f64 {
    let mut indices: Vec<usize> = (0..data.time.len()).collect();
    indices.sort_by(|&a, &b| data.time[a].partial_cmp(&data.time[b]).unwrap());

    let mut events_trt = 0.0;
    let mut events_ctrl = 0.0;
    let mut time_trt = 0.0;
    let mut time_ctrl = 0.0;
    let mut event_count = 0;

    for &idx in &indices {
        if data.treatment[idx] == 1 {
            time_trt += data.time[idx];
            if data.status[idx] == 1 {
                events_trt += 1.0;
                event_count += 1;
            }
        } else {
            time_ctrl += data.time[idx];
            if data.status[idx] == 1 {
                events_ctrl += 1.0;
                event_count += 1;
            }
        }

        if let Some(max) = max_events {
            if event_count >= max {
                break;
            }
        }
    }

    let rate_trt = if time_trt > 0.0 { events_trt / time_trt } else { 0.0 };
    let rate_ctrl = if time_ctrl > 0.0 { events_ctrl / time_ctrl } else { 0.0 };

    if rate_ctrl > 0.0 {
        rate_trt / rate_ctrl
    } else {
        1.0
    }
}

// === FUTILITY MONTE CARLO ===

fn required_hr_for_success<R: Rng + ?Sized>(
    rng: &mut R,
    current_wealth: f64,
    n_remaining: usize,
    shape: f64,
    scale: f64,
    cens_prop: f64,
    burn_in: usize,
    ramp: usize,
    lambda_max: f64,
    success_threshold: f64,
    mc_sims: usize,
) -> f64 {
    if n_remaining == 0 {
        return 0.5;
    }

    let mut low = 0.3;
    let mut high = 0.99;

    for _ in 0..6 {
        let mid = (low + high) / 2.0;

        let mut successes = 0;
        for _ in 0..mc_sims {
            let trial = simulate_trial(rng, n_remaining, mid, shape, scale, cens_prop);
            let wealth_vec = compute_e_survival(&trial, burn_in, ramp, lambda_max);

            let final_wealth = current_wealth * wealth_vec.last().unwrap_or(&1.0);
            if final_wealth >= success_threshold {
                successes += 1;
            }
        }

        let rate = successes as f64 / mc_sims as f64;
        if rate < 0.5 {
            high = mid;
        } else {
            low = mid;
        }
    }

    (low + high) / 2.0
}

// === MAIN SIMULATION ===

pub fn run() {
    println!("\n==========================================");
    println!("   e-SURVIVAL SIMULATION");
    println!("==========================================\n");

    let target_hr = get_input("Target HR (e.g., 0.80): ");

    let n_patients = if get_bool("Calculate Sample Size automatically?") {
        let power = get_input("Target Power (e.g., 0.80): ");
        let freq_n = calculate_n_survival(target_hr, power);
        println!("\nSchoenfeld N (Power {:.0}%, HR={:.2}): {}", power * 100.0, target_hr, freq_n);

        if get_bool("Add buffer?") {
            let buffer_pct = get_input("Buffer percentage (e.g., 15): ");
            let buffered = (freq_n as f64 * (1.0 + buffer_pct / 100.0)).ceil() as usize;
            println!("Buffered N: {}", buffered);
            buffered
        } else {
            freq_n
        }
    } else {
        get_input_usize("Enter Number of Patients: ")
    };

    let n_sims = get_input_usize("Number of simulations (e.g., 2000): ");

    println!("\nSuccess threshold (1/alpha). Default = 20");
    let success_threshold = get_input("Success threshold: ");

    println!("\nFutility watch threshold. Default = 0.5");
    let futility_watch = get_input("Futility watch: ");

    let run_futility = get_bool("Run futility analysis?");

    println!("\nWeibull parameters:");
    let shape = get_input("Shape (default 1.2): ");
    let scale = get_input("Scale (default 10): ");
    let cens_prop = get_input("Censoring proportion 0-1 (e.g., 0.05 for 5%, default 0): ");

    let seed = get_optional_input("Seed (press Enter for random): ");

    let burn_in: usize = 30;
    let ramp: usize = 50;
    let lambda_max: f64 = 0.25;

    println!("\n--- Configuration ---");
    println!("Target HR:       {:.2}", target_hr);
    println!("N:               {}", n_patients);
    println!("Simulations:     {}", n_sims);
    println!("Success (1/α):   {}", success_threshold);
    println!("Futility Watch:  {}", futility_watch);
    println!("Weibull Shape:   {}", shape);
    println!("Weibull Scale:   {}", scale);
    println!("Censoring:       {:.1}%", cens_prop * 100.0);
    println!("burn_in:         {} (default)", burn_in);
    println!("ramp:            {} (default)", ramp);
    println!("lambda_max:      {} (default)", lambda_max);

    let mut rng: Box<dyn RngCore> = match seed {
        Some(s) => Box::new(StdRng::seed_from_u64(s)),
        None => Box::new(rand::thread_rng()),
    };

    println!("\n==========================================");
    println!("   RUNNING SIMULATIONS");
    println!("==========================================\n");

    // === PHASE 1: TYPE I ERROR ===
    print!("Phase 1: Type I Error (HR=1.0)... ");
    io::stdout().flush().unwrap();

    let mut null_rejections = 0;
    for _ in 0..n_sims {
        let trial = simulate_trial(&mut *rng, n_patients, 1.0, shape, scale, cens_prop);
        let wealth = compute_e_survival(&trial, burn_in, ramp, lambda_max);
        if wealth.iter().any(|&w| w > success_threshold) {
            null_rejections += 1;
        }
    }
    let type1_error = (null_rejections as f64 / n_sims as f64) * 100.0;
    println!("Done. {:.2}%", type1_error);

    // === PHASE 2: POWER ===
    print!("Phase 2: Power (HR={:.2})", target_hr);
    if run_futility { print!(" + Futility"); }
    print!(" + Agnostic");
    println!("...");
    io::stdout().flush().unwrap();

    let mut results: Vec<TrialResult> = Vec::with_capacity(n_sims);
    let mut trajectories: Vec<Vec<f64>> = Vec::with_capacity(n_sims);
    let pb_interval = (n_sims / 20).max(1);

    // Agnostic tracking
    let mut agnostic_successes = 0;
    let mut agnostic_stop_events: Vec<usize> = Vec::new();

    for sim in 0..n_sims {
        if sim % pb_interval == 0 {
            print!(".");
            io::stdout().flush().unwrap();
        }

        let trial = simulate_trial(&mut *rng, n_patients, target_hr, shape, scale, cens_prop);
        let wealth = compute_e_survival(&trial, burn_in, ramp, lambda_max);

        // Run agnostic on the same trial
        let (agn_success, agn_stop) = compute_agnostic_survival(&trial, burn_in, ramp, success_threshold);
        if agn_success {
            agnostic_successes += 1;
            if let Some(e) = agn_stop {
                agnostic_stop_events.push(e);
            }
        }

        if trajectories.len() < 100 {
            trajectories.push(wealth.clone());
        }

        let mut stopped = false;
        let mut stop_step = None;
        let mut hr_at_stop = None;
        let mut futility_info: Option<FutilityInfo> = None;

        for (i, &w) in wealth.iter().enumerate() {
            if run_futility && futility_info.is_none() && w < futility_watch && i > burn_in {
                let n_remaining = n_patients.saturating_sub(i);
                let req_hr = required_hr_for_success(
                    &mut *rng,
                    w,
                    n_remaining,
                    shape,
                    scale,
                    cens_prop,
                    burn_in,
                    ramp,
                    lambda_max,
                    success_threshold,
                    50,
                );
                futility_info = Some(FutilityInfo {
                    event_number: i,
                    _wealth_at_trigger: w,
                    required_hr: req_hr,
                    ratio_to_design: req_hr / target_hr,
                });
            }

            if !stopped && w > success_threshold {
                stopped = true;
                stop_step = Some(i);
                hr_at_stop = Some(calculate_observed_hr(&trial, Some(i + 1)));
            }
        }

        let final_hr = calculate_observed_hr(&trial, None);

        results.push(TrialResult {
            stopped_at: stop_step,
            success: stopped,
            hr_at_stop,
            final_hr,
            futility_info,
        });
    }
    println!(" Done.");

    // === COMPUTE STATISTICS ===
    let success_count = results.iter().filter(|r| r.success).count();
    let no_stop_count = n_sims - success_count;

    let (avg_stop_event, avg_hr_stop, avg_hr_final, type_m_error) = if success_count > 0 {
        let successes: Vec<&TrialResult> = results.iter().filter(|r| r.success).collect();
        let avg_event = successes.iter().map(|r| r.stopped_at.unwrap() as f64).sum::<f64>() / success_count as f64;
        let avg_hr_s = successes.iter().map(|r| r.hr_at_stop.unwrap()).sum::<f64>() / success_count as f64;
        let avg_hr_f = successes.iter().map(|r| r.final_hr).sum::<f64>() / success_count as f64;
        let type_m = avg_hr_s.ln() / avg_hr_f.ln();
        (avg_event, avg_hr_s, avg_hr_f, type_m)
    } else {
        (0.0, 1.0, 1.0, 1.0)
    };

    let futility_stats = if run_futility {
        let trials_with_info: Vec<&TrialResult> = results.iter().filter(|r| r.futility_info.is_some()).collect();
        if !trials_with_info.is_empty() {
            let n_trig = trials_with_info.len();
            let mut events: Vec<f64> = trials_with_info.iter().map(|r| r.futility_info.as_ref().unwrap().event_number as f64).collect();
            events.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mut hrs: Vec<f64> = trials_with_info.iter().map(|r| r.futility_info.as_ref().unwrap().required_hr).collect();
            hrs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mut ratios: Vec<f64> = trials_with_info.iter().map(|r| r.futility_info.as_ref().unwrap().ratio_to_design).collect();
            ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let trig_success = trials_with_info.iter().filter(|r| r.success).count();
            Some((n_trig, events[n_trig / 2], hrs[n_trig / 2], ratios[n_trig / 4], ratios[n_trig / 2], ratios[(n_trig * 3) / 4], trig_success))
        } else {
            None
        }
    } else {
        None
    };

    let stop_events: Vec<f64> = results.iter().filter(|r| r.success).map(|r| r.stopped_at.unwrap() as f64).collect();
    let required_hrs: Vec<f64> = if run_futility {
        results.iter().filter(|r| r.futility_info.is_some()).map(|r| r.futility_info.as_ref().unwrap().required_hr).collect()
    } else {
        Vec::new()
    };

    // === COMPUTE AGNOSTIC STATS ===
    let agnostic_power = (agnostic_successes as f64 / n_sims as f64) * 100.0;
    let agnostic_avg_stop = if !agnostic_stop_events.is_empty() {
        agnostic_stop_events.iter().sum::<usize>() as f64 / agnostic_stop_events.len() as f64
    } else {
        0.0
    };

    // === CONSOLE OUTPUT ===
    println!("\n==========================================");
    println!("   RESULTS");
    println!("==========================================");
    println!("Type I Error:    {:.2}%", type1_error);

    // Estimate expected events for log-rank power calculation
    // With censoring proportion c, expect roughly N*(1-c) events
    let expected_events = (n_patients as f64 * (1.0 - cens_prop)).round() as usize;
    let alpha = 1.0 / success_threshold;
    let lr_power = log_rank_power(target_hr, expected_events, alpha) * 100.0;

    // Three-tier power comparison
    let e_surv_power = (success_count as f64 / n_sims as f64) * 100.0;
    println!("\n--- Power Comparison at N={} (~{} events) ---", n_patients, expected_events);
    println!("Log-rank (fixed):    {:.1}%  <- ceiling (traditional)", lr_power);
    println!("e-RTs:               {:.1}%  <- domain-aware sequential", e_surv_power);
    println!("e-RTu:               {:.1}%  <- floor (universal)", agnostic_power);
    println!("\nDomain knowledge:    {:+.1}%", e_surv_power - agnostic_power);
    println!("Sequential cost:     -{:.1}%", lr_power - e_surv_power);

    if success_count > 0 {
        println!("\n--- Stopping Analysis ---");
        println!("e-RTs Avg Stop:      {:.0} events ({:.0}% of N)", avg_stop_event, (avg_stop_event / n_patients as f64) * 100.0);
        if !agnostic_stop_events.is_empty() {
            println!("e-RTu Avg Stop:      {:.0} events ({:.0}% of N)", agnostic_avg_stop, (agnostic_avg_stop / n_patients as f64) * 100.0);
        }
        println!("HR @ Stop:           {:.3}", avg_hr_stop);
        println!("HR @ End:            {:.3}", avg_hr_final);
        println!("Type M Error:        {:.2}x", type_m_error);
    }

    // === GENERATE HTML REPORT ===
    println!("\nGenerating report...");

    let html = build_html_report(
        target_hr, n_patients, n_sims,
        success_threshold, futility_watch,
        shape, scale, cens_prop,
        burn_in, ramp, lambda_max,
        seed, run_futility,
        type1_error, success_count, no_stop_count,
        avg_stop_event, avg_hr_stop, avg_hr_final, type_m_error,
        futility_stats,
        &trajectories, &stop_events, &required_hrs,
        lr_power / 100.0, agnostic_power, agnostic_avg_stop,
    );

    let mut file = File::create("survival_report.html").unwrap();
    file.write_all(html.as_bytes()).unwrap();

    println!("\n>> Report saved: survival_report.html");
    println!("==========================================");
}

fn build_html_report(
    _target_hr: f64, _n_patients: usize, n_sims: usize,
    success_threshold: f64, futility_watch: f64,
    _shape: f64, _scale: f64, _cens_prop: f64,
    _burn_in: usize, _ramp: usize, _lambda_max: f64,
    _seed: Option<u64>, _run_futility: bool,
    type1_error: f64, success_count: usize, _no_stop_count: usize,
    _avg_stop_event: f64, _avg_hr_stop: f64, _avg_hr_final: f64, _type_m_error: f64,
    _futility_stats: Option<(usize, f64, f64, f64, f64, f64, usize)>,
    trajectories: &[Vec<f64>], stop_events: &[f64], _required_hrs: &[f64],
    log_rank_power: f64, agnostic_power: f64, _agnostic_avg_stop: f64,
) -> String {
    let timestamp = chrono_lite();
    let power = (success_count as f64 / n_sims as f64) * 100.0;

    let max_len = trajectories.iter().map(|t| t.len()).max().unwrap_or(0);
    let mut x_axis: Vec<usize> = Vec::new();
    let mut y_median: Vec<f64> = Vec::new();
    let mut y_lower: Vec<f64> = Vec::new();
    let mut y_upper: Vec<f64> = Vec::new();

    for i in (0..max_len).step_by(5) {
        x_axis.push(i);
        let mut vals: Vec<f64> = trajectories.iter()
            .filter_map(|t| t.get(i).copied())
            .collect();
        if vals.is_empty() { continue; }
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = vals.len();
        y_lower.push(vals[(n as f64 * 0.025) as usize]);
        y_median.push(vals[n / 2]);
        y_upper.push(vals[(n as f64 * 0.975) as usize]);
    }

    let x_json = format!("{:?}", x_axis);
    let med_json = format!("{:?}", y_median);
    let low_json = format!("{:?}", y_lower);
    let up_json = format!("{:?}", y_upper);
    let stops_json = format!("{:?}", stop_events);

    let mut sample_traces = String::new();
    for idx in 0..30.min(trajectories.len()) {
        let ds: Vec<f64> = trajectories[idx].iter().enumerate()
            .filter(|(i, _)| i % 5 == 0)
            .map(|(_, v)| *v).collect();
        let ds_x: Vec<usize> = (0..trajectories[idx].len()).step_by(5).collect();
        sample_traces.push_str(&format!(
            "{{type:'scatter',mode:'lines',x:{:?},y:{:?},line:{{color:'rgba(0,100,0,0.3)',width:1}},showlegend:false}},",
            ds_x, ds
        ));
    }

    format!(r#"<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>e-RTs Survival</title>
<script src="https://cdn.plot.ly/plotly-2.12.1.min.js"></script>
<style>body{{font-family:monospace;max-width:1200px;margin:0 auto;padding:20px}}pre{{background:#f5f5f5;padding:10px}}</style>
</head><body>
<h1>e-RTs Survival</h1>
<pre>
{}
Type I: {:.2}%  |  Power: {:.1}%  |  Log-rank: {:.1}%  |  e-RTu: {:.1}%
</pre>
<div id="p1" style="height:400px"></div>
<div id="p2" style="height:400px"></div>
<div id="p3" style="height:300px"></div>
<script>
Plotly.newPlot('p1',[
  {{type:'scatter',x:{},y:{},line:{{width:0}},showlegend:false}},
  {{type:'scatter',x:{},y:{},fill:'tonexty',fillcolor:'rgba(0,100,0,0.3)',line:{{width:0}},showlegend:false}},
  {{type:'scatter',x:{},y:{},line:{{color:'darkgreen',width:2}},name:'Median'}}
],{{yaxis:{{type:'log',title:'e-value'}},xaxis:{{title:'Events'}},
  shapes:[{{type:'line',x0:0,x1:1,xref:'paper',y0:{},y1:{},line:{{color:'green',width:2,dash:'dash'}}}}]}});
Plotly.newPlot('p2',[{}
  {{type:'scatter',x:[0,{}],y:[{},{}],line:{{color:'green',width:2,dash:'dash'}},name:'Threshold'}},
  {{type:'scatter',x:[0,{}],y:[{},{}],line:{{color:'orange',width:1,dash:'dot'}},name:'Futility'}}
],{{yaxis:{{type:'log',title:'e-value'}},xaxis:{{title:'Events'}}}});
Plotly.newPlot('p3',[{{type:'histogram',x:{},marker:{{color:'steelblue'}}}}],{{xaxis:{{title:'Stop Event'}},yaxis:{{title:'Count'}}}});
</script></body></html>"#,
        timestamp, type1_error, power, log_rank_power * 100.0, agnostic_power,
        x_json, low_json, x_json, up_json, x_json, med_json,
        success_threshold, success_threshold,
        sample_traces, max_len, success_threshold, success_threshold,
        max_len, futility_watch, futility_watch,
        stops_json
    )
}