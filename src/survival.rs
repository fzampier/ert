use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::RngCore;
use std::io::{self, Write};
use std::fs::File;

use crate::agnostic::{AgnosticERT, Signal, Arm};

// === HELPERS ===

fn get_input(prompt: &str) -> f64 {
    loop {
        print!("{}", prompt);
        io::stdout().flush().unwrap();
        let mut buffer = String::new();
        match io::stdin().read_line(&mut buffer) {
            Ok(_) => match buffer.trim().parse::<f64>() {
                Ok(num) => return num,
                Err(_) => println!("Invalid number."),
            },
            Err(_) => println!("Error."),
        }
    }
}

fn get_input_usize(prompt: &str) -> usize {
    loop {
        print!("{}", prompt);
        io::stdout().flush().unwrap();
        let mut buffer = String::new();
        match io::stdin().read_line(&mut buffer) {
            Ok(_) => match buffer.trim().parse::<usize>() {
                Ok(num) => return num,
                Err(_) => println!("Invalid number."),
            },
            Err(_) => println!("Error."),
        }
    }
}

fn get_bool(prompt: &str) -> bool {
    loop {
        print!("{} (y/n): ", prompt);
        io::stdout().flush().unwrap();
        let mut buffer = String::new();
        io::stdin().read_line(&mut buffer).unwrap();
        match buffer.trim().to_lowercase().as_str() {
            "y" | "yes" => return true,
            "n" | "no" => return false,
            _ => println!("Please type 'y' or 'n'."),
        }
    }
}

fn get_optional_input(prompt: &str) -> Option<u64> {
    print!("{}", prompt);
    io::stdout().flush().unwrap();
    let mut buffer = String::new();
    io::stdin().read_line(&mut buffer).unwrap();
    let trimmed = buffer.trim();
    if trimmed.is_empty() {
        None
    } else {
        trimmed.parse::<u64>().ok()
    }
}

fn chrono_lite() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let secs = duration.as_secs();
    let days = secs / 86400;
    let years = 1970 + days / 365;
    let remaining_days = days % 365;
    let months = remaining_days / 30 + 1;
    let day = remaining_days % 30 + 1;
    let hours = (secs % 86400) / 3600;
    let mins = (secs % 3600) / 60;
    format!("{}-{:02}-{:02} {:02}:{:02} UTC", years, months, day, hours, mins)
}

fn calculate_n_survival(target_hr: f64, power: f64) -> usize {
    let z_alpha: f64 = 1.96;
    let z_beta: f64 = if power > 0.85 { 1.28 } else { 0.84 };
    let log_hr = target_hr.ln();
    let req_events = (4.0 * ((z_alpha + z_beta) / log_hr).powi(2)).ceil() as usize;
    req_events
}

/// Calculate log-rank test power given number of events
/// Uses Schoenfeld's formula: Power = Φ(|log(HR)| × √(d/4) - z_α)
fn log_rank_power(target_hr: f64, n_events: usize, alpha: f64) -> f64 {
    let z_alpha = z_from_alpha(alpha);
    let log_hr = target_hr.ln().abs();
    let z_effect = log_hr * (n_events as f64 / 4.0).sqrt();
    normal_cdf(z_effect - z_alpha)
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    let a1 =  0.254829592;
    let a2 = -0.284496736;
    let a3 =  1.421413741;
    let a4 = -1.453152027;
    let a5 =  1.061405429;
    let p  =  0.3275911;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

fn z_from_alpha(alpha: f64) -> f64 {
    if (alpha - 0.05).abs() < 0.001 { return 1.96; }
    if (alpha - 0.01).abs() < 0.001 { return 2.576; }
    1.96 // default
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
    println!("e-survival:          {:.1}%  <- domain-aware sequential", e_surv_power);
    println!("Agnostic (universal):{:.1}%  <- floor (universal)", agnostic_power);
    println!("\nDomain knowledge:    {:+.1}%", e_surv_power - agnostic_power);
    println!("Sequential cost:     -{:.1}%", lr_power - e_surv_power);

    if success_count > 0 {
        println!("\n--- Stopping Analysis ---");
        println!("e-surv Avg Stop:     {:.0} events ({:.0}% of N)", avg_stop_event, (avg_stop_event / n_patients as f64) * 100.0);
        if !agnostic_stop_events.is_empty() {
            println!("Agnostic Avg Stop:   {:.0} events ({:.0}% of N)", agnostic_avg_stop, (agnostic_avg_stop / n_patients as f64) * 100.0);
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
    target_hr: f64, n_patients: usize, n_sims: usize,
    success_threshold: f64, futility_watch: f64,
    shape: f64, scale: f64, cens_prop: f64,
    burn_in: usize, ramp: usize, lambda_max: f64,
    seed: Option<u64>, run_futility: bool,
    type1_error: f64, success_count: usize, no_stop_count: usize,
    avg_stop_event: f64, avg_hr_stop: f64, avg_hr_final: f64, type_m_error: f64,
    futility_stats: Option<(usize, f64, f64, f64, f64, f64, usize)>,
    trajectories: &[Vec<f64>], stop_events: &[f64], required_hrs: &[f64],
    log_rank_power: f64, agnostic_power: f64, agnostic_avg_stop: f64,
) -> String {
    let timestamp = chrono_lite();
    let seed_str = seed.map_or("random".to_string(), |s| s.to_string());

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
    let req_hrs_json = format!("{:?}", required_hrs);

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

    let futility_html = if run_futility {
        if let Some((n_trig, med_event, med_hr, q25, q50, q75, trig_success)) = futility_stats {
            format!(r#"
        <h2>Futility Analysis</h2>
        <table>
            <tr><td>Trials triggering (wealth &lt; {:.1}):</td><td><strong>{} ({:.1}%)</strong></td></tr>
            <tr><td>Median event at trigger:</td><td>{:.0} ({:.0}% of N)</td></tr>
            <tr><td>Median required HR:</td><td>{:.3}</td></tr>
            <tr><td>Design HR:</td><td>{:.3}</td></tr>
            <tr><td>Ratio (Required/Design) - 25th pctl:</td><td>{:.2}x</td></tr>
            <tr><td>Ratio (Required/Design) - Median:</td><td>{:.2}x</td></tr>
            <tr><td>Ratio (Required/Design) - 75th pctl:</td><td>{:.2}x</td></tr>
            <tr><td>Triggered trials that succeeded:</td><td>{} ({:.1}%)</td></tr>
        </table>
            "#,
                futility_watch, n_trig, (n_trig as f64 / n_sims as f64) * 100.0,
                med_event, (med_event / n_patients as f64) * 100.0,
                med_hr, target_hr,
                q25, q50, q75,
                trig_success, (trig_success as f64 / n_trig as f64) * 100.0)
        } else {
            "<h2>Futility Analysis</h2><p>No trials triggered futility watch.</p>".to_string()
        }
    } else {
        String::new()
    };

    let req_hr_plot = if run_futility && !required_hrs.is_empty() {
        format!(r#"
        <h3>Required HR Distribution (at Futility Trigger)</h3>
        <div id="plot4" style="width:100%;height:350px;"></div>
        <script>
            Plotly.newPlot('plot4', [{{
                type: 'histogram',
                x: {},
                marker: {{color: 'steelblue'}}
            }}], {{
                shapes: [{{type:'line',x0:{:.3},x1:{:.3},y0:0,y1:1,yref:'paper',line:{{color:'red',width:2,dash:'dash'}}}}],
                xaxis: {{title: 'Required HR'}},
                yaxis: {{title: 'Count'}},
                annotations: [{{x:{:.3},y:1,yref:'paper',text:'Design HR',showarrow:false,font:{{color:'red'}}}}]
            }});
        </script>
        "#, req_hrs_json, target_hr, target_hr, target_hr)
    } else {
        String::new()
    };

    format!(r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>e-Survival Simulation Report</title>
    <script src="https://cdn.plot.ly/plotly-2.12.1.min.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        .container {{ background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        h3 {{ color: #7f8c8d; }}
        table {{ border-collapse: collapse; margin: 15px 0; }}
        td {{ padding: 8px 16px; border-bottom: 1px solid #eee; }}
        td:first-child {{ color: #7f8c8d; }}
        .highlight {{ background: #e8f4f8; font-weight: bold; }}
        .timestamp {{ color: #95a5a6; font-size: 0.9em; }}
        pre {{ background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>e-Survival Simulation Report</h1>
        <p class="timestamp">Generated: {}</p>
        
        <h2>Parameters</h2>
        <table>
            <tr><td>Target HR:</td><td><strong>{:.2}</strong></td></tr>
            <tr><td>Sample Size (N):</td><td>{}</td></tr>
            <tr><td>Simulations:</td><td>{}</td></tr>
            <tr><td>Success Threshold (1/α):</td><td>{}</td></tr>
            <tr><td>Futility Watch:</td><td>{}</td></tr>
            <tr><td>Weibull Shape:</td><td>{}</td></tr>
            <tr><td>Weibull Scale:</td><td>{}</td></tr>
            <tr><td>Censoring:</td><td>{:.1}%</td></tr>
            <tr><td>burn_in:</td><td>{}</td></tr>
            <tr><td>ramp:</td><td>{}</td></tr>
            <tr><td>lambda_max:</td><td>{}</td></tr>
            <tr><td>Seed:</td><td>{}</td></tr>
        </table>

        <h2>Equations</h2>
        <pre>
pⱼ = Y₁(tⱼ) / (Y₁(tⱼ) + Y₀(tⱼ))       # null probability (risk set proportion)
Uⱼ = Xⱼ - pⱼ                           # score (observed - expected)
Zⱼ = Σ Uₖ                              # cumulative score
λⱼ = sign(Zⱼ₋₁) × c × λmax            # betting fraction
Wⱼ = Wⱼ₋₁ × (1 + λⱼ × Uⱼ)             # wealth update (events only)
        </pre>

        <h2>Operating Characteristics</h2>
        <table>
            <tr class="highlight"><td>Type I Error:</td><td>{:.2}%</td></tr>
            <tr class="highlight"><td>Power (Success Rate):</td><td>{:.1}%</td></tr>
            <tr><td>No Stop:</td><td>{} ({:.1}%)</td></tr>
        </table>

        <h2>Power Comparison (Three-Tier Hierarchy)</h2>
        <table>
            <tr style="background:#e8f8e8;"><td>Log-rank (fixed sample):</td><td><strong>{:.1}%</strong></td><td>← ceiling (traditional)</td></tr>
            <tr style="background:#fff8e8;"><td>e-survival (sequential):</td><td><strong>{:.1}%</strong></td><td>← domain-aware sequential</td></tr>
            <tr style="background:#f8e8e8;"><td>Agnostic (universal):</td><td><strong>{:.1}%</strong></td><td>← floor (universal)</td></tr>
            <tr><td>Domain knowledge value:</td><td>{:+.1}%</td><td>(e-survival − agnostic)</td></tr>
            <tr><td>Sequential cost:</td><td>−{:.1}%</td><td>(log-rank − e-survival)</td></tr>
        </table>
        <p><em>The hierarchy shows: Traditional fixed-sample test sets the ceiling, domain-aware sequential captures part of it, and agnostic provides the floor.</em></p>

        <h2>Success Analysis</h2>
        <table>
            <tr><td>Number of Successes:</td><td>{}</td></tr>
            <tr><td>Average Stopping Point:</td><td>{:.0} events ({:.0}% of N)</td></tr>
            <tr><td>HR at Stop:</td><td>{:.3}</td></tr>
            <tr><td>HR at End:</td><td>{:.3}</td></tr>
            <tr><td>Type M Error (Magnification):</td><td>{:.2}x</td></tr>
        </table>

        {}

        <h2>Visualizations</h2>
        
        <h3>e-Value Trajectories (Median with 95% CI)</h3>
        <div id="plot1" style="width:100%;height:450px;"></div>
        
        <h3>Sample Trajectories (30 runs)</h3>
        <div id="plot2" style="width:100%;height:450px;"></div>
        
        <h3>Stopping Events Distribution</h3>
        <div id="plot3" style="width:100%;height:350px;"></div>
        
        {}
    </div>

    <script>
        // Plot 1: Median + 95% CI
        Plotly.newPlot('plot1', [
            {{type:'scatter',mode:'lines',x:{},y:{},line:{{width:0}},showlegend:false}},
            {{type:'scatter',mode:'lines',x:{},y:{},fill:'tonexty',fillcolor:'rgba(0,100,0,0.3)',line:{{width:0}},showlegend:false}},
            {{type:'scatter',mode:'lines',x:{},y:{},line:{{color:'darkgreen',width:2.5}},name:'Median'}}
        ], {{
            yaxis: {{type:'log',title:'e-value'}},
            xaxis: {{title:'Number of Events'}},
            shapes: [
                {{type:'line',x0:0,x1:1,xref:'paper',y0:{},y1:{},line:{{color:'red',width:2,dash:'dash'}}}},
                {{type:'line',x0:0,x1:1,xref:'paper',y0:{},y1:{},line:{{color:'orange',width:1.5,dash:'dot'}}}}
            ]
        }});

        // Plot 2: Sample trajectories
        Plotly.newPlot('plot2', [
            {}
            {{type:'scatter',mode:'lines',x:[0,{}],y:[{},{}],line:{{color:'red',width:2,dash:'dash'}},name:'Success'}},
            {{type:'scatter',mode:'lines',x:[0,{}],y:[{},{}],line:{{color:'orange',width:1.5,dash:'dot'}},name:'Futility Watch'}}
        ], {{
            yaxis: {{type:'log',title:'e-value'}},
            xaxis: {{title:'Number of Events'}}
        }});

        // Plot 3: Stopping events
        Plotly.newPlot('plot3', [{{
            type: 'histogram',
            x: {},
            marker: {{color: 'darkgreen'}}
        }}], {{
            xaxis: {{title: 'Event Number at Stop'}},
            yaxis: {{title: 'Count'}}
        }});
    </script>
</body>
</html>"#,
        timestamp,
        target_hr, n_patients, n_sims,
        success_threshold, futility_watch,
        shape, scale, cens_prop * 100.0,
        burn_in, ramp, lambda_max, seed_str,
        type1_error, (success_count as f64 / n_sims as f64) * 100.0,
        no_stop_count, (no_stop_count as f64 / n_sims as f64) * 100.0,
        // Power comparison section
        log_rank_power * 100.0,
        (success_count as f64 / n_sims as f64) * 100.0, // e-survival power
        agnostic_power,
        (success_count as f64 / n_sims as f64) * 100.0 - agnostic_power, // domain knowledge value
        log_rank_power * 100.0 - (success_count as f64 / n_sims as f64) * 100.0, // sequential cost
        success_count,
        avg_stop_event, (avg_stop_event / n_patients as f64) * 100.0,
        avg_hr_stop, avg_hr_final, type_m_error,
        futility_html,
        req_hr_plot,
        x_json, low_json, x_json, up_json, x_json, med_json,
        success_threshold, success_threshold, futility_watch, futility_watch,
        sample_traces, max_len, success_threshold, success_threshold,
        max_len, futility_watch, futility_watch,
        stops_json
    )
}