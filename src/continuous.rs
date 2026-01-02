use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::RngCore;
use std::io::{self, Write};
use std::fs::File;

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

fn get_choice(prompt: &str, options: &[&str]) -> usize {
    loop {
        println!("{}", prompt);
        for (i, opt) in options.iter().enumerate() {
            println!("  {}. {}", i + 1, opt);
        }
        print!("Select: ");
        io::stdout().flush().unwrap();
        let mut buffer = String::new();
        if io::stdin().read_line(&mut buffer).is_ok() {
            if let Ok(num) = buffer.trim().parse::<usize>() {
                if num >= 1 && num <= options.len() {
                    return num;
                }
            }
        }
        println!("Invalid choice.");
    }
}

fn median(data: &[f64]) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 0 {
        (sorted[n/2 - 1] + sorted[n/2]) / 2.0
    } else {
        sorted[n/2]
    }
}

fn mad(data: &[f64]) -> f64 {
    let med = median(data);
    let deviations: Vec<f64> = data.iter().map(|x| (x - med).abs()).collect();
    median(&deviations)
}

fn calculate_n_continuous(cohen_d: f64, power: f64) -> usize {
    let z_alpha: f64 = 1.96;
    let z_beta: f64 = if power > 0.85 { 1.28 } else { 0.84 };
    let n_per_arm = (2.0 * ((z_alpha + z_beta) / cohen_d).powi(2)).ceil() as usize;
    2 * n_per_arm
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

// === STRUCTS ===

#[derive(Clone, Copy, PartialEq)]
enum Method {
    LinearERT,
    MAD,
}

struct FutilityInfo {
    patient_number: usize,
    wealth_at_trigger: f64,
    required_effect: f64,
    ratio_to_design: f64,
}

struct TrialResult {
    stopped_at: Option<usize>,
    success: bool,
    effect_at_stop: Option<f64>,
    final_effect: f64,
    futility_info: Option<FutilityInfo>,
}

struct MethodResults {
    method: Method,
    type1_error: f64,
    success_count: usize,
    no_stop_count: usize,
    avg_stop_n: f64,
    avg_effect_stop: f64,
    avg_effect_final: f64,
    type_m_error: f64,
    futility_stats: Option<(usize, f64, f64, f64, f64, f64, usize)>,
    trajectories: Vec<Vec<f64>>,
    stop_times: Vec<f64>,
    required_effects: Vec<f64>,
}

// === LinearERT PROCESS ===

struct LinearERTProcess {
    wealth: f64,
    burn_in: usize,
    ramp: usize,
    min_val: f64,
    max_val: f64,
    sum_trt: f64,
    n_trt: f64,
    sum_ctrl: f64,
    n_ctrl: f64,
}

impl LinearERTProcess {
    fn new(burn_in: usize, ramp: usize, min_val: f64, max_val: f64) -> Self {
        LinearERTProcess {
            wealth: 1.0,
            burn_in,
            ramp,
            min_val,
            max_val,
            sum_trt: 0.0,
            n_trt: 0.0,
            sum_ctrl: 0.0,
            n_ctrl: 0.0,
        }
    }

    fn update(&mut self, i: usize, outcome: f64, is_trt: bool) {
        // Estimate delta from past data
        let mean_trt = if self.n_trt > 0.0 { self.sum_trt / self.n_trt } else { (self.min_val + self.max_val) / 2.0 };
        let mean_ctrl = if self.n_ctrl > 0.0 { self.sum_ctrl / self.n_ctrl } else { (self.min_val + self.max_val) / 2.0 };
        let delta_hat = mean_trt - mean_ctrl;

        // Update counts
        if is_trt {
            self.n_trt += 1.0;
            self.sum_trt += outcome;
        } else {
            self.n_ctrl += 1.0;
            self.sum_ctrl += outcome;
        }

        // Bet only after burn-in
        if i > self.burn_in {
            let c_i = (((i - self.burn_in) as f64) / self.ramp as f64).clamp(0.0, 1.0);

            // Normalize outcome to [0,1], then to [-1,1]
            let x = (outcome - self.min_val) / (self.max_val - self.min_val);
            let scalar = 2.0 * x - 1.0;

            // Normalize delta by range for betting magnitude
            let range = self.max_val - self.min_val;
            let delta_norm = delta_hat / range;

            let lambda = 0.5 + 0.5 * c_i * delta_norm * scalar;
            let lambda = lambda.clamp(0.001, 0.999);

            let multiplier = if is_trt { lambda / 0.5 } else { (1.0 - lambda) / 0.5 };
            self.wealth *= multiplier;
        }
    }

    fn current_effect(&self) -> f64 {
        let mean_trt = if self.n_trt > 0.0 { self.sum_trt / self.n_trt } else { 0.0 };
        let mean_ctrl = if self.n_ctrl > 0.0 { self.sum_ctrl / self.n_ctrl } else { 0.0 };
        mean_trt - mean_ctrl
    }
}

// === MAD-based PROCESS ===

struct MADProcess {
    wealth: f64,
    burn_in: usize,
    ramp: usize,
    c_max: f64,
    outcomes: Vec<f64>,
    treatments: Vec<bool>,
}

impl MADProcess {
    fn new(burn_in: usize, ramp: usize, c_max: f64) -> Self {
        MADProcess {
            wealth: 1.0,
            burn_in,
            ramp,
            c_max,
            outcomes: Vec::new(),
            treatments: Vec::new(),
        }
    }

    fn update(&mut self, i: usize, outcome: f64, is_trt: bool) {
        // Get direction from past data
        let direction = if self.outcomes.len() > 0 {
            let trt_vals: Vec<f64> = self.outcomes.iter().zip(self.treatments.iter())
                .filter(|(_, &t)| t).map(|(&o, _)| o).collect();
            let ctrl_vals: Vec<f64> = self.outcomes.iter().zip(self.treatments.iter())
                .filter(|(_, &t)| !t).map(|(&o, _)| o).collect();
            
            let mean_trt = if trt_vals.len() > 0 { trt_vals.iter().sum::<f64>() / trt_vals.len() as f64 } else { 0.0 };
            let mean_ctrl = if ctrl_vals.len() > 0 { ctrl_vals.iter().sum::<f64>() / ctrl_vals.len() as f64 } else { 0.0 };
            
            if mean_trt > mean_ctrl { 1.0 } else if mean_trt < mean_ctrl { -1.0 } else { 0.0 }
        } else {
            0.0
        };

        // Store for history
        self.outcomes.push(outcome);
        self.treatments.push(is_trt);

        // Bet only after burn-in with enough data
        if i > self.burn_in && self.outcomes.len() > 1 {
            let past_outcomes: Vec<f64> = self.outcomes[..self.outcomes.len()-1].to_vec();
            
            let med = median(&past_outcomes);
            let mad_val = mad(&past_outcomes);
            let s = if mad_val > 0.0 { mad_val } else { 1.0 };

            let r = (outcome - med) / s;
            let g = r / (1.0 + r.abs());

            let c_i = (((i - self.burn_in) as f64) / self.ramp as f64).clamp(0.0, 1.0);

            let lambda = 0.5 + c_i * self.c_max * g * direction;
            let lambda = lambda.clamp(0.001, 0.999);

            let multiplier = if is_trt { lambda / 0.5 } else { (1.0 - lambda) / 0.5 };
            self.wealth *= multiplier;
        }
    }

    fn current_effect(&self, sd: f64) -> f64 {
        let trt_vals: Vec<f64> = self.outcomes.iter().zip(self.treatments.iter())
            .filter(|(_, &t)| t).map(|(&o, _)| o).collect();
        let ctrl_vals: Vec<f64> = self.outcomes.iter().zip(self.treatments.iter())
            .filter(|(_, &t)| !t).map(|(&o, _)| o).collect();
        
        let mean_trt = if trt_vals.len() > 0 { trt_vals.iter().sum::<f64>() / trt_vals.len() as f64 } else { 0.0 };
        let mean_ctrl = if ctrl_vals.len() > 0 { ctrl_vals.iter().sum::<f64>() / ctrl_vals.len() as f64 } else { 0.0 };
        
        (mean_trt - mean_ctrl) / sd
    }
}

// === FUTILITY MONTE CARLO ===

fn required_effect_linear<R: Rng + ?Sized>(
    rng: &mut R,
    current_wealth: f64,
    n_remaining: usize,
    mu_ctrl: f64,
    sd: f64,
    min_val: f64,
    max_val: f64,
    burn_in: usize,
    ramp: usize,
    success_threshold: f64,
    mc_sims: usize,
) -> f64 {
    if n_remaining == 0 { return 1.0; }
    
    let mut low = 0.001;
    let mut high = (max_val - min_val) / 2.0;
    
    for _ in 0..6 {
        let mid = (low + high) / 2.0;
        let mu_trt = mu_ctrl + mid;
        
        let mut successes = 0;
        for _ in 0..mc_sims {
            let mut wealth = current_wealth;
            let mut sum_trt = 0.0;
            let mut n_trt = 0.0;
            let mut sum_ctrl = 0.0;
            let mut n_ctrl = 0.0;
            
            for j in 1..=n_remaining {
                let is_trt = rng.gen_bool(0.5);
                let mu = if is_trt { mu_trt } else { mu_ctrl };
                let outcome = (rng.gen::<f64>() * 2.0 - 1.0) * sd * 2.0 + mu;
                let outcome = outcome.clamp(min_val, max_val);
                
                let mean_trt = if n_trt > 0.0 { sum_trt / n_trt } else { (min_val + max_val) / 2.0 };
                let mean_ctrl = if n_ctrl > 0.0 { sum_ctrl / n_ctrl } else { (min_val + max_val) / 2.0 };
                let delta_hat = mean_trt - mean_ctrl;
                
                if is_trt { n_trt += 1.0; sum_trt += outcome; }
                else { n_ctrl += 1.0; sum_ctrl += outcome; }
                
                if j > burn_in {
                    let c_i = (((j - burn_in) as f64) / ramp as f64).clamp(0.0, 1.0);
                    let x = (outcome - min_val) / (max_val - min_val);
                    let scalar = 2.0 * x - 1.0;
                    let range = max_val - min_val;
                    let delta_norm = delta_hat / range;
                    let lambda = (0.5 + 0.5 * c_i * delta_norm * scalar).clamp(0.001, 0.999);
                    let mult = if is_trt { lambda / 0.5 } else { (1.0 - lambda) / 0.5 };
                    wealth *= mult;
                }
                
                if wealth >= success_threshold { successes += 1; break; }
            }
        }
        
        let rate = successes as f64 / mc_sims as f64;
        if rate < 0.5 { low = mid; } else { high = mid; }
    }
    
    (low + high) / 2.0
}

fn required_effect_mad<R: Rng + ?Sized>(
    rng: &mut R,
    current_wealth: f64,
    n_remaining: usize,
    mu_ctrl: f64,
    sd: f64,
    burn_in: usize,
    ramp: usize,
    c_max: f64,
    success_threshold: f64,
    mc_sims: usize,
) -> f64 {
    if n_remaining == 0 { return 1.0; }
    
    let mut low = 0.001;
    let mut high = 2.0; // Cohen's d up to 2
    
    for _ in 0..6 {
        let mid = (low + high) / 2.0;
        let mu_trt = mu_ctrl + mid * sd;
        
        let mut successes = 0;
        for _ in 0..mc_sims {
            let mut wealth = current_wealth;
            let mut outcomes: Vec<f64> = Vec::new();
            let mut treatments: Vec<bool> = Vec::new();
            
            for j in 1..=n_remaining {
                let is_trt = rng.gen_bool(0.5);
                let mu = if is_trt { mu_trt } else { mu_ctrl };
                let outcome = rng.gen::<f64>() * sd * 2.0 - sd + mu;
                
                let direction = if outcomes.len() > 0 {
                    let trt_vals: Vec<f64> = outcomes.iter().zip(treatments.iter())
                        .filter(|(_, &t)| t).map(|(&o, _)| o).collect();
                    let ctrl_vals: Vec<f64> = outcomes.iter().zip(treatments.iter())
                        .filter(|(_, &t)| !t).map(|(&o, _)| o).collect();
                    let m_t = if trt_vals.len() > 0 { trt_vals.iter().sum::<f64>() / trt_vals.len() as f64 } else { 0.0 };
                    let m_c = if ctrl_vals.len() > 0 { ctrl_vals.iter().sum::<f64>() / ctrl_vals.len() as f64 } else { 0.0 };
                    if m_t > m_c { 1.0 } else if m_t < m_c { -1.0 } else { 0.0 }
                } else { 0.0 };
                
                outcomes.push(outcome);
                treatments.push(is_trt);
                
                if j > burn_in && outcomes.len() > 1 {
                    let past: Vec<f64> = outcomes[..outcomes.len()-1].to_vec();
                    let med = median(&past);
                    let mad_val = mad(&past);
                    let s = if mad_val > 0.0 { mad_val } else { 1.0 };
                    let r = (outcome - med) / s;
                    let g = r / (1.0 + r.abs());
                    let c_i = (((j - burn_in) as f64) / ramp as f64).clamp(0.0, 1.0);
                    let lambda = (0.5 + c_i * c_max * g * direction).clamp(0.001, 0.999);
                    let mult = if is_trt { lambda / 0.5 } else { (1.0 - lambda) / 0.5 };
                    wealth *= mult;
                }
                
                if wealth >= success_threshold { successes += 1; break; }
            }
        }
        
        let rate = successes as f64 / mc_sims as f64;
        if rate < 0.5 { low = mid; } else { high = mid; }
    }
    
    (low + high) / 2.0
}

// === SIMULATION ===

fn run_simulation<R: Rng + ?Sized>(
    rng: &mut R,
    method: Method,
    n_patients: usize,
    n_sims: usize,
    mu_ctrl: f64,
    mu_trt: f64,
    sd: f64,
    min_val: f64,
    max_val: f64,
    design_effect: f64,
    success_threshold: f64,
    futility_watch: f64,
    run_futility: bool,
    burn_in: usize,
    ramp: usize,
    c_max: f64,
) -> MethodResults {
    let method_name = if method == Method::LinearERT { "LinearERT" } else { "MAD" };
    
    // Phase 1: Type I Error
    print!("  {} Type I Error... ", method_name);
    io::stdout().flush().unwrap();
    
    let mut null_rejections = 0;
    for _ in 0..n_sims {
        if method == Method::LinearERT {
            let mut proc = LinearERTProcess::new(burn_in, ramp, min_val, max_val);
            for i in 1..=n_patients {
                let is_trt = rng.gen_bool(0.5);
                let outcome = rng.gen::<f64>() * sd * 2.0 - sd + mu_ctrl;
                let outcome = if method == Method::LinearERT { outcome.clamp(min_val, max_val) } else { outcome };
                proc.update(i, outcome, is_trt);
                if proc.wealth > success_threshold { null_rejections += 1; break; }
            }
        } else {
            let mut proc = MADProcess::new(burn_in, ramp, c_max);
            for i in 1..=n_patients {
                let is_trt = rng.gen_bool(0.5);
                let outcome = rng.gen::<f64>() * sd * 2.0 - sd + mu_ctrl;
                proc.update(i, outcome, is_trt);
                if proc.wealth > success_threshold { null_rejections += 1; break; }
            }
        }
    }
    let type1_error = (null_rejections as f64 / n_sims as f64) * 100.0;
    println!("Done. {:.2}%", type1_error);

    // Phase 2: Power
    print!("  {} Power", method_name);
    if run_futility { print!(" + Futility"); }
    print!("... ");
    io::stdout().flush().unwrap();

    let mut results: Vec<TrialResult> = Vec::with_capacity(n_sims);
    let mut trajectories: Vec<Vec<f64>> = vec![vec![0.0; n_patients + 1]; n_sims];
    let pb_interval = (n_sims / 20).max(1);

    for sim in 0..n_sims {
        if sim % pb_interval == 0 { print!("."); io::stdout().flush().unwrap(); }

        let mut stopped = false;
        let mut stop_step = None;
        let mut stop_effect = None;
        let mut futility_info: Option<FutilityInfo> = None;
        let final_effect: f64;

        trajectories[sim][0] = 1.0;

        if method == Method::LinearERT {
            let mut proc = LinearERTProcess::new(burn_in, ramp, min_val, max_val);
            
            for i in 1..=n_patients {
                let is_trt = rng.gen_bool(0.5);
                let mu = if is_trt { mu_trt } else { mu_ctrl };
                let outcome = rng.gen::<f64>() * sd * 2.0 - sd + mu;
                let outcome = outcome.clamp(min_val, max_val);
                
                proc.update(i, outcome, is_trt);
                trajectories[sim][i] = proc.wealth;

                if run_futility && futility_info.is_none() && proc.wealth < futility_watch && i > burn_in {
                    let n_remaining = n_patients - i;
                    let req = required_effect_linear(
                        rng, proc.wealth, n_remaining, mu_ctrl, sd, min_val, max_val,
                        burn_in, ramp, success_threshold, 50
                    );
                    futility_info = Some(FutilityInfo {
                        patient_number: i,
                        wealth_at_trigger: proc.wealth,
                        required_effect: req,
                        ratio_to_design: req / design_effect,
                    });
                }

                if !stopped && proc.wealth > success_threshold {
                    stopped = true;
                    stop_step = Some(i);
                    stop_effect = Some(proc.current_effect());
                }
            }
            final_effect = proc.current_effect();
        } else {
            let mut proc = MADProcess::new(burn_in, ramp, c_max);
            
            for i in 1..=n_patients {
                let is_trt = rng.gen_bool(0.5);
                let mu = if is_trt { mu_trt } else { mu_ctrl };
                let outcome = rng.gen::<f64>() * sd * 2.0 - sd + mu;
                
                proc.update(i, outcome, is_trt);
                trajectories[sim][i] = proc.wealth;

                if run_futility && futility_info.is_none() && proc.wealth < futility_watch && i > burn_in {
                    let n_remaining = n_patients - i;
                    let req = required_effect_mad(
                        rng, proc.wealth, n_remaining, mu_ctrl, sd,
                        burn_in, ramp, c_max, success_threshold, 50
                    );
                    futility_info = Some(FutilityInfo {
                        patient_number: i,
                        wealth_at_trigger: proc.wealth,
                        required_effect: req,
                        ratio_to_design: req / design_effect,
                    });
                }

                if !stopped && proc.wealth > success_threshold {
                    stopped = true;
                    stop_step = Some(i);
                    stop_effect = Some(proc.current_effect(sd));
                }
            }
            final_effect = proc.current_effect(sd);
        }

        results.push(TrialResult {
            stopped_at: stop_step,
            success: stopped,
            effect_at_stop: stop_effect,
            final_effect,
            futility_info,
        });
    }
    println!(" Done.");

    // Compute statistics
    let success_count = results.iter().filter(|r| r.success).count();
    let no_stop_count = n_sims - success_count;

    let (avg_stop_n, avg_effect_stop, avg_effect_final, type_m_error) = if success_count > 0 {
        let successes: Vec<&TrialResult> = results.iter().filter(|r| r.success).collect();
        let avg_n = successes.iter().map(|r| r.stopped_at.unwrap() as f64).sum::<f64>() / success_count as f64;
        let avg_stop = successes.iter().map(|r| r.effect_at_stop.unwrap().abs()).sum::<f64>() / success_count as f64;
        let avg_final = successes.iter().map(|r| r.final_effect.abs()).sum::<f64>() / success_count as f64;
        (avg_n, avg_stop, avg_final, avg_stop / avg_final)
    } else {
        (0.0, 0.0, 0.0, 0.0)
    };

    let futility_stats = if run_futility {
        let trials_with_info: Vec<&TrialResult> = results.iter().filter(|r| r.futility_info.is_some()).collect();
        if !trials_with_info.is_empty() {
            let n_trig = trials_with_info.len();
            let mut patients: Vec<f64> = trials_with_info.iter().map(|r| r.futility_info.as_ref().unwrap().patient_number as f64).collect();
            patients.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mut effects: Vec<f64> = trials_with_info.iter().map(|r| r.futility_info.as_ref().unwrap().required_effect).collect();
            effects.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mut ratios: Vec<f64> = trials_with_info.iter().map(|r| r.futility_info.as_ref().unwrap().ratio_to_design).collect();
            ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let trig_success = trials_with_info.iter().filter(|r| r.success).count();
            Some((n_trig, patients[n_trig/2], effects[n_trig/2], ratios[n_trig/4], ratios[n_trig/2], ratios[(n_trig*3)/4], trig_success))
        } else { None }
    } else { None };

    let stop_times: Vec<f64> = results.iter().filter(|r| r.success).map(|r| r.stopped_at.unwrap() as f64).collect();
    let required_effects: Vec<f64> = if run_futility {
        results.iter().filter(|r| r.futility_info.is_some()).map(|r| r.futility_info.as_ref().unwrap().required_effect).collect()
    } else { Vec::new() };

    MethodResults {
        method,
        type1_error,
        success_count,
        no_stop_count,
        avg_stop_n,
        avg_effect_stop,
        avg_effect_final,
        type_m_error,
        futility_stats,
        trajectories,
        stop_times,
        required_effects,
    }
}

// === HTML REPORT ===

fn build_html_report(
    method_choice: usize,
    mu_ctrl: f64, mu_trt: f64, sd: f64,
    min_val: Option<f64>, max_val: Option<f64>,
    design_effect_linear: Option<f64>, design_effect_mad: Option<f64>,
    n_patients: usize, n_sims: usize,
    success_threshold: f64, futility_watch: f64,
    burn_in: usize, ramp: usize, c_max: f64,
    seed: Option<u64>, run_futility: bool,
    linear_results: Option<&MethodResults>,
    mad_results: Option<&MethodResults>,
) -> String {
    let timestamp = chrono_lite();
    let seed_str = seed.map_or("random".to_string(), |s| s.to_string());

    let method_str = match method_choice {
        1 => "LinearERT",
        2 => "MAD-based",
        _ => "LinearERT + MAD-based",
    };

    // Build method sections
    let mut method_sections = String::new();

    if let Some(res) = linear_results {
        let effect_label = "Mean Difference";
        let design_eff = design_effect_linear.unwrap_or(0.0);
        method_sections.push_str(&build_method_section(
            "LinearERT", effect_label, res, design_eff, n_patients, n_sims,
            min_val, max_val, run_futility, futility_watch, 1
        ));
    }

    if let Some(res) = mad_results {
        let effect_label = "Cohen's d";
        let design_eff = design_effect_mad.unwrap_or(0.0);
        method_sections.push_str(&build_method_section(
            "MAD-based", effect_label, res, design_eff, n_patients, n_sims,
            None, None, run_futility, futility_watch, 2
        ));
    }

    // Equations
    let linear_eq = r#"
    <h3>LinearERT Equations</h3>
    <pre>
x = (Y - min) / (max - min)           # normalize to [0,1]
scalar = 2x - 1                        # map to [-1,1]
δ̂ = (mean_trt - mean_ctrl) / range    # normalized effect estimate
λ = 0.5 + 0.5 × c × δ̂ × scalar        # betting fraction
    </pre>"#;

    let mad_eq = r#"
    <h3>MAD-based Equations</h3>
    <pre>
m = median(Y₁, ..., Y_{i-1})           # robust center
s = MAD(Y₁, ..., Y_{i-1})              # robust scale
r = (Yᵢ - m) / s                       # standardized residual
g = r / (1 + |r|)                      # squash to (-1,1)
direction = sign(mean_trt - mean_ctrl) # from past data
λ = 0.5 + c × c_max × g × direction   # betting fraction
    </pre>"#;

    let equations = match method_choice {
        1 => linear_eq.to_string(),
        2 => mad_eq.to_string(),
        _ => format!("{}{}", linear_eq, mad_eq),
    };

    let bounds_row = if let (Some(mn), Some(mx)) = (min_val, max_val) {
        format!("<tr><td>Bounds (min, max):</td><td>{}, {}</td></tr>", mn, mx)
    } else { String::new() };

    let design_effect_row = match method_choice {
        1 => format!("<tr><td>Design Effect (Mean Diff):</td><td><strong>{:.2}</strong></td></tr>", design_effect_linear.unwrap_or(0.0)),
        2 => format!("<tr><td>Design Effect (Cohen's d):</td><td><strong>{:.2}</strong></td></tr>", design_effect_mad.unwrap_or(0.0)),
        _ => format!(
            "<tr><td>Design Effect (Mean Diff):</td><td><strong>{:.2}</strong></td></tr><tr><td>Design Effect (Cohen's d):</td><td><strong>{:.2}</strong></td></tr>",
            design_effect_linear.unwrap_or(0.0), design_effect_mad.unwrap_or(0.0)
        ),
    };

    format!(r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Continuous e-RT Simulation Report</title>
    <script src="https://cdn.plot.ly/plotly-2.12.1.min.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        .container {{ background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; border-bottom: 2px solid #bdc3c7; padding-bottom: 5px; }}
        h3 {{ color: #7f8c8d; }}
        table {{ border-collapse: collapse; margin: 15px 0; }}
        td {{ padding: 8px 16px; border-bottom: 1px solid #eee; }}
        td:first-child {{ color: #7f8c8d; }}
        .highlight {{ background: #e8f4f8; font-weight: bold; }}
        .timestamp {{ color: #95a5a6; font-size: 0.9em; }}
        pre {{ background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        .method-section {{ border-left: 4px solid #3498db; padding-left: 20px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Continuous e-RT Simulation Report</h1>
        <p class="timestamp">Generated: {}</p>
        
        <h2>Parameters</h2>
        <table>
            <tr><td>Method:</td><td><strong>{}</strong></td></tr>
            <tr><td>Control Mean (μ_ctrl):</td><td>{:.2}</td></tr>
            <tr><td>Treatment Mean (μ_trt):</td><td>{:.2}</td></tr>
            <tr><td>Standard Deviation (σ):</td><td>{:.2}</td></tr>
            {}
            {}
            <tr><td>Sample Size (N):</td><td>{}</td></tr>
            <tr><td>Simulations:</td><td>{}</td></tr>
            <tr><td>Success Threshold (1/α):</td><td>{}</td></tr>
            <tr><td>Futility Watch:</td><td>{}</td></tr>
            <tr><td>Burn-in:</td><td>{}</td></tr>
            <tr><td>Ramp:</td><td>{}</td></tr>
            <tr><td>c_max (MAD):</td><td>{}</td></tr>
            <tr><td>Seed:</td><td>{}</td></tr>
        </table>

        {}

        {}
    </div>
</body>
</html>"#,
        timestamp, method_str,
        mu_ctrl, mu_trt, sd,
        bounds_row, design_effect_row,
        n_patients, n_sims, success_threshold, futility_watch,
        burn_in, ramp, c_max, seed_str,
        equations,
        method_sections
    )
}

fn build_method_section(
    name: &str,
    effect_label: &str,
    res: &MethodResults,
    design_effect: f64,
    n_patients: usize,
    n_sims: usize,
    min_val: Option<f64>,
    max_val: Option<f64>,
    run_futility: bool,
    futility_watch: f64,
    plot_id: usize,
) -> String {
    let mut html = format!(r#"
        <div class="method-section">
        <h2>{} Results</h2>
        
        <h3>Operating Characteristics</h3>
        <table>
            <tr class="highlight"><td>Type I Error:</td><td>{:.2}%</td></tr>
            <tr class="highlight"><td>Power (Success Rate):</td><td>{:.1}%</td></tr>
            <tr><td>No Stop:</td><td>{} ({:.1}%)</td></tr>
        </table>

        <h3>Success Analysis</h3>
        <table>
            <tr><td>Number of Successes:</td><td>{}</td></tr>
            <tr><td>Average Stopping Point:</td><td>{:.0} patients ({:.0}% of N)</td></tr>
            <tr><td>Effect at Stop ({}):</td><td>{:.3}</td></tr>
            <tr><td>Effect at End ({}):</td><td>{:.3}</td></tr>
            <tr><td>Type M Error (Magnification):</td><td>{:.2}x</td></tr>
        </table>
    "#,
        name,
        res.type1_error,
        (res.success_count as f64 / n_sims as f64) * 100.0,
        res.no_stop_count, (res.no_stop_count as f64 / n_sims as f64) * 100.0,
        res.success_count,
        res.avg_stop_n, (res.avg_stop_n / n_patients as f64) * 100.0,
        effect_label, res.avg_effect_stop,
        effect_label, res.avg_effect_final,
        res.type_m_error
    );

    // Futility section
    if run_futility {
        if let Some((n_trig, med_patient, med_effect, q25, q50, q75, trig_success)) = res.futility_stats {
            html.push_str(&format!(r#"
        <h3>Futility Analysis</h3>
        <table>
            <tr><td>Trials triggering (wealth &lt; {:.1}):</td><td><strong>{} ({:.1}%)</strong></td></tr>
            <tr><td>Median patient at trigger:</td><td>{:.0} ({:.0}% of N)</td></tr>
            <tr><td>Median required {} :</td><td>{:.3}</td></tr>
            <tr><td>Design {}:</td><td>{:.3}</td></tr>
            <tr><td>Ratio (Required/Design) - 25th pctl:</td><td>{:.2}x</td></tr>
            <tr><td>Ratio (Required/Design) - Median:</td><td>{:.2}x</td></tr>
            <tr><td>Ratio (Required/Design) - 75th pctl:</td><td>{:.2}x</td></tr>
            <tr><td>Triggered trials that succeeded:</td><td>{} ({:.1}%)</td></tr>
        </table>
            "#,
                futility_watch, n_trig, (n_trig as f64 / n_sims as f64) * 100.0,
                med_patient, (med_patient / n_patients as f64) * 100.0,
                effect_label, med_effect,
                effect_label, design_effect,
                q25, q50, q75,
                trig_success, (trig_success as f64 / n_trig as f64) * 100.0
            ));
        }
    }

    // Trajectory data
    let mut x_axis: Vec<usize> = Vec::new();
    let mut y_median: Vec<f64> = Vec::new();
    let mut y_lower: Vec<f64> = Vec::new();
    let mut y_upper: Vec<f64> = Vec::new();

    for i in 0..=n_patients {
        if i % 5 != 0 && i != n_patients { continue; }
        x_axis.push(i);
        let mut vals: Vec<f64> = res.trajectories.iter().map(|v| v[i]).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        y_lower.push(vals[(n_sims as f64 * 0.025) as usize]);
        y_median.push(vals[(n_sims as f64 * 0.50) as usize]);
        y_upper.push(vals[(n_sims as f64 * 0.975) as usize]);
    }

    let x_json = format!("{:?}", x_axis);
    let med_json = format!("{:?}", y_median);
    let low_json = format!("{:?}", y_lower);
    let up_json = format!("{:?}", y_upper);
    let stops_json = format!("{:?}", res.stop_times);

    // Sample trajectories
    let mut sample_traces = String::new();
    for idx in 0..30.min(res.trajectories.len()) {
        let ds: Vec<f64> = res.trajectories[idx].iter().enumerate()
            .filter(|(i, _)| i % 5 == 0 || *i == n_patients)
            .map(|(_, v)| *v).collect();
        sample_traces.push_str(&format!(
            "{{type:'scatter',mode:'lines',x:{},y:{:?},line:{{color:'rgba(100,100,100,0.3)',width:1}},showlegend:false}},",
            x_json, ds
        ));
    }

    let success_thresh = 20.0; // Will use actual value

    html.push_str(&format!(r#"
        <h3>Visualizations</h3>
        
        <h4>e-Value Trajectories (Median with 95% CI)</h4>
        <div id="plot{}_1" style="width:100%;height:450px;"></div>
        
        <h4>Sample Trajectories (30 runs)</h4>
        <div id="plot{}_2" style="width:100%;height:450px;"></div>
        
        <h4>Stopping Times Distribution</h4>
        <div id="plot{}_3" style="width:100%;height:350px;"></div>
        "#, plot_id, plot_id, plot_id));

    if run_futility && !res.required_effects.is_empty() {
        let req_json = format!("{:?}", res.required_effects);
        html.push_str(&format!(r#"
        <h4>Required {} Distribution (at Futility Trigger)</h4>
        <div id="plot{}_4" style="width:100%;height:350px;"></div>
        "#, effect_label, plot_id));

        html.push_str(&format!(r#"
        <script>
            Plotly.newPlot('plot{}_4', [{{
                type: 'histogram',
                x: {},
                marker: {{color: 'steelblue'}}
            }}], {{
                shapes: [{{type:'line',x0:{:.3},x1:{:.3},y0:0,y1:1,yref:'paper',line:{{color:'red',width:2,dash:'dash'}}}}],
                xaxis: {{title: 'Required {}'}},
                yaxis: {{title: 'Count'}},
                annotations: [{{x:{:.3},y:1,yref:'paper',text:'Design',showarrow:false,font:{{color:'red'}}}}]
            }});
        </script>
        "#, plot_id, req_json, design_effect, design_effect, effect_label, design_effect));
    }

    html.push_str(&format!(r#"
        <script>
            // Plot 1: Median + 95% CI
            Plotly.newPlot('plot{}_1', [
                {{type:'scatter',mode:'lines',x:{},y:{},line:{{width:0}},showlegend:false}},
                {{type:'scatter',mode:'lines',x:{},y:{},fill:'tonexty',fillcolor:'rgba(31,119,180,0.3)',line:{{width:0}},showlegend:false}},
                {{type:'scatter',mode:'lines',x:{},y:{},line:{{color:'blue',width:2.5}},name:'Median'}}
            ], {{
                yaxis: {{type:'log',title:'e-value'}},
                xaxis: {{title:'Patients Enrolled'}},
                shapes: [
                    {{type:'line',x0:0,x1:1,xref:'paper',y0:20,y1:20,line:{{color:'green',width:2,dash:'dash'}}}},
                    {{type:'line',x0:0,x1:1,xref:'paper',y0:{},y1:{},line:{{color:'orange',width:1.5,dash:'dot'}}}}
                ]
            }});

            // Plot 2: Sample trajectories
            Plotly.newPlot('plot{}_2', [
                {}
                {{type:'scatter',mode:'lines',x:[0,{}],y:[20,20],line:{{color:'green',width:2,dash:'dash'}},name:'Success'}},
                {{type:'scatter',mode:'lines',x:[0,{}],y:[{},{}],line:{{color:'orange',width:1.5,dash:'dot'}},name:'Futility Watch'}}
            ], {{
                yaxis: {{type:'log',title:'e-value'}},
                xaxis: {{title:'Patients Enrolled'}}
            }});

            // Plot 3: Stopping times
            Plotly.newPlot('plot{}_3', [{{
                type: 'histogram',
                x: {},
                marker: {{color: 'green'}}
            }}], {{
                xaxis: {{title: 'Patient Number at Stop'}},
                yaxis: {{title: 'Count'}}
            }});
        </script>
        </div>
    "#,
        plot_id, x_json, low_json, x_json, up_json, x_json, med_json,
        futility_watch, futility_watch,
        plot_id, sample_traces, n_patients, n_patients, futility_watch, futility_watch,
        plot_id, stops_json
    ));

    html
}

// === MAIN ===

pub fn run() {
    println!("\n==========================================");
    println!("   CONTINUOUS e-RT SIMULATION");
    println!("==========================================\n");

    // Method selection
    let method_choice = get_choice("Select method:", &[
        "LinearERT (bounded outcomes, e.g., VFD 0-28)",
        "MAD-based (unbounded, matches paper)",
        "Both",
    ]);

    // Common inputs
    let mu_ctrl = get_input("Control Mean (μ_ctrl, e.g., 14): ");
    let mu_trt = get_input("Treatment Mean (μ_trt, e.g., 18): ");

    // Method-specific inputs
    let (min_val, max_val): (Option<f64>, Option<f64>) = if method_choice == 1 || method_choice == 3 {
        let mn = get_input("Min bound (e.g., 0): ");
        let mx = get_input("Max bound (e.g., 28): ");
        (Some(mn), Some(mx))
    } else {
        (None, None)
    };

    let sd = if method_choice == 2 || method_choice == 3 {
        get_input("Standard Deviation (σ, e.g., 10): ")
    } else if get_bool("Enter SD for sample size calculation?") {
        get_input("Standard Deviation (σ): ")
    } else {
        1.0 // placeholder, won't be used for betting
    };

    // Design effects
    let design_effect_linear = if method_choice == 1 || method_choice == 3 {
        Some((mu_trt - mu_ctrl).abs())
    } else { None };

    let design_effect_mad = if method_choice == 2 || method_choice == 3 {
        Some(((mu_trt - mu_ctrl) / sd).abs())
    } else { None };

    // Sample size
    let n_patients = if get_bool("Calculate Sample Size automatically?") {
        let power = get_input("Target Power (e.g., 0.80): ");
        let cohen_d = ((mu_trt - mu_ctrl) / sd).abs();
        let freq_n = calculate_n_continuous(cohen_d, power);
        println!("\nFrequentist N (Power {:.0}%, d={:.2}): {}", power * 100.0, cohen_d, freq_n);
        
        if get_bool("Add buffer?") {
            let buffer_pct = get_input("Buffer percentage (e.g., 15): ");
            let buffered = (freq_n as f64 * (1.0 + buffer_pct / 100.0)).ceil() as usize;
            println!("Buffered N: {}", buffered);
            buffered
        } else { freq_n }
    } else {
        get_input_usize("Enter Number of Patients: ")
    };

    let n_sims = get_input_usize("Number of simulations (e.g., 2000): ");

    println!("\nSuccess threshold (1/alpha). Default = 20");
    let success_threshold = get_input("Success threshold: ");

    println!("\nFutility watch threshold. Default = 0.5");
    let futility_watch = get_input("Futility watch: ");

    let run_futility = get_bool("Run futility analysis?");
    let seed = get_optional_input("Seed (press Enter for random): ");

    // Fixed parameters
    let burn_in: usize = 20;
    let ramp: usize = 50;
    let c_max: f64 = 0.6;

    println!("\n--- Configuration ---");
    println!("Control Mean:    {:.2}", mu_ctrl);
    println!("Treatment Mean:  {:.2}", mu_trt);
    if let Some(d) = design_effect_linear { println!("Design Mean Diff: {:.2}", d); }
    if let Some(d) = design_effect_mad { println!("Design Cohen's d: {:.2}", d); }
    if let (Some(mn), Some(mx)) = (min_val, max_val) { println!("Bounds:          [{}, {}]", mn, mx); }
    println!("SD:              {:.2}", sd);
    println!("N:               {}", n_patients);
    println!("Simulations:     {}", n_sims);
    println!("Burn-in:         {} (default)", burn_in);
    println!("Ramp:            {} (default)", ramp);
    println!("c_max:           {} (default)", c_max);

    // Initialize RNG
    let mut rng: Box<dyn RngCore> = match seed {
        Some(s) => Box::new(StdRng::seed_from_u64(s)),
        None => Box::new(rand::thread_rng()),
    };

    println!("\n==========================================");
    println!("   RUNNING SIMULATIONS");
    println!("==========================================\n");

    // Run simulations
    let linear_results = if method_choice == 1 || method_choice == 3 {
        Some(run_simulation(
            &mut *rng, Method::LinearERT, n_patients, n_sims,
            mu_ctrl, mu_trt, sd, min_val.unwrap_or(0.0), max_val.unwrap_or(1.0),
            design_effect_linear.unwrap(), success_threshold, futility_watch, run_futility,
            burn_in, ramp, c_max
        ))
    } else { None };

    let mad_results = if method_choice == 2 || method_choice == 3 {
        Some(run_simulation(
            &mut *rng, Method::MAD, n_patients, n_sims,
            mu_ctrl, mu_trt, sd, min_val.unwrap_or(0.0), max_val.unwrap_or(1.0),
            design_effect_mad.unwrap(), success_threshold, futility_watch, run_futility,
            burn_in, ramp, c_max
        ))
    } else { None };

    // Generate report
    println!("\nGenerating report...");

    let html = build_html_report(
        method_choice,
        mu_ctrl, mu_trt, sd,
        min_val, max_val,
        design_effect_linear, design_effect_mad,
        n_patients, n_sims,
        success_threshold, futility_watch,
        burn_in, ramp, c_max,
        seed, run_futility,
        linear_results.as_ref(), mad_results.as_ref(),
    );

    let mut file = File::create("continuous_report.html").unwrap();
    file.write_all(html.as_bytes()).unwrap();

    println!("\n>> Report saved: continuous_report.html");
    println!("==========================================");
}