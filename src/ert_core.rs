//! Core e-RT (e-Randomization Test) functionality
//! Shared structs, helpers, and statistical functions

use std::io::{self, Write};

// ============================================================================
// INPUT HELPERS
// ============================================================================

/// Get f64 input from user with prompt
pub fn get_input(prompt: &str) -> f64 {
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

/// Get usize input from user with prompt
pub fn get_input_usize(prompt: &str) -> usize {
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

/// Get yes/no input from user
pub fn get_bool(prompt: &str) -> bool {
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

/// Get optional u64 input (empty = None)
pub fn get_optional_input(prompt: &str) -> Option<u64> {
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

/// Get string input from user
pub fn get_string(prompt: &str) -> String {
    print!("{}", prompt);
    io::stdout().flush().unwrap();
    let mut buffer = String::new();
    io::stdin().read_line(&mut buffer).unwrap();
    buffer.trim().to_string()
}

/// Get choice from numbered options
pub fn get_choice(prompt: &str, options: &[&str]) -> usize {
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

// ============================================================================
// STATISTICAL HELPERS
// ============================================================================

/// Calculate median of a slice
pub fn median(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

/// Calculate median absolute deviation
pub fn mad(data: &[f64]) -> f64 {
    let med = median(data);
    let deviations: Vec<f64> = data.iter().map(|x| (x - med).abs()).collect();
    median(&deviations)
}

/// Simple timestamp without external crate (proper leap year handling)
pub fn chrono_lite() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let secs = duration.as_secs();
    let mut days = (secs / 86400) as i64;

    // Find year with leap year handling
    let mut year = 1970i64;
    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        year += 1;
    }

    // Find month
    let days_in_months: [i64; 12] = if is_leap_year(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut month = 1;
    for &dm in &days_in_months {
        if days < dm {
            break;
        }
        days -= dm;
        month += 1;
    }

    let day = days + 1;  // 1-indexed
    let hours = (secs % 86400) / 3600;
    let mins = (secs % 3600) / 60;
    format!("{}-{:02}-{:02} {:02}:{:02} UTC", year, month, day, hours, mins)
}

fn is_leap_year(year: i64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

// ============================================================================
// BINARY e-RT CORE
// ============================================================================

/// Binary e-RT process for sequential analysis
pub struct BinaryERTProcess {
    pub wealth: f64,
    pub burn_in: usize,
    pub ramp: usize,
    pub n_trt: f64,
    pub events_trt: f64,
    pub n_ctrl: f64,
    pub events_ctrl: f64,
}

impl BinaryERTProcess {
    pub fn new(burn_in: usize, ramp: usize) -> Self {
        BinaryERTProcess {
            wealth: 1.0,
            burn_in,
            ramp,
            n_trt: 0.0,
            events_trt: 0.0,
            n_ctrl: 0.0,
            events_ctrl: 0.0,
        }
    }

    /// Update the e-process with a new observation
    /// i: patient number (1-indexed)
    /// outcome: 1.0 for event, 0.0 for no event
    /// is_trt: true if treatment arm
    pub fn update(&mut self, i: usize, outcome: f64, is_trt: bool) {
        // Estimate delta from past data (before updating counts)
        let rate_trt = if self.n_trt > 0.0 {
            self.events_trt / self.n_trt
        } else {
            0.5
        };
        let rate_ctrl = if self.n_ctrl > 0.0 {
            self.events_ctrl / self.n_ctrl
        } else {
            0.5
        };
        let delta_hat = rate_trt - rate_ctrl;

        // Update counts
        if is_trt {
            self.n_trt += 1.0;
            if outcome == 1.0 {
                self.events_trt += 1.0;
            }
        } else {
            self.n_ctrl += 1.0;
            if outcome == 1.0 {
                self.events_ctrl += 1.0;
            }
        }

        // Bet only after burn-in
        if i > self.burn_in {
            let num = ((i - self.burn_in) as f64).max(0.0);
            let c_i = (num / self.ramp as f64).clamp(0.0, 1.0);

            let lambda = if outcome == 1.0 {
                0.5 + 0.5 * c_i * delta_hat
            } else {
                0.5 - 0.5 * c_i * delta_hat
            };

            let lambda = lambda.clamp(0.001, 0.999);
            let multiplier = if is_trt {
                lambda / 0.5
            } else {
                (1.0 - lambda) / 0.5
            };
            self.wealth *= multiplier;
        }
    }

    /// Current risk difference (treatment - control)
    pub fn current_risk_diff(&self) -> f64 {
        let r_t = if self.n_trt > 0.0 {
            self.events_trt / self.n_trt
        } else {
            0.0
        };
        let r_c = if self.n_ctrl > 0.0 {
            self.events_ctrl / self.n_ctrl
        } else {
            0.0
        };
        r_t - r_c
    }

    /// Current odds ratio with 95% CI
    /// Returns (OR, lower_95, upper_95)
    pub fn current_odds_ratio(&self) -> (f64, f64, f64) {
        // Add 0.5 continuity correction to avoid division by zero
        let a = self.events_trt + 0.5; // treatment events
        let b = self.n_trt - self.events_trt + 0.5; // treatment non-events
        let c = self.events_ctrl + 0.5; // control events
        let d = self.n_ctrl - self.events_ctrl + 0.5; // control non-events

        let or = (a * d) / (b * c);

        // Log OR standard error (Woolf's method with continuity correction)
        let se_log_or = (1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d).sqrt();

        let log_or = or.ln();
        let lower = (log_or - 1.96 * se_log_or).exp();
        let upper = (log_or + 1.96 * se_log_or).exp();

        (or, lower, upper)
    }

    /// Get event rates for both arms
    pub fn get_rates(&self) -> (f64, f64) {
        let r_t = if self.n_trt > 0.0 {
            self.events_trt / self.n_trt
        } else {
            0.0
        };
        let r_c = if self.n_ctrl > 0.0 {
            self.events_ctrl / self.n_ctrl
        } else {
            0.0
        };
        (r_t, r_c)
    }

    /// Get sample sizes for both arms
    pub fn get_ns(&self) -> (usize, usize) {
        (self.n_trt as usize, self.n_ctrl as usize)
    }
}

// ============================================================================
// CONTINUOUS e-RTo PROCESS (bounded/ordinal)
// ============================================================================

/// e-RTo process for bounded continuous outcomes (e.g., VFD 0-28)
pub struct LinearERTProcess {
    pub wealth: f64,
    pub burn_in: usize,
    pub ramp: usize,
    pub min_val: f64,
    pub max_val: f64,
    sum_trt: f64,
    n_trt: f64,
    sum_ctrl: f64,
    n_ctrl: f64,
}

impl LinearERTProcess {
    pub fn new(burn_in: usize, ramp: usize, min_val: f64, max_val: f64) -> Self {
        Self {
            wealth: 1.0, burn_in, ramp, min_val, max_val,
            sum_trt: 0.0, n_trt: 0.0, sum_ctrl: 0.0, n_ctrl: 0.0,
        }
    }

    pub fn update(&mut self, i: usize, outcome: f64, is_trt: bool) {
        let mean_trt = if self.n_trt > 0.0 { self.sum_trt / self.n_trt } else { (self.min_val + self.max_val) / 2.0 };
        let mean_ctrl = if self.n_ctrl > 0.0 { self.sum_ctrl / self.n_ctrl } else { (self.min_val + self.max_val) / 2.0 };
        let delta_hat = mean_trt - mean_ctrl;

        if is_trt { self.n_trt += 1.0; self.sum_trt += outcome; }
        else { self.n_ctrl += 1.0; self.sum_ctrl += outcome; }

        if i > self.burn_in {
            let c_i = (((i - self.burn_in) as f64) / self.ramp as f64).clamp(0.0, 1.0);
            let x = (outcome - self.min_val) / (self.max_val - self.min_val);
            let scalar = 2.0 * x - 1.0;
            let range = self.max_val - self.min_val;
            let delta_norm = delta_hat / range;
            let lambda = (0.5 + 0.5 * c_i * delta_norm * scalar).clamp(0.001, 0.999);
            let mult = if is_trt { lambda / 0.5 } else { (1.0 - lambda) / 0.5 };
            self.wealth *= mult;
        }
    }

    pub fn current_effect(&self) -> f64 {
        let mt = if self.n_trt > 0.0 { self.sum_trt / self.n_trt } else { 0.0 };
        let mc = if self.n_ctrl > 0.0 { self.sum_ctrl / self.n_ctrl } else { 0.0 };
        mt - mc
    }

    pub fn get_means(&self) -> (f64, f64) {
        (if self.n_trt > 0.0 { self.sum_trt / self.n_trt } else { 0.0 },
         if self.n_ctrl > 0.0 { self.sum_ctrl / self.n_ctrl } else { 0.0 })
    }

    pub fn get_ns(&self) -> (usize, usize) {
        (self.n_trt as usize, self.n_ctrl as usize)
    }
}

// ============================================================================
// CONTINUOUS e-RTc PROCESS (unbounded/MAD-based)
// ============================================================================

/// e-RTc process for unbounded continuous outcomes (biomarkers, lab values)
pub struct MADProcess {
    pub wealth: f64,
    pub burn_in: usize,
    pub ramp: usize,
    pub c_max: f64,
    outcomes: Vec<f64>,
    treatments: Vec<bool>,
}

impl MADProcess {
    pub fn new(burn_in: usize, ramp: usize, c_max: f64) -> Self {
        Self { wealth: 1.0, burn_in, ramp, c_max, outcomes: Vec::new(), treatments: Vec::new() }
    }

    pub fn update(&mut self, i: usize, outcome: f64, is_trt: bool) {
        // Continuous direction: standardized effect estimate
        let direction = if !self.outcomes.is_empty() {
            let (mut sum_t, mut ss_t, mut n_t) = (0.0, 0.0, 0.0);
            let (mut sum_c, mut ss_c, mut n_c) = (0.0, 0.0, 0.0);
            for (&o, &t) in self.outcomes.iter().zip(self.treatments.iter()) {
                if t { sum_t += o; ss_t += o * o; n_t += 1.0; }
                else { sum_c += o; ss_c += o * o; n_c += 1.0; }
            }
            if n_t > 1.0 && n_c > 1.0 {
                let m_t = sum_t / n_t;
                let m_c = sum_c / n_c;
                let var_t = (ss_t - sum_t * sum_t / n_t) / (n_t - 1.0);
                let var_c = (ss_c - sum_c * sum_c / n_c) / (n_c - 1.0);
                let pooled_sd = ((var_t + var_c) / 2.0).sqrt().max(0.001);
                let delta = (m_t - m_c) / pooled_sd;
                delta.clamp(-1.0, 1.0)
            } else { 0.0 }
        } else { 0.0 };

        self.outcomes.push(outcome);
        self.treatments.push(is_trt);

        if i > self.burn_in && self.outcomes.len() > 1 {
            let past: Vec<f64> = self.outcomes[..self.outcomes.len()-1].to_vec();
            let med = median(&past);
            let mad_val = mad(&past);
            let s = if mad_val > 0.0 { mad_val } else { 1.0 };
            let r = (outcome - med) / s;
            let g = r / (1.0 + r.abs());
            let c_i = (((i - self.burn_in) as f64) / self.ramp as f64).clamp(0.0, 1.0);
            let lambda = (0.5 + c_i * self.c_max * g * direction).clamp(0.001, 0.999);
            let mult = if is_trt { lambda / 0.5 } else { (1.0 - lambda) / 0.5 };
            self.wealth *= mult;
        }
    }

    pub fn current_effect(&self, sd: f64) -> f64 {
        let (mut sum_t, mut n_t, mut sum_c, mut n_c) = (0.0, 0.0, 0.0, 0.0);
        for (&o, &t) in self.outcomes.iter().zip(self.treatments.iter()) {
            if t { sum_t += o; n_t += 1.0; } else { sum_c += o; n_c += 1.0; }
        }
        let m_t = if n_t > 0.0 { sum_t / n_t } else { 0.0 };
        let m_c = if n_c > 0.0 { sum_c / n_c } else { 0.0 };
        (m_t - m_c) / sd
    }

    pub fn get_means(&self) -> (f64, f64) {
        let (mut sum_t, mut n_t, mut sum_c, mut n_c) = (0.0, 0.0, 0.0, 0.0);
        for (&o, &t) in self.outcomes.iter().zip(self.treatments.iter()) {
            if t { sum_t += o; n_t += 1.0; } else { sum_c += o; n_c += 1.0; }
        }
        (if n_t > 0.0 { sum_t / n_t } else { 0.0 },
         if n_c > 0.0 { sum_c / n_c } else { 0.0 })
    }

    pub fn get_ns(&self) -> (usize, usize) {
        let n_trt = self.treatments.iter().filter(|&&t| t).count();
        (n_trt, self.treatments.len() - n_trt)
    }

    pub fn get_pooled_sd(&self) -> f64 {
        let (mut sum_t, mut ss_t, mut n_t) = (0.0, 0.0, 0.0);
        let (mut sum_c, mut ss_c, mut n_c) = (0.0, 0.0, 0.0);
        for (&o, &t) in self.outcomes.iter().zip(self.treatments.iter()) {
            if t { sum_t += o; ss_t += o * o; n_t += 1.0; }
            else { sum_c += o; ss_c += o * o; n_c += 1.0; }
        }
        if n_t < 2.0 || n_c < 2.0 { return 1.0; }
        let var_t = (ss_t - sum_t * sum_t / n_t) / (n_t - 1.0);
        let var_c = (ss_c - sum_c * sum_c / n_c) / (n_c - 1.0);
        ((var_t * (n_t - 1.0) + var_c * (n_c - 1.0)) / (n_t + n_c - 2.0)).sqrt()
    }
}

// ============================================================================
// SAMPLE SIZE CALCULATIONS
// ============================================================================

/// Calculate required N for binary endpoint (two-sided, equal allocation)
pub fn calculate_n_binary(p_ctrl: f64, p_trt: f64, power: f64) -> usize {
    let z_alpha: f64 = 1.96;
    let z_beta: f64 = if power > 0.85 { 1.28 } else { 0.84 };

    let p_bar = (p_ctrl + p_trt) / 2.0;
    let delta = (p_ctrl - p_trt).abs();

    let term1 = 4.0 * (z_alpha + z_beta).powi(2);
    let term2 = p_bar * (1.0 - p_bar);
    let term3 = delta.powi(2);

    ((term1 * term2) / term3).ceil() as usize
}

/// Calculate required N for continuous endpoint
pub fn calculate_n_continuous(cohen_d: f64, power: f64) -> usize {
    let z_alpha: f64 = 1.96;
    let z_beta: f64 = if power > 0.85 { 1.28 } else { 0.84 };
    let n_per_arm = (2.0 * ((z_alpha + z_beta) / cohen_d).powi(2)).ceil() as usize;
    2 * n_per_arm
}

/// Standard normal CDF approximation
pub fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Standard normal quantile (inverse CDF)
pub fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 { return f64::NEG_INFINITY; }
    if p >= 1.0 { return f64::INFINITY; }
    if (p - 0.5).abs() < 1e-10 { return 0.0; }

    let p_low = if p < 0.5 { p } else { 1.0 - p };
    let t = (-2.0 * p_low.ln()).sqrt();

    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);
    if p < 0.5 { -z } else { z }
}

/// Error function approximation (Abramowitz and Stegun)
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

/// Calculate z-test power for binary endpoint given sample size
/// p_ctrl, p_trt: event rates
/// n_total: total sample size (assumes 1:1 allocation)
/// alpha: significance level (two-sided)
pub fn z_test_power_binary(p_ctrl: f64, p_trt: f64, n_total: usize, alpha: f64) -> f64 {
    let n_per_arm = n_total as f64 / 2.0;
    let delta = (p_ctrl - p_trt).abs();

    // Pooled standard error under H1
    let var_trt = p_trt * (1.0 - p_trt);
    let var_ctrl = p_ctrl * (1.0 - p_ctrl);
    let se = ((var_trt + var_ctrl) / n_per_arm).sqrt();

    if se < 1e-10 {
        return 1.0;
    }

    // z_alpha for two-sided test
    let z_alpha = z_from_alpha(alpha);

    // Effect size in z-scale
    let z_effect = delta / se;

    // Power = P(|Z| > z_alpha | H1)
    normal_cdf(z_effect - z_alpha)
}

/// Get z-score from alpha (two-sided)
fn z_from_alpha(alpha: f64) -> f64 {
    // Common values
    if (alpha - 0.05).abs() < 0.001 { return 1.96; }
    if (alpha - 0.01).abs() < 0.001 { return 2.576; }
    if (alpha - 0.10).abs() < 0.001 { return 1.645; }

    // Binary search for general case
    let target = 1.0 - alpha / 2.0;
    let mut low = 0.0;
    let mut high = 5.0;
    for _ in 0..50 {
        let mid = (low + high) / 2.0;
        if normal_cdf(mid) < target {
            low = mid;
        } else {
            high = mid;
        }
    }
    (low + high) / 2.0
}

/// Calculate t-test power for continuous endpoint given sample size
/// effect: mean difference (mu_trt - mu_ctrl)
/// sd: common standard deviation
/// n_total: total sample size (assumes 1:1 allocation)
/// alpha: significance level (two-sided)
pub fn t_test_power_continuous(effect: f64, sd: f64, n_total: usize, alpha: f64) -> f64 {
    let n_per_arm = n_total as f64 / 2.0;

    // Standard error of the difference
    let se = sd * (2.0 / n_per_arm).sqrt();

    if se < 1e-10 {
        return 1.0;
    }

    // z_alpha for two-sided test
    let z_alpha = z_from_alpha(alpha);

    // Effect size in z-scale (non-centrality parameter)
    let z_effect = effect.abs() / se;

    // Power = P(|Z| > z_alpha | H1)
    normal_cdf(z_effect - z_alpha)
}

// ============================================================================
// FUTILITY MONITORING
// ============================================================================
//
// IMPORTANT: This is NOT a martingale or e-process. It is a simulation-based
// decision support tool that runs alongside the e-RT process.
//
// The FutilityMonitor uses Monte Carlo simulation to answer:
// "What treatment effect (ARR) would we need for X% probability of recovery?"
//
// If the required ARR is much larger than the design ARR (ratio > stop_ratio),
// the monitor recommends considering stopping for futility.
//
// This provides actionable information but does NOT have the anytime-valid
// statistical guarantees of the e-process itself.
//

/// Checkpoint result from futility monitoring
#[derive(Clone, Debug)]
pub struct FutilityCheckpoint {
    pub patient: usize,
    pub wealth: f64,
    pub mode: FutilityMode,
    pub ratio: f64,           // required_arr / design_arr
    pub required_arr: f64,    // ARR needed for recovery_target probability
    pub recommend_stop: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum FutilityMode {
    Normal,  // Checking every normal_interval
    Alert,   // Checking every alert_interval (wealth below danger threshold)
}

/// Configuration for futility monitoring (simulation-based, NOT a martingale)
#[derive(Clone, Debug)]
pub struct FutilityConfig {
    pub recovery_target: f64,      // Default 0.10 (10% - "less than 10% chance of recovery")
    pub normal_interval_pct: f64,  // Default 0.05 (check every 5% of N)
    pub alert_interval_pct: f64,   // Default 0.01 (check every 1% when in danger)
    pub danger_threshold: f64,     // Default 0.5 (wealth below this triggers alert mode)
    pub stop_ratio: f64,           // Default 1.75 (recommend stop if ratio exceeds this)
    pub mc_samples: usize,         // Default 200
    pub binary_iterations: usize,  // Default 10
    pub hysteresis_count: usize,   // Default 3 (consecutive checks above threshold to exit alert)
}

impl Default for FutilityConfig {
    fn default() -> Self {
        FutilityConfig {
            recovery_target: 0.10,
            normal_interval_pct: 0.05,
            alert_interval_pct: 0.01,
            danger_threshold: 0.5,
            stop_ratio: 1.75,  // User can override; 1.75 = need 75% more than design ARR
            mc_samples: 200,
            binary_iterations: 10,
            hysteresis_count: 3,
        }
    }
}

/// Futility monitor for binary e-RT trials
pub struct FutilityMonitor {
    pub config: FutilityConfig,
    pub design_arr: f64,           // Design effect (p_ctrl - p_trt)
    pub p_ctrl: f64,               // Control event rate
    pub n_total: usize,            // Total planned sample size
    pub threshold: f64,            // e-value threshold for success (e.g., 20)
    pub burn_in: usize,
    pub ramp: usize,

    // State
    mode: FutilityMode,
    next_checkpoint: usize,
    consecutive_above: usize,      // Count for hysteresis
    pub checkpoints: Vec<FutilityCheckpoint>,
}

impl FutilityMonitor {
    pub fn new(
        config: FutilityConfig,
        p_ctrl: f64,
        p_trt: f64,
        n_total: usize,
        threshold: f64,
        burn_in: usize,
        ramp: usize,
    ) -> Self {
        let design_arr = (p_ctrl - p_trt).abs();
        let first_checkpoint = ((n_total as f64 * config.normal_interval_pct).ceil() as usize)
            .max(burn_in + 1);

        FutilityMonitor {
            config,
            design_arr,
            p_ctrl,
            n_total,
            threshold,
            burn_in,
            ramp,
            mode: FutilityMode::Normal,
            next_checkpoint: first_checkpoint,
            consecutive_above: 0,
            checkpoints: Vec::new(),
        }
    }

    /// Check if we should evaluate futility at this patient number
    pub fn should_check(&self, patient: usize) -> bool {
        patient >= self.next_checkpoint && patient > self.burn_in
    }

    /// Perform futility check and return checkpoint result
    /// Returns None if not at a checkpoint
    pub fn check(&mut self, patient: usize, wealth: f64) -> Option<FutilityCheckpoint> {
        if !self.should_check(patient) {
            return None;
        }

        let n_remaining = self.n_total.saturating_sub(patient);

        // Calculate required ARR for recovery_target probability
        let required_arr = self.required_arr_for_recovery(wealth, n_remaining);
        let ratio = if self.design_arr > 0.0 {
            required_arr / self.design_arr
        } else {
            f64::INFINITY
        };

        // Determine mode transition
        if wealth < self.config.danger_threshold {
            self.mode = FutilityMode::Alert;
            self.consecutive_above = 0;
        } else {
            self.consecutive_above += 1;
            if self.consecutive_above >= self.config.hysteresis_count
                && self.mode == FutilityMode::Alert {
                self.mode = FutilityMode::Normal;
            }
        }

        // Calculate next checkpoint
        let interval = match self.mode {
            FutilityMode::Normal => self.config.normal_interval_pct,
            FutilityMode::Alert => self.config.alert_interval_pct,
        };
        let step = ((self.n_total as f64 * interval).ceil() as usize).max(1);
        self.next_checkpoint = patient + step;

        // Build checkpoint
        let checkpoint = FutilityCheckpoint {
            patient,
            wealth,
            mode: self.mode.clone(),
            ratio,
            required_arr,
            recommend_stop: ratio > self.config.stop_ratio,
        };

        self.checkpoints.push(checkpoint.clone());
        Some(checkpoint)
    }

    /// Monte Carlo binary search to find ARR needed for recovery_target probability
    fn required_arr_for_recovery(&self, current_wealth: f64, n_remaining: usize) -> f64 {
        use rand::Rng;
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        if n_remaining == 0 {
            return f64::INFINITY;
        }

        // Already above threshold - no additional effect needed
        if current_wealth >= self.threshold {
            return 0.0;
        }

        let mut rng = StdRng::seed_from_u64(42); // Deterministic for reproducibility
        let (mut lo, mut hi) = (0.0001, 0.50);

        for _ in 0..self.config.binary_iterations {
            let mid = (lo + hi) / 2.0;
            let p_trt = (self.p_ctrl - mid).max(0.001);

            let mut successes = 0usize;

            for _ in 0..self.config.mc_samples {
                let mut wealth = current_wealth;
                let (mut n_t, mut e_t, mut n_c, mut e_c) = (0.0, 0.0, 0.0, 0.0);

                for j in 1..=n_remaining {
                    let is_trt = rng.gen_bool(0.5);
                    let event = rng.gen_bool(if is_trt { p_trt } else { self.p_ctrl });

                    // Estimate delta from accumulated data in this simulation
                    let delta = if n_t > 0.0 && n_c > 0.0 {
                        e_t / n_t - e_c / n_c
                    } else {
                        0.0
                    };

                    // Update counts
                    if is_trt {
                        n_t += 1.0;
                        if event { e_t += 1.0; }
                    } else {
                        n_c += 1.0;
                        if event { e_c += 1.0; }
                    }

                    // Apply e-process update (simplified - no burn-in in forward sim)
                    let c = (j as f64 / self.ramp as f64).min(1.0);
                    let outcome_sign = if event { 1.0 } else { -1.0 };
                    let lambda = (0.5 + 0.5 * c * delta * outcome_sign).clamp(0.001, 0.999);
                    let mult = if is_trt { lambda / 0.5 } else { (1.0 - lambda) / 0.5 };
                    wealth *= mult;

                    if wealth >= self.threshold {
                        successes += 1;
                        break;
                    }
                }
            }

            let success_rate = successes as f64 / self.config.mc_samples as f64;

            // Binary search: find ARR where success_rate = recovery_target
            if success_rate < self.config.recovery_target {
                lo = mid; // Need larger effect
            } else {
                hi = mid; // Effect is sufficient
            }
        }

        (lo + hi) / 2.0
    }

    /// Get summary of futility monitoring
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("\n--- Futility Monitoring Summary ---\n"));
        s.push_str(&format!("Design ARR: {:.1}%\n", self.design_arr * 100.0));
        s.push_str(&format!("Recovery target: {:.0}%\n", self.config.recovery_target * 100.0));
        s.push_str(&format!("Stop ratio: {:.1}x\n", self.config.stop_ratio));
        s.push_str(&format!("Checkpoints: {}\n\n", self.checkpoints.len()));

        if self.checkpoints.is_empty() {
            s.push_str("No checkpoints recorded.\n");
            return s;
        }

        s.push_str("Patient | Wealth |  Mode  | Ratio | Req ARR | Recommend\n");
        s.push_str("--------|--------|--------|-------|---------|----------\n");

        for cp in &self.checkpoints {
            let mode_str = match cp.mode {
                FutilityMode::Normal => "NORMAL",
                FutilityMode::Alert => "ALERT ",
            };
            let recommend = if cp.recommend_stop { "STOP" } else { "-" };
            s.push_str(&format!(
                "{:>7} | {:>6.3} | {} | {:>5.2}x | {:>6.1}% | {}\n",
                cp.patient,
                cp.wealth,
                mode_str,
                cp.ratio,
                cp.required_arr * 100.0,
                recommend
            ));
        }

        // Final recommendation
        if let Some(last) = self.checkpoints.last() {
            s.push_str(&format!("\nFinal status: "));
            if last.recommend_stop {
                s.push_str(&format!(
                    "RECOMMEND STOP (ratio {:.2}x > {:.1}x threshold)\n",
                    last.ratio, self.config.stop_ratio
                ));
            } else {
                s.push_str(&format!("Continue (ratio {:.2}x)\n", last.ratio));
            }
        }

        s
    }

    /// Check if any checkpoint recommended stopping
    pub fn ever_recommended_stop(&self) -> bool {
        self.checkpoints.iter().any(|cp| cp.recommend_stop)
    }

    /// Get the worst (highest) ratio observed
    pub fn worst_ratio(&self) -> Option<f64> {
        self.checkpoints.iter().map(|cp| cp.ratio).fold(None, |acc, r| {
            match acc {
                None => Some(r),
                Some(a) if r > a => Some(r),
                Some(a) => Some(a),
            }
        })
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_futility_monitor_basic() {
        let config = FutilityConfig::default();
        let mut monitor = FutilityMonitor::new(
            config,
            0.25,  // p_ctrl
            0.20,  // p_trt (5% ARR)
            500,   // n_total
            20.0,  // threshold
            50,    // burn_in
            100,   // ramp
        );

        assert!(monitor.should_check(51));

        let cp = monitor.check(51, 0.3);
        assert!(cp.is_some());
        let cp = cp.unwrap();
        println!("Checkpoint: patient={}, wealth={:.3}, ratio={:.2}x, recommend_stop={}",
                 cp.patient, cp.wealth, cp.ratio, cp.recommend_stop);

        assert_eq!(monitor.mode, FutilityMode::Alert);
    }

    #[test]
    fn test_futility_monitor_recovery() {
        let config = FutilityConfig::default();
        let mut monitor = FutilityMonitor::new(
            config,
            0.25,
            0.20,
            500,
            20.0,
            50,
            100,
        );

        let cp = monitor.check(51, 5.0);
        assert!(cp.is_some());
        let cp = cp.unwrap();
        assert!(!cp.recommend_stop);
        assert_eq!(cp.mode, FutilityMode::Normal);
    }

    /// Helper: run calibration test for a given ARR
    fn run_calibration(p_ctrl: f64, arr: f64, n_total: usize, n_sims: usize, seed: u64) -> (f64, f64, f64, f64) {
        let p_trt = p_ctrl - arr;
        let threshold = 20.0;
        let burn_in = 50;
        let ramp = 100;

        let mut rng = StdRng::seed_from_u64(seed);

        let mut recommended_stop = 0usize;
        let mut would_have_recovered = 0usize;
        let mut never_recommended = 0usize;
        let mut never_recommended_success = 0usize;

        for _ in 0..n_sims {
            let config = FutilityConfig::default();
            let mut monitor = FutilityMonitor::new(
                config, p_ctrl, p_trt, n_total, threshold, burn_in, ramp,
            );
            let mut proc = BinaryERTProcess::new(burn_in, ramp);

            let mut ever_recommended_stop = false;
            let mut crossed = false;

            for i in 1..=n_total {
                let is_trt = rng.gen_bool(0.5);
                let event = rng.gen_bool(if is_trt { p_trt } else { p_ctrl });
                proc.update(i, if event { 1.0 } else { 0.0 }, is_trt);

                if proc.wealth >= threshold {
                    crossed = true;
                }

                if let Some(cp) = monitor.check(i, proc.wealth) {
                    if cp.recommend_stop && !ever_recommended_stop {
                        ever_recommended_stop = true;
                    }
                }
            }

            if ever_recommended_stop {
                recommended_stop += 1;
                if crossed { would_have_recovered += 1; }
            } else {
                never_recommended += 1;
                if crossed { never_recommended_success += 1; }
            }
        }

        let stop_rate = recommended_stop as f64 / n_sims as f64 * 100.0;
        let recovery_rate = if recommended_stop > 0 {
            would_have_recovered as f64 / recommended_stop as f64 * 100.0
        } else { 0.0 };
        let no_stop_success = if never_recommended > 0 {
            never_recommended_success as f64 / never_recommended as f64 * 100.0
        } else { 0.0 };
        let overall_success = (would_have_recovered + never_recommended_success) as f64 / n_sims as f64 * 100.0;

        (stop_rate, recovery_rate, no_stop_success, overall_success)
    }

    /// Test across varying effect sizes
    #[test]
    fn test_futility_varying_effect_sizes() {
        let p_ctrl = 0.30;
        let n_total = 1000;
        let n_sims = 300;  // Reduced for speed, increase for precision

        println!("\n=== FUTILITY CALIBRATION ACROSS EFFECT SIZES ===");
        println!("Control rate: {:.0}%, N={}, Sims={}", p_ctrl * 100.0, n_total, n_sims);
        println!("Recovery target: 10%, Stop ratio: 1.75x\n");
        println!("  ARR  | Stop% | Recovery% | No-Stop Success% | Overall Success%");
        println!("-------|-------|-----------|------------------|------------------");

        let arrs = [0.0, 0.03, 0.05, 0.07, 0.10];

        for (i, &arr) in arrs.iter().enumerate() {
            let (stop_rate, recovery_rate, no_stop_success, overall_success) =
                run_calibration(p_ctrl, arr, n_total, n_sims, 10000 + i as u64);

            println!(" {:>4.0}% | {:>5.1} | {:>9.1} | {:>16.1} | {:>16.1}",
                     arr * 100.0, stop_rate, recovery_rate, no_stop_success, overall_success);
        }

        println!("\nInterpretation:");
        println!("- Recovery%: Of trials where STOP recommended, % that would have succeeded");
        println!("- Target is ~10% (should be low = good calibration)");
        println!("- No-Stop Success%: Of trials without STOP, % that succeeded (should be high)");
    }

    /// Test under H0: should recommend stop frequently
    #[test]
    fn test_futility_under_h0() {
        let (stop_rate, recovery_rate, _, _) = run_calibration(0.25, 0.0, 1000, 300, 54321);

        println!("\n=== H0 VALIDATION (No Effect) ===");
        println!("Recommended STOP: {:.1}%", stop_rate);
        println!("Would have crossed (Type I): {:.1}%", recovery_rate);

        assert!(stop_rate > 50.0, "Should recommend STOP >50% under H0, got {:.1}%", stop_rate);
    }

    /// Quick sanity check for basic functionality
    #[test]
    fn test_futility_monitor_sanity() {
        let config = FutilityConfig::default();
        assert_eq!(config.stop_ratio, 1.75);
        assert_eq!(config.recovery_target, 0.10);

        let mut monitor = FutilityMonitor::new(
            config, 0.25, 0.20, 500, 20.0, 50, 100,
        );

        // Low wealth should recommend stop
        let cp = monitor.check(51, 0.2);
        assert!(cp.is_some());
        println!("Low wealth: ratio={:.2}x, recommend_stop={}", cp.as_ref().unwrap().ratio, cp.as_ref().unwrap().recommend_stop);

        // Reset and check high wealth
        let config2 = FutilityConfig::default();
        let mut monitor2 = FutilityMonitor::new(
            config2, 0.25, 0.20, 500, 20.0, 50, 100,
        );
        let cp2 = monitor2.check(51, 10.0);
        assert!(cp2.is_some());
        assert!(!cp2.unwrap().recommend_stop, "High wealth should not recommend stop");
    }
}
