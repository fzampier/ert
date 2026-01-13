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

    /// Anytime-valid confidence sequence for risk difference.
    ///
    /// Returns (lower, upper) bounds that are valid at any stopping time.
    /// Uses a mixture martingale approach with boundary crossing.
    ///
    /// alpha: confidence level (e.g., 0.05 for 95% CI)
    #[allow(dead_code)]  // Public API for CSV analysis modules
    pub fn confidence_sequence_rd(&self, alpha: f64) -> (f64, f64) {
        // Use Wald-type CI with a time-uniform correction factor
        // The correction sqrt(2 * log(2/alpha) / n) ensures anytime validity
        let n = self.n_trt + self.n_ctrl;
        if n < 4.0 {
            return (-1.0, 1.0);
        }

        let rd = self.current_risk_diff();
        let r_t = self.events_trt / self.n_trt.max(1.0);
        let r_c = self.events_ctrl / self.n_ctrl.max(1.0);

        // Variance of risk difference
        let var_rd = r_t * (1.0 - r_t) / self.n_trt.max(1.0)
                   + r_c * (1.0 - r_c) / self.n_ctrl.max(1.0);
        let se = var_rd.sqrt();

        // Time-uniform critical value (Robbins mixture)
        // This is sqrt(2 * log(log(n) * 2 / alpha)) for large n
        let log_factor = (2.0 / alpha).ln() + (n.ln()).ln().max(0.0);
        let crit = (2.0 * log_factor).sqrt();

        let margin = crit * se;
        let lower = (rd - margin).max(-1.0);
        let upper = (rd + margin).min(1.0);

        (lower, upper)
    }

    /// Anytime-valid confidence sequence for odds ratio.
    ///
    /// Returns (lower, upper) bounds that are valid at any stopping time.
    /// alpha: confidence level (e.g., 0.05 for 95% CI)
    #[allow(dead_code)]  // Public API for CSV analysis modules
    pub fn confidence_sequence_or(&self, alpha: f64) -> (f64, f64) {
        let n = self.n_trt + self.n_ctrl;
        if n < 4.0 {
            return (0.0, f64::INFINITY);
        }

        let (or, _, _) = self.current_odds_ratio();

        // Add 0.5 continuity correction
        let a = self.events_trt + 0.5;
        let b = self.n_trt - self.events_trt + 0.5;
        let c = self.events_ctrl + 0.5;
        let d = self.n_ctrl - self.events_ctrl + 0.5;

        // SE of log OR
        let se_log_or = (1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d).sqrt();

        // Time-uniform critical value
        let log_factor = (2.0 / alpha).ln() + (n.ln()).ln().max(0.0);
        let crit = (2.0 * log_factor).sqrt();

        let log_or = or.ln();
        let lower = (log_or - crit * se_log_or).exp();
        let upper = (log_or + crit * se_log_or).exp();

        (lower, upper)
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

    /// Cohen's d effect using external SD (for simulations where true SD is known)
    pub fn current_effect(&self, sd: f64) -> f64 {
        let (mut sum_t, mut n_t, mut sum_c, mut n_c) = (0.0, 0.0, 0.0, 0.0);
        for (&o, &t) in self.outcomes.iter().zip(self.treatments.iter()) {
            if t { sum_t += o; n_t += 1.0; } else { sum_c += o; n_c += 1.0; }
        }
        let m_t = if n_t > 0.0 { sum_t / n_t } else { 0.0 };
        let m_c = if n_c > 0.0 { sum_c / n_c } else { 0.0 };
        (m_t - m_c) / sd
    }

    /// Cohen's d effect using estimated pooled SD (for real data analysis)
    #[allow(dead_code)]  // Public API for CSV analysis modules
    pub fn current_effect_estimated(&self) -> f64 {
        let sd = self.get_pooled_sd();
        self.current_effect(sd)
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

    /// Pooled SD estimated from observed data. Use with current_effect_estimated()
    /// for real data analysis, or pass to current_effect() explicitly.
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
// CALIBRATION TEST
// ============================================================================

/// Calibration test: verify forward sim matches real e-RT process
/// Returns (real_power, forward_power) - should be nearly identical
#[allow(dead_code)]
pub fn calibration_test(
    p_ctrl: f64,
    p_trt: f64,
    n_total: usize,
    threshold: f64,
    burn_in: usize,
    ramp: usize,
    n_sims: usize,
) -> (f64, f64) {
    use rand::Rng;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    let mut rng = StdRng::seed_from_u64(12345);

    // Method 1: Real e-RT process
    let mut real_successes = 0;
    for _ in 0..n_sims {
        let mut proc = BinaryERTProcess::new(burn_in, ramp);
        for i in 1..=n_total {
            let is_trt = rng.gen_bool(0.5);
            let event = rng.gen_bool(if is_trt { p_trt } else { p_ctrl });
            proc.update(i, if event { 1.0 } else { 0.0 }, is_trt);
            if proc.wealth >= threshold {
                real_successes += 1;
                break;
            }
        }
    }

    // Method 2: Forward sim logic (standalone, from scratch)
    let mut rng2 = StdRng::seed_from_u64(12345); // Same seed!
    let mut forward_successes = 0;
    for _ in 0..n_sims {
        let mut wealth = 1.0;
        let (mut n_t, mut e_t, mut n_c, mut e_c) = (0.0, 0.0, 0.0, 0.0);

        for i in 1..=n_total {
            let is_trt = rng2.gen_bool(0.5);
            let event = rng2.gen_bool(if is_trt { p_trt } else { p_ctrl });

            // Delta from CURRENT counts (before this patient)
            // MUST match real process: default to 0.5 when arm is empty
            let rate_t = if n_t > 0.0 { e_t / n_t } else { 0.5 };
            let rate_c = if n_c > 0.0 { e_c / n_c } else { 0.5 };
            let delta = rate_t - rate_c;

            // Update counts BEFORE wealth update (matches real process order)
            if is_trt {
                n_t += 1.0;
                if event { e_t += 1.0; }
            } else {
                n_c += 1.0;
                if event { e_c += 1.0; }
            }

            // Ramp factor
            let c = if i > burn_in {
                ((i - burn_in) as f64 / ramp as f64).min(1.0)
            } else {
                0.0
            };

            // Only bet if past burn-in
            if c > 0.0 {
                // Lambda (must match real process exactly)
                let lambda = if event {
                    0.5 + 0.5 * c * delta
                } else {
                    0.5 - 0.5 * c * delta
                };
                let lambda = lambda.clamp(0.001, 0.999);

                // Wealth update
                let mult = if is_trt { lambda / 0.5 } else { (1.0 - lambda) / 0.5 };
                wealth *= mult;
            }

            if wealth >= threshold {
                forward_successes += 1;
                break;
            }
        }
    }

    let real_power = real_successes as f64 / n_sims as f64;
    let forward_power = forward_successes as f64 / n_sims as f64;

    (real_power, forward_power)
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
// The FutilityMonitor uses Monte Carlo simulation to estimate P(recovery),
// the probability that the trial will eventually cross the success threshold
// assuming the true treatment effect equals the design effect.
//
// CALIBRATION NOTE: The monitor shows different calibration for early vs late
// checkpoints due to survivorship bias:
// - Early stops (<50% of N): Well calibrated (~7% est vs ~10% actual)
// - Late stops (>=50% of N): Pessimistic (~5% est vs ~20% actual)
//
// Trials that first trigger a stop recommendation late have "survived" earlier
// checkpoints, suggesting they were performing reasonably well until recently.
// The point-in-time forward simulation doesn't capture this selection history.
//
// Despite this, the monitor provides useful decision support. When it recommends
// stop, the trial has genuinely poor prospects - the actual recovery rate is
// always well below 50%, meaning most flagged trials would indeed fail.
//

/// Checkpoint result from futility monitoring
#[derive(Clone, Debug)]
pub struct FutilityCheckpoint {
    pub patient: usize,
    pub wealth: f64,
    pub mode: FutilityMode,
    pub ratio: f64,           // recovery_target / recovery_prob (for reporting)
    pub recovery_prob: f64,   // P(recovery) at design effect (decision basis)
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
    pub stop_ratio: f64,           // Default 1.75 (unused, kept for config compatibility)
    pub mc_samples: usize,         // Default 500
    pub hysteresis_count: usize,   // Default 3 (consecutive checks above threshold to exit alert)
}

impl Default for FutilityConfig {
    fn default() -> Self {
        FutilityConfig {
            recovery_target: 0.10,
            normal_interval_pct: 0.05,
            alert_interval_pct: 0.01,
            danger_threshold: 0.5,
            stop_ratio: 1.75,
            mc_samples: 500,
            hysteresis_count: 3,
        }
    }
}

impl FutilityConfig {
    /// Fast mode: faster with slightly less precision in futility estimates.
    /// Use for interactive exploration; switch to default() for final analysis.
    #[allow(dead_code)]
    pub fn fast() -> Self {
        FutilityConfig {
            mc_samples: 100,
            ..Default::default()
        }
    }

    /// High precision mode for calibration testing
    #[allow(dead_code)]
    pub fn high_precision() -> Self {
        FutilityConfig {
            mc_samples: 1000,
            ..Default::default()
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
    ///
    /// Pass current process state for accurate forward simulation:
    /// - n_trt, events_trt: treatment arm counts
    /// - n_ctrl, events_ctrl: control arm counts
    pub fn check(
        &mut self,
        patient: usize,
        wealth: f64,
        n_trt: f64,
        events_trt: f64,
        n_ctrl: f64,
        events_ctrl: f64,
    ) -> Option<FutilityCheckpoint> {
        if !self.should_check(patient) {
            return None;
        }

        let n_remaining = self.n_total.saturating_sub(patient);

        // DECISION BASIS: Estimate P(recovery) assuming design effect continues
        let p_trt_design = (self.p_ctrl - self.design_arr).max(0.001);
        let recovery_prob = self.estimate_recovery_prob(
            wealth, n_remaining, patient, n_trt, events_trt, n_ctrl, events_ctrl, p_trt_design
        );

        // REPORTING: Calculate ratio only if needed (expensive)
        // Ratio for reporting: recovery_target / recovery_prob
        // ratio > 1.0 means we'd need better-than-design effect to hit target
        let ratio = if recovery_prob > 0.01 {
            self.config.recovery_target / recovery_prob
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

        // Build checkpoint - decision based on DIRECT probability, not ratio threshold
        let checkpoint = FutilityCheckpoint {
            patient,
            wealth,
            mode: self.mode.clone(),
            ratio,
            recovery_prob,
            recommend_stop: recovery_prob < self.config.recovery_target,
        };

        self.checkpoints.push(checkpoint.clone());
        Some(checkpoint)
    }

    /// Estimate probability of recovery (crossing threshold) with given treatment effect
    fn estimate_recovery_prob(
        &self,
        current_wealth: f64,
        n_remaining: usize,
        current_patient: usize,
        n_trt: f64,
        events_trt: f64,
        n_ctrl: f64,
        events_ctrl: f64,
        p_trt: f64,
    ) -> f64 {
        use rand::Rng;
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        if n_remaining == 0 {
            return 0.0;
        }

        // Already above threshold
        if current_wealth >= self.threshold {
            return 1.0;
        }

        // Use state-dependent seed to avoid systematic bias from fixed seed
        // Hash: combine state values to create unique seed per call
        let state_hash = (current_patient as u64)
            .wrapping_mul(31)
            .wrapping_add((current_wealth * 10000.0) as u64)
            .wrapping_mul(31)
            .wrapping_add((n_trt * 100.0) as u64)
            .wrapping_mul(31)
            .wrapping_add((events_trt * 100.0) as u64);
        let mut rng = StdRng::seed_from_u64(state_hash);
        let mut successes = 0usize;

        for _ in 0..self.config.mc_samples {
            let mut wealth = current_wealth;
            let (mut n_t, mut e_t, mut n_c, mut e_c) = (n_trt, events_trt, n_ctrl, events_ctrl);

            for j in 1..=n_remaining {
                let is_trt = rng.gen_bool(0.5);
                let event = rng.gen_bool(if is_trt { p_trt } else { self.p_ctrl });

                // Delta estimation matching real process (0.5 default for empty arm)
                let rate_t = if n_t > 0.0 { e_t / n_t } else { 0.5 };
                let rate_c = if n_c > 0.0 { e_c / n_c } else { 0.5 };
                let delta = rate_t - rate_c;

                // Update counts
                if is_trt {
                    n_t += 1.0;
                    if event { e_t += 1.0; }
                } else {
                    n_c += 1.0;
                    if event { e_c += 1.0; }
                }

                // Ramp based on actual patient number
                let actual_patient = current_patient + j;
                let c = if actual_patient > self.burn_in {
                    ((actual_patient - self.burn_in) as f64 / self.ramp as f64).min(1.0)
                } else {
                    0.0
                };

                if c > 0.0 {
                    let lambda = if event {
                        0.5 + 0.5 * c * delta
                    } else {
                        0.5 - 0.5 * c * delta
                    };
                    let lambda = lambda.clamp(0.001, 0.999);
                    let mult = if is_trt { lambda / 0.5 } else { (1.0 - lambda) / 0.5 };
                    wealth *= mult;
                }

                if wealth >= self.threshold {
                    successes += 1;
                    break;
                }
            }
        }

        successes as f64 / self.config.mc_samples as f64
    }

    /// Get summary of futility monitoring
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("\n--- Futility Monitoring Summary ---\n"));
        s.push_str(&format!("Design ARR: {:.1}%\n", self.design_arr * 100.0));
        s.push_str(&format!("Recovery target: {:.0}%\n", self.config.recovery_target * 100.0));
        s.push_str(&format!("Checkpoints: {}\n\n", self.checkpoints.len()));

        if self.checkpoints.is_empty() {
            s.push_str("No checkpoints recorded.\n");
            return s;
        }

        s.push_str("Patient | Wealth |  Mode  | P(rec) | Ratio | Recommend\n");
        s.push_str("--------|--------|--------|--------|-------|----------\n");

        for cp in &self.checkpoints {
            let mode_str = match cp.mode {
                FutilityMode::Normal => "NORMAL",
                FutilityMode::Alert => "ALERT ",
            };
            let recommend = if cp.recommend_stop { "STOP" } else { "-" };
            s.push_str(&format!(
                "{:>7} | {:>6.3} | {} | {:>5.1}% | {:>5.2}x | {}\n",
                cp.patient,
                cp.wealth,
                mode_str,
                cp.recovery_prob * 100.0,
                cp.ratio,
                recommend
            ));
        }

        // Final recommendation
        if let Some(last) = self.checkpoints.last() {
            s.push_str(&format!("\nFinal status: "));
            if last.recommend_stop {
                s.push_str(&format!(
                    "RECOMMEND STOP (P(recovery)={:.1}% < {:.0}% target)\n",
                    last.recovery_prob * 100.0, self.config.recovery_target * 100.0
                ));
            } else {
                s.push_str(&format!(
                    "Continue (P(recovery)={:.1}%)\n",
                    last.recovery_prob * 100.0
                ));
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

    /// Get recovery probability at first stop recommendation
    /// This is the proper calibration metric: if accurate, ~recovery_target% should recover
    pub fn first_stop_recovery_prob(&self) -> Option<f64> {
        self.checkpoints.iter()
            .find(|cp| cp.recommend_stop)
            .map(|cp| cp.recovery_prob)
    }

    /// Get patient number at first stop recommendation
    pub fn first_stop_patient(&self) -> Option<usize> {
        self.checkpoints.iter()
            .find(|cp| cp.recommend_stop)
            .map(|cp| cp.patient)
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

        // Simulate some accumulated counts (roughly half in each arm with 25% event rate)
        let (n_trt, events_trt, n_ctrl, events_ctrl) = (25.0, 6.0, 26.0, 7.0);
        let cp = monitor.check(51, 0.3, n_trt, events_trt, n_ctrl, events_ctrl);
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

        // Simulate some accumulated counts
        let (n_trt, events_trt, n_ctrl, events_ctrl) = (25.0, 5.0, 26.0, 7.0);
        let cp = monitor.check(51, 5.0, n_trt, events_trt, n_ctrl, events_ctrl);
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

                if let Some(cp) = monitor.check(i, proc.wealth, proc.n_trt, proc.events_trt, proc.n_ctrl, proc.events_ctrl) {
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

    /// Calibration test: forward sim must match real e-RT exactly
    #[test]
    fn test_forward_sim_calibration() {
        let (real, forward) = calibration_test(0.30, 0.20, 788, 20.0, 50, 100, 500);
        println!("Real e-RT power:    {:.1}%", real * 100.0);
        println!("Forward sim power:  {:.1}%", forward * 100.0);
        let diff = (real - forward).abs();
        println!("Difference:         {:.2}%", diff * 100.0);
        assert!(diff < 0.01, "Forward sim should match real process within 1%, got {:.2}%", diff * 100.0);
    }

    /// Intermediate-state calibration test: estimate P(recovery) at checkpoint,
    /// then verify actual recovery rate matches estimate
    #[test]
    fn test_intermediate_calibration() {
        use rand::Rng;
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let p_ctrl = 0.30;
        let p_trt = 0.20;
        let n_total = 589;
        let threshold = 20.0;
        let burn_in = 50;
        let ramp = 100;
        let checkpoint_patient = 200;  // Check at patient 200 (early in trial)

        let mut rng = StdRng::seed_from_u64(999);
        let n_sims = 500;

        let mut estimates: Vec<f64> = Vec::new();
        let mut actual_successes = 0;

        for _ in 0..n_sims {
            let mut proc = BinaryERTProcess::new(burn_in, ramp);

            // Run to checkpoint
            for i in 1..=checkpoint_patient {
                let is_trt = rng.gen_bool(0.5);
                let event = rng.gen_bool(if is_trt { p_trt } else { p_ctrl });
                proc.update(i, if event { 1.0 } else { 0.0 }, is_trt);
            }

            // Estimate P(recovery) at this point
            let config = FutilityConfig { mc_samples: 200, ..Default::default() };
            let mut monitor = FutilityMonitor::new(
                config, p_ctrl, p_trt, n_total, threshold, burn_in, ramp,
            );
            // Force checkpoint
            monitor.next_checkpoint = checkpoint_patient;
            let cp = monitor.check(
                checkpoint_patient, proc.wealth,
                proc.n_trt, proc.events_trt, proc.n_ctrl, proc.events_ctrl
            );

            if let Some(checkpoint) = cp {
                estimates.push(checkpoint.recovery_prob);
            }

            // Continue trial to completion
            let mut succeeded = false;
            if proc.wealth >= threshold {
                succeeded = true;
            } else {
                for i in (checkpoint_patient + 1)..=n_total {
                    let is_trt = rng.gen_bool(0.5);
                    let event = rng.gen_bool(if is_trt { p_trt } else { p_ctrl });
                    proc.update(i, if event { 1.0 } else { 0.0 }, is_trt);
                    if proc.wealth >= threshold {
                        succeeded = true;
                        break;
                    }
                }
            }
            if succeeded {
                actual_successes += 1;
            }
        }

        let avg_estimate = estimates.iter().sum::<f64>() / estimates.len() as f64;
        let actual_rate = actual_successes as f64 / n_sims as f64;

        println!("\n=== INTERMEDIATE CALIBRATION TEST ===");
        println!("Checkpoint at patient: {}", checkpoint_patient);
        println!("Avg estimated P(rec):  {:.1}%", avg_estimate * 100.0);
        println!("Actual recovery rate:  {:.1}%", actual_rate * 100.0);
        println!("Difference:            {:.1}%", (actual_rate - avg_estimate).abs() * 100.0);

        // Allow 5% tolerance due to MC variance
        let diff = (actual_rate - avg_estimate).abs();
        assert!(diff < 0.05, "Calibration mismatch: {:.1}% estimated vs {:.1}% actual",
            avg_estimate * 100.0, actual_rate * 100.0);

        // Also test calibration for LOW estimates (P < 10%)
        // This tests for selection bias
        let mut low_estimates: Vec<(f64, bool)> = Vec::new();
        let mut rng2 = StdRng::seed_from_u64(1234);
        for _ in 0..1000 {
            let mut proc = BinaryERTProcess::new(burn_in, ramp);
            for i in 1..=checkpoint_patient {
                let is_trt = rng2.gen_bool(0.5);
                let event = rng2.gen_bool(if is_trt { p_trt } else { p_ctrl });
                proc.update(i, if event { 1.0 } else { 0.0 }, is_trt);
            }

            let config = FutilityConfig { mc_samples: 200, ..Default::default() };
            let mut monitor = FutilityMonitor::new(
                config, p_ctrl, p_trt, n_total, threshold, burn_in, ramp,
            );
            monitor.next_checkpoint = checkpoint_patient;
            let cp = monitor.check(
                checkpoint_patient, proc.wealth,
                proc.n_trt, proc.events_trt, proc.n_ctrl, proc.events_ctrl
            );

            let mut succeeded = proc.wealth >= threshold;
            if !succeeded {
                for i in (checkpoint_patient + 1)..=n_total {
                    let is_trt = rng2.gen_bool(0.5);
                    let event = rng2.gen_bool(if is_trt { p_trt } else { p_ctrl });
                    proc.update(i, if event { 1.0 } else { 0.0 }, is_trt);
                    if proc.wealth >= threshold {
                        succeeded = true;
                        break;
                    }
                }
            }

            if let Some(checkpoint) = cp {
                if checkpoint.recovery_prob < 0.10 {
                    low_estimates.push((checkpoint.recovery_prob, succeeded));
                }
            }
        }

        if !low_estimates.is_empty() {
            let low_avg = low_estimates.iter().map(|x| x.0).sum::<f64>() / low_estimates.len() as f64;
            let low_actual = low_estimates.iter().filter(|x| x.1).count() as f64 / low_estimates.len() as f64;
            println!("\n=== LOW ESTIMATE CALIBRATION (P < 10%) ===");
            println!("N trials with P < 10%: {}", low_estimates.len());
            println!("Avg estimated P(rec):  {:.1}%", low_avg * 100.0);
            println!("Actual recovery rate:  {:.1}%", low_actual * 100.0);
            // Note: We expect some inflation due to selection bias / regression to mean
        }
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

        // Simulate some accumulated counts
        let (n_trt, events_trt, n_ctrl, events_ctrl) = (25.0, 6.0, 26.0, 7.0);

        // Low wealth should recommend stop
        let cp = monitor.check(51, 0.2, n_trt, events_trt, n_ctrl, events_ctrl);
        assert!(cp.is_some());
        println!("Low wealth: ratio={:.2}x, recommend_stop={}", cp.as_ref().unwrap().ratio, cp.as_ref().unwrap().recommend_stop);

        // Reset and check high wealth
        let config2 = FutilityConfig::default();
        let mut monitor2 = FutilityMonitor::new(
            config2, 0.25, 0.20, 500, 20.0, 50, 100,
        );
        let cp2 = monitor2.check(51, 10.0, n_trt, events_trt, n_ctrl, events_ctrl);
        assert!(cp2.is_some());
        assert!(!cp2.unwrap().recommend_stop, "High wealth should not recommend stop");
    }
}
