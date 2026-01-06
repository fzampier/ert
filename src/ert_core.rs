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

/// Simple timestamp without external crate
pub fn chrono_lite() -> String {
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

