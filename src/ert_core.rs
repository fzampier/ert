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

// ============================================================================
// STATISTICAL HELPERS
// ============================================================================

/// Calculate median of a slice
pub fn median(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n.is_multiple_of(2) {
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

/// Build report output path: place report in same directory as input CSV
pub fn report_path(csv_path: &str, report_name: &str) -> String {
    use std::path::Path;
    let path = Path::new(csv_path);
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            return parent.join(report_name).to_string_lossy().to_string();
        }
    }
    report_name.to_string()
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
// CONTINUOUS e-RTc PROCESS (MAD-based)
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

    /// Anytime-valid confidence sequence for Cohen's d (standardized mean difference).
    ///
    /// Returns (lower, upper) bounds that are valid at any stopping time.
    /// alpha: confidence level (e.g., 0.05 for 95% CI)
    pub fn confidence_sequence_d(&self, alpha: f64) -> (f64, f64) {
        let (n_t, n_c) = self.get_ns();
        let n = (n_t + n_c) as f64;
        if n < 4.0 {
            return (f64::NEG_INFINITY, f64::INFINITY);
        }

        let d = self.current_effect_estimated();
        let n_t = n_t as f64;
        let n_c = n_c as f64;

        // Variance of Cohen's d (Hedges & Olkin approximation)
        // Var(d) ≈ (n_t + n_c) / (n_t * n_c) + d^2 / (2 * (n_t + n_c))
        let var_d = (n_t + n_c) / (n_t * n_c) + d * d / (2.0 * (n_t + n_c));
        let se = var_d.sqrt();

        // Time-uniform critical value (Robbins mixture)
        let log_factor = (2.0 / alpha).ln() + (n.ln()).ln().max(0.0);
        let crit = (2.0 * log_factor).sqrt();

        let margin = crit * se;
        (d - margin, d + margin)
    }

    /// Anytime-valid confidence sequence for raw mean difference.
    ///
    /// Returns (lower, upper) bounds in original outcome units.
    /// alpha: confidence level (e.g., 0.05 for 95% CI)
    pub fn confidence_sequence_mean_diff(&self, alpha: f64) -> (f64, f64) {
        let (n_t, n_c) = self.get_ns();
        let n = (n_t + n_c) as f64;
        if n < 4.0 {
            return (f64::NEG_INFINITY, f64::INFINITY);
        }

        let (m_t, m_c) = self.get_means();
        let diff = m_t - m_c;
        let n_t = n_t as f64;
        let n_c = n_c as f64;

        // Variance of mean difference
        let sd = self.get_pooled_sd();
        let var_diff = sd * sd * (1.0 / n_t + 1.0 / n_c);
        let se = var_diff.sqrt();

        // Time-uniform critical value (Robbins mixture)
        let log_factor = (2.0 / alpha).ln() + (n.ln()).ln().max(0.0);
        let crit = (2.0 * log_factor).sqrt();

        let margin = crit * se;
        (diff - margin, diff + margin)
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
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- Median tests ---

    #[test]
    fn test_median_odd() {
        assert_eq!(median(&[1.0, 2.0, 3.0]), 2.0);
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0); // unsorted input
    }

    #[test]
    fn test_median_even() {
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 2.5);
    }

    #[test]
    fn test_median_single() {
        assert_eq!(median(&[42.0]), 42.0);
    }

    #[test]
    fn test_median_empty() {
        assert_eq!(median(&[]), 0.0);
    }

    // --- MAD tests ---

    #[test]
    fn test_mad_basic() {
        // median = 2, deviations = [1, 0, 1], MAD = 1
        assert_eq!(mad(&[1.0, 2.0, 3.0]), 1.0);
    }

    #[test]
    fn test_mad_constant() {
        // All same values -> MAD = 0
        assert_eq!(mad(&[5.0, 5.0, 5.0, 5.0]), 0.0);
    }

    // --- Normal CDF tests ---

    #[test]
    fn test_normal_cdf_center() {
        let cdf_0 = normal_cdf(0.0);
        assert!((cdf_0 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_normal_cdf_known_values() {
        // P(Z < 1.96) ≈ 0.975
        let cdf_196 = normal_cdf(1.96);
        assert!((cdf_196 - 0.975).abs() < 0.001);

        // P(Z < -1.96) ≈ 0.025
        let cdf_neg196 = normal_cdf(-1.96);
        assert!((cdf_neg196 - 0.025).abs() < 0.001);
    }

    // --- Normal quantile tests ---

    #[test]
    fn test_normal_quantile_center() {
        let q_50 = normal_quantile(0.5);
        assert!(q_50.abs() < 1e-6);
    }

    #[test]
    fn test_normal_quantile_known_values() {
        // Q(0.975) ≈ 1.96
        let q_975 = normal_quantile(0.975);
        assert!((q_975 - 1.96).abs() < 0.01);

        // Q(0.025) ≈ -1.96
        let q_025 = normal_quantile(0.025);
        assert!((q_025 + 1.96).abs() < 0.01);
    }

    #[test]
    fn test_normal_quantile_extremes() {
        assert!(normal_quantile(0.0).is_infinite());
        assert!(normal_quantile(1.0).is_infinite());
    }

    // --- BinaryERTProcess tests ---

    #[test]
    fn test_binary_ert_initial_state() {
        let proc = BinaryERTProcess::new(50, 100);
        assert_eq!(proc.wealth, 1.0);
        assert_eq!(proc.n_trt, 0.0);
        assert_eq!(proc.n_ctrl, 0.0);
    }

    #[test]
    fn test_binary_ert_counts_update() {
        let mut proc = BinaryERTProcess::new(50, 100);
        proc.update(1, 1.0, true);  // treatment, event
        proc.update(2, 0.0, false); // control, no event

        assert_eq!(proc.n_trt, 1.0);
        assert_eq!(proc.events_trt, 1.0);
        assert_eq!(proc.n_ctrl, 1.0);
        assert_eq!(proc.events_ctrl, 0.0);
    }

    #[test]
    fn test_binary_ert_no_betting_during_burnin() {
        let mut proc = BinaryERTProcess::new(50, 100);
        for i in 1..=50 {
            proc.update(i, 1.0, i % 2 == 0);
        }
        // During burn-in, wealth should remain 1.0
        assert_eq!(proc.wealth, 1.0);
    }

    #[test]
    fn test_binary_ert_risk_diff() {
        let mut proc = BinaryERTProcess::new(0, 1);
        // 2 events in 4 treatment patients = 50%
        proc.update(1, 1.0, true);
        proc.update(2, 1.0, true);
        proc.update(3, 0.0, true);
        proc.update(4, 0.0, true);
        // 1 event in 4 control patients = 25%
        proc.update(5, 1.0, false);
        proc.update(6, 0.0, false);
        proc.update(7, 0.0, false);
        proc.update(8, 0.0, false);

        let rd = proc.current_risk_diff();
        assert!((rd - 0.25).abs() < 1e-6); // 50% - 25% = 25%
    }

    #[test]
    fn test_binary_ert_odds_ratio() {
        let mut proc = BinaryERTProcess::new(0, 1);
        // Treatment: 10 events, 10 non-events
        for _ in 0..10 { proc.n_trt += 1.0; proc.events_trt += 1.0; }
        for _ in 0..10 { proc.n_trt += 1.0; }
        // Control: 5 events, 15 non-events
        for _ in 0..5 { proc.n_ctrl += 1.0; proc.events_ctrl += 1.0; }
        for _ in 0..15 { proc.n_ctrl += 1.0; }

        let (or, lo, hi) = proc.current_odds_ratio();
        // OR = (10.5 * 15.5) / (10.5 * 5.5) ≈ 2.82 (with continuity correction)
        assert!(or > 2.0 && or < 4.0);
        assert!(lo < or && or < hi);
    }

    // --- Confidence sequence tests ---

    #[test]
    fn test_confidence_sequence_rd_bounds() {
        let mut proc = BinaryERTProcess::new(0, 1);
        // Add some data
        for i in 1..=100 {
            proc.update(i, if i % 3 == 0 { 1.0 } else { 0.0 }, i % 2 == 0);
        }

        let (lo, hi) = proc.confidence_sequence_rd(0.05);
        let rd = proc.current_risk_diff();

        // CI should contain point estimate
        assert!(lo <= rd && rd <= hi);
        // CI should be within [-1, 1]
        assert!(lo >= -1.0 && hi <= 1.0);
    }

    #[test]
    fn test_confidence_sequence_or_bounds() {
        let mut proc = BinaryERTProcess::new(0, 1);
        // Add some data
        for i in 1..=100 {
            proc.update(i, if i % 3 == 0 { 1.0 } else { 0.0 }, i % 2 == 0);
        }

        let (lo, hi) = proc.confidence_sequence_or(0.05);
        let (or, _, _) = proc.current_odds_ratio();

        // CI should contain point estimate
        assert!(lo <= or && or <= hi);
        // CI should be positive
        assert!(lo >= 0.0);
    }

    #[test]
    fn test_confidence_sequence_small_n() {
        let proc = BinaryERTProcess::new(50, 100);
        // With no data, should return wide bounds
        let (rd_lo, rd_hi) = proc.confidence_sequence_rd(0.05);
        assert_eq!(rd_lo, -1.0);
        assert_eq!(rd_hi, 1.0);

        let (or_lo, or_hi) = proc.confidence_sequence_or(0.05);
        assert_eq!(or_lo, 0.0);
        assert!(or_hi.is_infinite());
    }

    // --- Sample size calculation tests ---

    #[test]
    fn test_calculate_n_binary() {
        // 30% vs 20% should need reasonable N
        let n = calculate_n_binary(0.30, 0.20, 0.80);
        assert!(n > 100 && n < 1000);
    }

    #[test]
    fn test_calculate_n_continuous() {
        // Cohen's d = 0.5 (medium effect)
        let n = calculate_n_continuous(0.5, 0.80);
        assert!(n > 50 && n < 200);
    }

    // --- Power calculation tests ---

    #[test]
    fn test_z_test_power_large_effect() {
        // Large effect + large N should give high power
        let power = z_test_power_binary(0.5, 0.2, 200, 0.05);
        assert!(power > 0.9);
    }

    #[test]
    fn test_z_test_power_no_effect() {
        // No effect should give ~alpha power (type I error)
        let power = z_test_power_binary(0.3, 0.3, 200, 0.05);
        assert!(power < 0.1);
    }

    #[test]
    fn test_t_test_power() {
        // Medium effect (d=0.5), N=128 should give ~80% power
        let power = t_test_power_continuous(0.5, 1.0, 128, 0.05);
        assert!(power > 0.7 && power < 0.9);
    }

    // --- MADProcess tests ---

    #[test]
    fn test_mad_process_initial_state() {
        let proc = MADProcess::new(50, 100, 0.25);
        assert_eq!(proc.wealth, 1.0);
    }

    #[test]
    fn test_mad_process_means() {
        let mut proc = MADProcess::new(0, 1, 0.25);
        // Treatment: 10, 20
        proc.update(1, 10.0, true);
        proc.update(2, 20.0, true);
        // Control: 5, 15
        proc.update(3, 5.0, false);
        proc.update(4, 15.0, false);

        let (m_t, m_c) = proc.get_means();
        assert!((m_t - 15.0).abs() < 1e-6);
        assert!((m_c - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_mad_process_confidence_sequence_d() {
        let mut proc = MADProcess::new(0, 1, 0.25);
        // Add 50 treatment (mean ~10) and 50 control (mean ~5)
        for i in 1..=50 {
            proc.update(i, 10.0 + (i as f64 % 3.0) - 1.0, true);
        }
        for i in 51..=100 {
            proc.update(i, 5.0 + (i as f64 % 3.0) - 1.0, false);
        }

        let d = proc.current_effect_estimated();
        let (lo, hi) = proc.confidence_sequence_d(0.05);

        // CI should contain point estimate
        assert!(lo <= d && d <= hi);
        // Effect should be positive (treatment > control)
        assert!(d > 0.0);
    }
}

