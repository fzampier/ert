//! Analyze real continuous trial data from CSV

use std::error::Error;
use std::fs::File;
use std::io::Write;
use csv::ReaderBuilder;
use serde::Deserialize;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::ert_core::{
    get_input, get_input_usize, get_bool, get_string, get_choice,
    median, mad, chrono_lite,
};

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/// Raw CSV row (may have NA values)
#[derive(Debug, Deserialize)]
struct CsvRowRaw {
    #[serde(default)]
    #[allow(dead_code)]
    _index: Option<String>,  // Handle R-style row index column
    treatment: String,
    outcome: String,
}

/// Validated CSV row
#[derive(Debug, Clone)]
struct CsvRow {
    treatment: u8,
    outcome: f64,
}

/// Method choice
#[derive(Clone, Copy, PartialEq)]
enum Method {
    LinearERT,
    MAD,
}

/// Design assumptions for futility analysis
#[derive(Clone)]
struct DesignParams {
    control_mean: f64,
    treatment_mean: f64,
    sd: f64,
    design_effect_linear: f64,  // Mean difference
    design_effect_mad: f64,     // Cohen's d
}

/// A single futility checkpoint measurement
#[derive(Clone)]
struct FutilityPoint {
    patient_num: usize,
    wealth: f64,
    required_effect: f64,
    ratio_to_design: f64,
}

/// Futility analysis summary
struct FutilityAnalysis {
    design: DesignParams,
    points: Vec<FutilityPoint>,
    regions: Vec<(usize, usize)>,
    worst_point: Option<FutilityPoint>,
    ever_triggered: bool,
}

/// Result of analyzing real trial data
struct AnalysisResult {
    method: Method,
    n_total: usize,
    n_trt: usize,
    n_ctrl: usize,

    // Did we cross threshold?
    crossed: bool,
    crossed_at: Option<usize>,

    // Effect estimates at crossing (if crossed)
    effect_at_cross: Option<f64>,

    // Final effect estimates
    final_effect: f64,
    final_mean_trt: f64,
    final_mean_ctrl: f64,
    final_sd: f64,

    // Final e-value
    final_evalue: f64,

    // Type M error (if crossed)
    type_m: Option<f64>,

    // Trajectory for plotting
    trajectory: Vec<f64>,

    // Futility analysis (if enabled)
    futility_analysis: Option<FutilityAnalysis>,
}

// ============================================================================
// LinearERT PROCESS (for bounded continuous data)
// ============================================================================

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

    fn get_means(&self) -> (f64, f64) {
        let mean_trt = if self.n_trt > 0.0 { self.sum_trt / self.n_trt } else { 0.0 };
        let mean_ctrl = if self.n_ctrl > 0.0 { self.sum_ctrl / self.n_ctrl } else { 0.0 };
        (mean_trt, mean_ctrl)
    }

    fn get_ns(&self) -> (usize, usize) {
        (self.n_trt as usize, self.n_ctrl as usize)
    }
}

// ============================================================================
// MAD PROCESS (for unbounded continuous data)
// ============================================================================

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
        let direction = if !self.outcomes.is_empty() {
            let trt_vals: Vec<f64> = self.outcomes.iter().zip(self.treatments.iter())
                .filter(|(_, &t)| t).map(|(&o, _)| o).collect();
            let ctrl_vals: Vec<f64> = self.outcomes.iter().zip(self.treatments.iter())
                .filter(|(_, &t)| !t).map(|(&o, _)| o).collect();

            let mean_trt = if !trt_vals.is_empty() { trt_vals.iter().sum::<f64>() / trt_vals.len() as f64 } else { 0.0 };
            let mean_ctrl = if !ctrl_vals.is_empty() { ctrl_vals.iter().sum::<f64>() / ctrl_vals.len() as f64 } else { 0.0 };

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

        let mean_trt = if !trt_vals.is_empty() { trt_vals.iter().sum::<f64>() / trt_vals.len() as f64 } else { 0.0 };
        let mean_ctrl = if !ctrl_vals.is_empty() { ctrl_vals.iter().sum::<f64>() / ctrl_vals.len() as f64 } else { 0.0 };

        (mean_trt - mean_ctrl) / sd
    }

    fn get_means(&self) -> (f64, f64) {
        let trt_vals: Vec<f64> = self.outcomes.iter().zip(self.treatments.iter())
            .filter(|(_, &t)| t).map(|(&o, _)| o).collect();
        let ctrl_vals: Vec<f64> = self.outcomes.iter().zip(self.treatments.iter())
            .filter(|(_, &t)| !t).map(|(&o, _)| o).collect();

        let mean_trt = if !trt_vals.is_empty() { trt_vals.iter().sum::<f64>() / trt_vals.len() as f64 } else { 0.0 };
        let mean_ctrl = if !ctrl_vals.is_empty() { ctrl_vals.iter().sum::<f64>() / ctrl_vals.len() as f64 } else { 0.0 };

        (mean_trt, mean_ctrl)
    }

    fn get_ns(&self) -> (usize, usize) {
        let n_trt = self.treatments.iter().filter(|&&t| t).count();
        let n_ctrl = self.treatments.len() - n_trt;
        (n_trt, n_ctrl)
    }

    fn get_pooled_sd(&self) -> f64 {
        let trt_vals: Vec<f64> = self.outcomes.iter().zip(self.treatments.iter())
            .filter(|(_, &t)| t).map(|(&o, _)| o).collect();
        let ctrl_vals: Vec<f64> = self.outcomes.iter().zip(self.treatments.iter())
            .filter(|(_, &t)| !t).map(|(&o, _)| o).collect();

        let n1 = trt_vals.len() as f64;
        let n2 = ctrl_vals.len() as f64;

        if n1 < 2.0 || n2 < 2.0 {
            return 1.0;
        }

        let mean1 = trt_vals.iter().sum::<f64>() / n1;
        let mean2 = ctrl_vals.iter().sum::<f64>() / n2;

        let ss1: f64 = trt_vals.iter().map(|x| (x - mean1).powi(2)).sum();
        let ss2: f64 = ctrl_vals.iter().map(|x| (x - mean2).powi(2)).sum();

        ((ss1 + ss2) / (n1 + n2 - 2.0)).sqrt()
    }
}

// ============================================================================
// MAIN ENTRY POINT
// ============================================================================

pub fn run() -> Result<(), Box<dyn Error>> {
    println!("\n==========================================");
    println!("   ANALYZE CONTINUOUS TRIAL DATA");
    println!("==========================================\n");

    // Get CSV file path
    let csv_path = get_string("Path to CSV file: ");

    // Read and parse CSV
    println!("\nReading {}...", csv_path);
    let data = read_csv(&csv_path)?;
    let n_total = data.len();

    if n_total == 0 {
        println!("Error: CSV file is empty or has no valid rows.");
        return Ok(());
    }

    // Summary statistics
    let trt_outcomes: Vec<f64> = data.iter().filter(|r| r.treatment == 1).map(|r| r.outcome).collect();
    let ctrl_outcomes: Vec<f64> = data.iter().filter(|r| r.treatment == 0).map(|r| r.outcome).collect();

    let n_trt = trt_outcomes.len();
    let n_ctrl = ctrl_outcomes.len();

    let mean_trt = if n_trt > 0 { trt_outcomes.iter().sum::<f64>() / n_trt as f64 } else { 0.0 };
    let mean_ctrl = if n_ctrl > 0 { ctrl_outcomes.iter().sum::<f64>() / n_ctrl as f64 } else { 0.0 };

    let min_val = data.iter().map(|r| r.outcome).fold(f64::INFINITY, f64::min);
    let max_val = data.iter().map(|r| r.outcome).fold(f64::NEG_INFINITY, f64::max);

    // Pooled SD
    let ss_trt: f64 = trt_outcomes.iter().map(|x| (x - mean_trt).powi(2)).sum();
    let ss_ctrl: f64 = ctrl_outcomes.iter().map(|x| (x - mean_ctrl).powi(2)).sum();
    let pooled_sd = if n_trt > 1 && n_ctrl > 1 {
        ((ss_trt + ss_ctrl) / (n_trt + n_ctrl - 2) as f64).sqrt()
    } else {
        1.0
    };

    println!("\n--- Data Summary ---");
    println!("Total patients:     {}", n_total);
    println!("Treatment arm:      {} (mean: {:.2})", n_trt, mean_trt);
    println!("Control arm:        {} (mean: {:.2})", n_ctrl, mean_ctrl);
    println!("Outcome range:      [{:.2}, {:.2}]", min_val, max_val);
    println!("Pooled SD:          {:.2}", pooled_sd);
    println!("Observed diff:      {:.2}", mean_trt - mean_ctrl);
    println!("Observed Cohen's d: {:.2}", (mean_trt - mean_ctrl) / pooled_sd);

    // Method selection
    let method_choice = get_choice("\nSelect analysis method:", &[
        "e-RTo (ordinal/bounded, e.g., VFD 0-28)",
        "e-RTc (continuous/unbounded)",
    ]);

    let method = if method_choice == 1 { Method::LinearERT } else { Method::MAD };

    // Get bounds if e-RTo
    let (analysis_min, analysis_max) = if method == Method::LinearERT {
        println!("\ne-RTo requires outcome bounds.");
        println!("  Observed range: [{:.2}, {:.2}]", min_val, max_val);
        let mn = get_input("Min bound (e.g., 0): ");
        let mx = get_input("Max bound (e.g., 28): ");
        (mn, mx)
    } else {
        (min_val, max_val)
    };

    // Get analysis parameters
    println!("\n--- Analysis Parameters ---");

    println!("Burn-in period (default = 50):");
    let burn_in = get_input_usize("Burn-in: ");

    println!("Ramp period (default = 100):");
    let ramp = get_input_usize("Ramp: ");

    println!("Success threshold (1/alpha, default = 20 for alpha=0.05):");
    let success_threshold = get_input("Success threshold: ");

    let c_max = if method == Method::MAD {
        println!("c_max for MAD betting (default = 0.6):");
        get_input("c_max: ")
    } else {
        0.6
    };

    // Futility configuration
    let use_futility = get_bool("Enable futility monitoring?");
    let (futility_threshold, design_params) = if use_futility {
        let fut = get_input("Futility threshold (e.g., 0.5): ");

        println!("\n--- Design Assumptions (for futility analysis) ---");
        let design_ctrl = get_input("Design control mean: ");
        let design_trt = get_input("Design treatment mean: ");
        let design_sd = get_input("Design SD: ");

        let design_effect_linear = (design_trt - design_ctrl).abs();
        let design_effect_mad = design_effect_linear / design_sd;

        println!("Design mean diff:   {:.2}", design_effect_linear);
        println!("Design Cohen's d:   {:.2}", design_effect_mad);

        (Some(fut), Some(DesignParams {
            control_mean: design_ctrl,
            treatment_mean: design_trt,
            sd: design_sd,
            design_effect_linear,
            design_effect_mad,
        }))
    } else {
        (None, None)
    };

    // Run analysis
    println!("\n--- Running e-RT Analysis ---");
    let result = analyze_data(
        &data, method, burn_in, ramp, success_threshold, c_max,
        analysis_min, analysis_max, pooled_sd,
        futility_threshold, design_params,
    );

    // Print console results
    print_results(&result, success_threshold, futility_threshold);

    // Optional HTML report
    if get_bool("\nGenerate HTML report?") {
        let html = build_html_report(
            &result, &csv_path, burn_in, ramp,
            success_threshold, futility_threshold, c_max,
            analysis_min, analysis_max,
        );
        let report_path = "continuous_analysis_report.html";
        let mut file = File::create(report_path)?;
        file.write_all(html.as_bytes())?;
        println!("\n>> Report saved: {}", report_path);
    }

    println!("\n==========================================");
    Ok(())
}

// ============================================================================
// CSV PARSING
// ============================================================================

fn read_csv(path: &str) -> Result<Vec<CsvRow>, Box<dyn Error>> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_path(path)?;

    let mut data = Vec::new();
    let mut skipped = 0;

    for result in reader.deserialize() {
        let row: CsvRowRaw = result?;

        let treatment = match row.treatment.parse::<u8>() {
            Ok(v) if v <= 1 => v,
            _ => { skipped += 1; continue; }
        };
        let outcome = match row.outcome.parse::<f64>() {
            Ok(v) if v.is_finite() => v,
            _ => { skipped += 1; continue; }
        };

        data.push(CsvRow { treatment, outcome });
    }

    if skipped > 0 {
        println!("  (Skipped {} rows with NA or invalid values)", skipped);
    }

    Ok(data)
}

// ============================================================================
// MONTE CARLO: REQUIRED EFFECT FOR RECOVERY
// ============================================================================

fn required_effect_linear(
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
    if n_remaining == 0 {
        return max_val - min_val;
    }

    let mut rng = StdRng::seed_from_u64(42);
    let mut low = 0.001;
    let mut high = (max_val - min_val) / 2.0;

    for _ in 0..8 {
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
                let outcome = (rng.gen::<f64>() * 2.0 - 1.0) * sd * 1.5 + mu;
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

                if wealth >= success_threshold {
                    successes += 1;
                    break;
                }
            }
        }

        let rate = successes as f64 / mc_sims as f64;
        if rate < 0.5 { low = mid; } else { high = mid; }
    }

    (low + high) / 2.0
}

fn required_effect_mad(
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
    if n_remaining == 0 {
        return 2.0;  // Cohen's d upper bound
    }

    let mut rng = StdRng::seed_from_u64(42);
    let mut low = 0.001;
    let mut high = 2.0;

    for _ in 0..8 {
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

                let direction = if !outcomes.is_empty() {
                    let trt_vals: Vec<f64> = outcomes.iter().zip(treatments.iter())
                        .filter(|(_, &t)| t).map(|(&o, _)| o).collect();
                    let ctrl_vals: Vec<f64> = outcomes.iter().zip(treatments.iter())
                        .filter(|(_, &t)| !t).map(|(&o, _)| o).collect();
                    let m_t = if !trt_vals.is_empty() { trt_vals.iter().sum::<f64>() / trt_vals.len() as f64 } else { 0.0 };
                    let m_c = if !ctrl_vals.is_empty() { ctrl_vals.iter().sum::<f64>() / ctrl_vals.len() as f64 } else { 0.0 };
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

                if wealth >= success_threshold {
                    successes += 1;
                    break;
                }
            }
        }

        let rate = successes as f64 / mc_sims as f64;
        if rate < 0.5 { low = mid; } else { high = mid; }
    }

    (low + high) / 2.0
}

// ============================================================================
// DATA ANALYSIS
// ============================================================================

fn analyze_data(
    data: &[CsvRow],
    method: Method,
    burn_in: usize,
    ramp: usize,
    success_threshold: f64,
    c_max: f64,
    min_val: f64,
    max_val: f64,
    pooled_sd: f64,
    futility_threshold: Option<f64>,
    design_params: Option<DesignParams>,
) -> AnalysisResult {
    let n_total = data.len();
    let checkpoint_interval = (n_total as f64 * 0.02).ceil() as usize;

    let mut trajectory = Vec::with_capacity(n_total + 1);
    trajectory.push(1.0);

    let mut crossed = false;
    let mut crossed_at: Option<usize> = None;
    let mut effect_at_cross: Option<f64> = None;

    // Futility tracking
    let mut futility_points: Vec<FutilityPoint> = Vec::new();
    let mut futility_regions: Vec<(usize, usize)> = Vec::new();
    let mut in_futility = false;
    let mut futility_start: usize = 0;

    let final_effect: f64;
    let final_mean_trt: f64;
    let final_mean_ctrl: f64;
    let final_sd: f64;
    let n_trt: usize;
    let n_ctrl: usize;
    let final_evalue: f64;

    match method {
        Method::LinearERT => {
            let mut proc = LinearERTProcess::new(burn_in, ramp, min_val, max_val);

            for (i, row) in data.iter().enumerate() {
                let patient_num = i + 1;
                let is_trt = row.treatment == 1;
                let outcome = row.outcome;

                proc.update(patient_num, outcome, is_trt);
                trajectory.push(proc.wealth);

                // Check for success
                if !crossed && proc.wealth >= success_threshold {
                    crossed = true;
                    crossed_at = Some(patient_num);
                    effect_at_cross = Some(proc.current_effect());
                }

                // Futility tracking
                if let (Some(fut_thresh), Some(ref design)) = (futility_threshold, &design_params) {
                    let below_futility = proc.wealth < fut_thresh;

                    if below_futility && !in_futility {
                        in_futility = true;
                        futility_start = patient_num;
                    } else if !below_futility && in_futility {
                        in_futility = false;
                        futility_regions.push((futility_start, patient_num - 1));
                    }

                    if below_futility && patient_num % checkpoint_interval == 0 && patient_num > burn_in {
                        let n_remaining = n_total - patient_num;
                        let required = required_effect_linear(
                            proc.wealth, n_remaining, design.control_mean, design.sd,
                            min_val, max_val, burn_in, ramp, success_threshold, 100,
                        );

                        futility_points.push(FutilityPoint {
                            patient_num,
                            wealth: proc.wealth,
                            required_effect: required,
                            ratio_to_design: required / design.design_effect_linear,
                        });
                    }
                }
            }

            if in_futility {
                futility_regions.push((futility_start, n_total));
            }

            final_effect = proc.current_effect();
            let (mt, mc) = proc.get_means();
            final_mean_trt = mt;
            final_mean_ctrl = mc;
            final_sd = pooled_sd;
            let (nt, nc) = proc.get_ns();
            n_trt = nt;
            n_ctrl = nc;
            final_evalue = proc.wealth;
        }

        Method::MAD => {
            let mut proc = MADProcess::new(burn_in, ramp, c_max);

            for (i, row) in data.iter().enumerate() {
                let patient_num = i + 1;
                let is_trt = row.treatment == 1;
                let outcome = row.outcome;

                proc.update(patient_num, outcome, is_trt);
                trajectory.push(proc.wealth);

                // Check for success
                if !crossed && proc.wealth >= success_threshold {
                    crossed = true;
                    crossed_at = Some(patient_num);
                    effect_at_cross = Some(proc.current_effect(pooled_sd));
                }

                // Futility tracking
                if let (Some(fut_thresh), Some(ref design)) = (futility_threshold, &design_params) {
                    let below_futility = proc.wealth < fut_thresh;

                    if below_futility && !in_futility {
                        in_futility = true;
                        futility_start = patient_num;
                    } else if !below_futility && in_futility {
                        in_futility = false;
                        futility_regions.push((futility_start, patient_num - 1));
                    }

                    if below_futility && patient_num % checkpoint_interval == 0 && patient_num > burn_in {
                        let n_remaining = n_total - patient_num;
                        let required = required_effect_mad(
                            proc.wealth, n_remaining, design.control_mean, design.sd,
                            burn_in, ramp, c_max, success_threshold, 100,
                        );

                        futility_points.push(FutilityPoint {
                            patient_num,
                            wealth: proc.wealth,
                            required_effect: required,
                            ratio_to_design: required / design.design_effect_mad,
                        });
                    }
                }
            }

            if in_futility {
                futility_regions.push((futility_start, n_total));
            }

            final_effect = proc.current_effect(proc.get_pooled_sd());
            let (mt, mc) = proc.get_means();
            final_mean_trt = mt;
            final_mean_ctrl = mc;
            final_sd = proc.get_pooled_sd();
            let (nt, nc) = proc.get_ns();
            n_trt = nt;
            n_ctrl = nc;
            final_evalue = proc.wealth;
        }
    }

    let type_m = if crossed {
        let eff_cross = effect_at_cross.unwrap().abs();
        let eff_final = final_effect.abs();
        if eff_final > 0.0 { Some(eff_cross / eff_final) } else { None }
    } else {
        None
    };

    // Build futility analysis
    let futility_analysis = if let Some(design) = design_params {
        let worst_point = futility_points.iter()
            .max_by(|a, b| a.ratio_to_design.partial_cmp(&b.ratio_to_design).unwrap())
            .cloned();

        let ever_triggered = !futility_regions.is_empty();

        Some(FutilityAnalysis {
            design,
            points: futility_points,
            regions: futility_regions,
            worst_point,
            ever_triggered,
        })
    } else {
        None
    };

    AnalysisResult {
        method,
        n_total,
        n_trt,
        n_ctrl,
        crossed,
        crossed_at,
        effect_at_cross,
        final_effect,
        final_mean_trt,
        final_mean_ctrl,
        final_sd,
        final_evalue,
        type_m,
        trajectory,
        futility_analysis,
    }
}

// ============================================================================
// CONSOLE OUTPUT
// ============================================================================

fn print_results(result: &AnalysisResult, success_threshold: f64, futility_threshold: Option<f64>) {
    let method_name = match result.method {
        Method::LinearERT => "e-RTo",
        Method::MAD => "e-RTc",
    };

    let effect_label = match result.method {
        Method::LinearERT => "Mean Difference",
        Method::MAD => "Cohen's d",
    };

    println!("\n==========================================");
    println!("   RESULTS ({})", method_name);
    println!("==========================================");

    // e-value result
    println!("\n--- e-Value ---");
    println!("Final e-value:      {:.4}", result.final_evalue);
    println!("Threshold:          {:.1}", success_threshold);

    if result.crossed {
        println!("Status:             CROSSED at patient {}", result.crossed_at.unwrap());
    } else if let Some(fut) = futility_threshold {
        if result.final_evalue < fut {
            println!("Status:             Below futility threshold ({:.2})", fut);
        } else {
            println!("Status:             Did not cross (ongoing)");
        }
    } else {
        println!("Status:             Did not cross (ongoing)");
    }

    // Effect sizes
    println!("\n--- Effect Sizes ---");

    if result.crossed {
        println!("\nAt Crossing (patient {}):", result.crossed_at.unwrap());
        println!("  {}:  {:.3}", effect_label, result.effect_at_cross.unwrap());
    }

    println!("\nFinal (patient {}):", result.n_total);
    println!("  {}:  {:.3}", effect_label, result.final_effect);
    println!("  Mean (Treatment):   {:.2}", result.final_mean_trt);
    println!("  Mean (Control):     {:.2}", result.final_mean_ctrl);
    println!("  Pooled SD:          {:.2}", result.final_sd);

    // Also show the other effect measure for reference
    match result.method {
        Method::LinearERT => {
            let cohens_d = result.final_effect / result.final_sd;
            println!("  Cohen's d:          {:.3}", cohens_d);
        }
        Method::MAD => {
            let mean_diff = result.final_effect * result.final_sd;
            println!("  Mean Difference:    {:.2}", mean_diff);
        }
    }

    // Type M error
    if let Some(type_m) = result.type_m {
        println!("\n--- Type M Error (Magnification) ---");
        println!("  |Effect at cross| / |Effect final|: {:.2}x", type_m);
    }

    // Sample sizes
    println!("\n--- Sample Sizes ---");
    println!("  Treatment:        {} patients", result.n_trt);
    println!("  Control:          {} patients", result.n_ctrl);

    // Futility analysis
    if let Some(ref fut_analysis) = result.futility_analysis {
        let design_effect = match result.method {
            Method::LinearERT => fut_analysis.design.design_effect_linear,
            Method::MAD => fut_analysis.design.design_effect_mad,
        };

        println!("\n--- Futility Analysis ---");
        println!("  Design {}:  {:.3}", effect_label, design_effect);

        if fut_analysis.ever_triggered {
            println!("  Episodes:         {} time(s) below threshold", fut_analysis.regions.len());
            for (i, (start, end)) in fut_analysis.regions.iter().enumerate() {
                println!("    Episode {}: patients {} - {}", i + 1, start, end);
            }

            if let Some(ref worst) = fut_analysis.worst_point {
                println!("  Worst point:");
                println!("    Patient:        {}", worst.patient_num);
                println!("    Wealth:         {:.4}", worst.wealth);
                println!("    Required {}:  {:.3}", effect_label, worst.required_effect);
                println!("    Ratio to design: {:.2}x", worst.ratio_to_design);
            }
        } else {
            println!("  Status:           Never dropped below futility threshold");
        }
    }
}

// ============================================================================
// HTML REPORT
// ============================================================================

fn build_html_report(
    result: &AnalysisResult,
    csv_path: &str,
    burn_in: usize,
    ramp: usize,
    success_threshold: f64,
    futility_threshold: Option<f64>,
    c_max: f64,
    min_val: f64,
    max_val: f64,
) -> String {
    let timestamp = chrono_lite();

    let method_name = match result.method {
        Method::LinearERT => "e-RTo",
        Method::MAD => "e-RTc",
    };

    let effect_label = match result.method {
        Method::LinearERT => "Mean Difference",
        Method::MAD => "Cohen's d",
    };

    let x_axis: Vec<usize> = (0..=result.n_total).collect();
    let x_json = format!("{:?}", x_axis);
    let y_json = format!("{:?}", result.trajectory);

    let status_text = if result.crossed {
        format!("<span style='color:green;font-weight:bold'>CROSSED at patient {}</span>",
                result.crossed_at.unwrap())
    } else if let Some(fut) = futility_threshold {
        if result.final_evalue < fut {
            format!("<span style='color:orange'>Below futility threshold ({:.2})</span>", fut)
        } else {
            "<span style='color:gray'>Did not cross (ongoing)</span>".to_string()
        }
    } else {
        "<span style='color:gray'>Did not cross (ongoing)</span>".to_string()
    };

    let crossing_section = if result.crossed {
        format!(r#"
        <h3>At Crossing (Patient {})</h3>
        <table>
            <tr><td>{}:</td><td><strong>{:.3}</strong></td></tr>
        </table>
        "#,
        result.crossed_at.unwrap(),
        effect_label,
        result.effect_at_cross.unwrap())
    } else {
        String::new()
    };

    let type_m_section = if let Some(type_m) = result.type_m {
        format!(r#"
        <h3>Type M Error (Magnification)</h3>
        <table>
            <tr><td>|Effect at cross| / |Effect final|:</td><td><strong>{:.2}x</strong></td></tr>
        </table>
        "#, type_m)
    } else {
        String::new()
    };

    // Futility lines for plots
    let futility_line = if let Some(fut) = futility_threshold {
        format!("{{type:'line',x0:0,x1:1,xref:'paper',y0:{},y1:{},line:{{color:'orange',width:1.5,dash:'dot'}}}}", fut, fut)
    } else {
        String::new()
    };

    let futility_line_support = if let Some(fut) = futility_threshold {
        format!("{{type:'line',x0:0,x1:1,xref:'paper',y0:{:.4},y1:{:.4},line:{{color:'orange',width:1.5,dash:'dot'}}}}", fut.ln(), fut.ln())
    } else {
        String::new()
    };

    // Futility shading regions
    let mut futility_shapes = String::new();
    let mut futility_shapes_support = String::new();

    if let Some(ref fut_analysis) = result.futility_analysis {
        for (start, end) in &fut_analysis.regions {
            futility_shapes.push_str(&format!(
                "{{type:'rect',x0:{},x1:{},y0:0,y1:1,yref:'paper',fillcolor:'rgba(255,165,0,0.15)',line:{{width:0}}}},",
                start, end
            ));
            futility_shapes_support.push_str(&format!(
                "{{type:'rect',x0:{},x1:{},y0:0,y1:1,yref:'paper',fillcolor:'rgba(255,165,0,0.15)',line:{{width:0}}}},",
                start, end
            ));
        }
    }

    // Futility analysis section
    let futility_section = if let Some(ref fut_analysis) = result.futility_analysis {
        let design_effect = match result.method {
            Method::LinearERT => fut_analysis.design.design_effect_linear,
            Method::MAD => fut_analysis.design.design_effect_mad,
        };

        let mut html = format!(r#"
        <h2>Futility Analysis</h2>
        <p><em>Note: Futility monitoring is decision support, not anytime-valid inference.</em></p>
        <table>
            <tr><td>Design Control Mean:</td><td>{:.2}</td></tr>
            <tr><td>Design Treatment Mean:</td><td>{:.2}</td></tr>
            <tr><td>Design SD:</td><td>{:.2}</td></tr>
            <tr><td>Design {}:</td><td><strong>{:.3}</strong></td></tr>
            <tr><td>Ever Below Threshold:</td><td>{}</td></tr>
        </table>
        "#,
        fut_analysis.design.control_mean,
        fut_analysis.design.treatment_mean,
        fut_analysis.design.sd,
        effect_label, design_effect,
        if fut_analysis.ever_triggered { "Yes" } else { "No" }
        );

        if fut_analysis.ever_triggered {
            html.push_str("<h3>Futility Episodes</h3><table>");
            for (i, (start, end)) in fut_analysis.regions.iter().enumerate() {
                html.push_str(&format!(
                    "<tr><td>Episode {}:</td><td>Patients {} - {}</td></tr>",
                    i + 1, start, end
                ));
            }
            html.push_str("</table>");

            if let Some(ref worst) = fut_analysis.worst_point {
                html.push_str(&format!(r#"
                <h3>Worst Point</h3>
                <table>
                    <tr><td>Patient:</td><td>{}</td></tr>
                    <tr><td>Wealth:</td><td>{:.4}</td></tr>
                    <tr><td>Required {} for recovery:</td><td><strong>{:.3}</strong></td></tr>
                    <tr><td>Ratio to design:</td><td><strong>{:.2}x</strong></td></tr>
                </table>
                "#,
                worst.patient_num, worst.wealth, effect_label, worst.required_effect, worst.ratio_to_design
                ));
            }
        }

        // Ratio plot
        if !fut_analysis.points.is_empty() {
            let ratio_x: Vec<usize> = fut_analysis.points.iter().map(|p| p.patient_num).collect();
            let ratio_y: Vec<f64> = fut_analysis.points.iter().map(|p| p.ratio_to_design).collect();
            let ratio_x_json = format!("{:?}", ratio_x);
            let ratio_y_json = format!("{:?}", ratio_y);

            html.push_str(&format!(r#"
            <h3>Required Effect Ratio Over Time</h3>
            <p>Shows ratio of required effect to design effect when below futility threshold. Higher = harder to recover.</p>
            <div id="plot3" style="width:100%;height:350px;"></div>
            <script>
                Plotly.newPlot('plot3', [
                    {{
                        type: 'scatter',
                        mode: 'lines+markers',
                        x: {},
                        y: {},
                        line: {{color: 'darkorange', width: 2}},
                        marker: {{size: 6}},
                        name: 'Required/Design Ratio'
                    }}
                ], {{
                    yaxis: {{title: 'Required / Design', rangemode: 'tozero'}},
                    xaxis: {{title: 'Patient Number'}},
                    shapes: [
                        {{type:'line',x0:0,x1:1,xref:'paper',y0:1,y1:1,line:{{color:'gray',width:1.5,dash:'dash'}}}}
                    ],
                    annotations: [
                        {{x:1,xref:'paper',y:1,text:'Design (1.0x)',showarrow:false,font:{{color:'gray'}}}}
                    ]
                }});
            </script>
            "#, ratio_x_json, ratio_y_json));
        }

        html
    } else {
        String::new()
    };

    // Method-specific parameters
    let method_params = match result.method {
        Method::LinearERT => format!(r#"
            <tr><td>Bounds (min, max):</td><td>[{:.1}, {:.1}]</td></tr>
        "#, min_val, max_val),
        Method::MAD => format!(r#"
            <tr><td>c_max:</td><td>{:.2}</td></tr>
        "#, c_max),
    };

    // Secondary effect measure
    let secondary_effect = match result.method {
        Method::LinearERT => {
            let cohens_d = result.final_effect / result.final_sd;
            format!("<tr><td>Cohen's d:</td><td>{:.3}</td></tr>", cohens_d)
        }
        Method::MAD => {
            let mean_diff = result.final_effect * result.final_sd;
            format!("<tr><td>Mean Difference:</td><td>{:.2}</td></tr>", mean_diff)
        }
    };

    format!(r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Continuous e-RT Analysis Report</title>
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
        em {{ color: #95a5a6; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Continuous e-RT Analysis Report</h1>
        <p class="timestamp">Generated: {}</p>

        <h2>Data Source</h2>
        <table>
            <tr><td>File:</td><td>{}</td></tr>
            <tr><td>Total Patients:</td><td>{}</td></tr>
            <tr><td>Treatment Arm:</td><td>{} (mean: {:.2})</td></tr>
            <tr><td>Control Arm:</td><td>{} (mean: {:.2})</td></tr>
            <tr><td>Pooled SD:</td><td>{:.2}</td></tr>
        </table>

        <h2>Parameters</h2>
        <table>
            <tr><td>Method:</td><td><strong>{}</strong></td></tr>
            {}
            <tr><td>Burn-in:</td><td>{}</td></tr>
            <tr><td>Ramp:</td><td>{}</td></tr>
            <tr><td>Success Threshold:</td><td>{}</td></tr>
            <tr><td>Futility Threshold:</td><td>{}</td></tr>
        </table>

        <h2>Results</h2>
        <table>
            <tr class="highlight"><td>Final e-value:</td><td>{:.4}</td></tr>
            <tr class="highlight"><td>Status:</td><td>{}</td></tr>
        </table>

        {}

        <h3>Final Effect Estimates (Patient {})</h3>
        <table>
            <tr><td>{}:</td><td><strong>{:.3}</strong></td></tr>
            {}
            <tr><td>Mean (Treatment):</td><td>{:.2}</td></tr>
            <tr><td>Mean (Control):</td><td>{:.2}</td></tr>
        </table>

        {}

        {}

        <h2>e-Value Trajectory</h2>
        <div id="plot1" style="width:100%;height:400px;"></div>

        <h2>Support (ln e-value)</h2>
        <div id="plot2" style="width:100%;height:400px;"></div>
    </div>

    <script>
        // e-value trajectory (log scale)
        Plotly.newPlot('plot1', [
            {{
                type: 'scatter',
                mode: 'lines',
                x: {},
                y: {},
                line: {{color: 'blue', width: 2}},
                name: 'e-value'
            }}
        ], {{
            yaxis: {{type: 'log', title: 'e-value', range: [-0.5, 2]}},
            xaxis: {{title: 'Patients Enrolled'}},
            shapes: [
                {}
                {{type:'line',x0:0,x1:1,xref:'paper',y0:{},y1:{},line:{{color:'green',width:2,dash:'dash'}}}},
                {}
            ],
            annotations: [
                {{x:1,xref:'paper',y:{},text:'Success (e={:.0})',showarrow:false,font:{{color:'green'}}}}
            ]
        }});

        // Support trajectory (linear scale)
        var support = {}.map(function(e) {{ return Math.log(e); }});
        Plotly.newPlot('plot2', [
            {{
                type: 'scatter',
                mode: 'lines',
                x: {},
                y: support,
                line: {{color: 'blue', width: 2}},
                name: 'Support'
            }}
        ], {{
            yaxis: {{title: 'Support (ln e-value)'}},
            xaxis: {{title: 'Patients Enrolled'}},
            shapes: [
                {}
                {{type:'line',x0:0,x1:1,xref:'paper',y0:{:.4},y1:{:.4},line:{{color:'green',width:2,dash:'dash'}}}},
                {}
            ],
            annotations: [
                {{x:1,xref:'paper',y:{:.4},text:'Success (ln {:.0} = {:.2})',showarrow:false,font:{{color:'green'}}}}
            ]
        }});
    </script>
</body>
</html>"#,
        // Header
        timestamp,
        // Data source
        csv_path, result.n_total,
        result.n_trt, result.final_mean_trt,
        result.n_ctrl, result.final_mean_ctrl,
        result.final_sd,
        // Parameters
        method_name,
        method_params,
        burn_in, ramp, success_threshold,
        futility_threshold.map_or("None".to_string(), |f| format!("{:.2}", f)),
        // Results
        result.final_evalue,
        status_text,
        // Crossing section
        crossing_section,
        // Final estimates
        result.n_total,
        effect_label, result.final_effect,
        secondary_effect,
        result.final_mean_trt,
        result.final_mean_ctrl,
        // Type M section
        type_m_section,
        // Futility section
        futility_section,
        // Plot 1: e-value
        x_json, y_json,
        futility_shapes,
        success_threshold, success_threshold,
        futility_line,
        success_threshold, success_threshold,
        // Plot 2: Support
        y_json,
        x_json,
        futility_shapes_support,
        success_threshold.ln(), success_threshold.ln(),
        futility_line_support,
        success_threshold.ln(), success_threshold, success_threshold.ln()
    )
}
