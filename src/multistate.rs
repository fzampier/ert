// multistate.rs - Flexible ordinal multi-state e-process (e-RTms)
//
// User configures:
// - N states ordered worst→best
// - Which states are absorbing
// - Transition matrices for control and treatment

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::RngCore;
use std::fs::File;
use std::io::{self, Write};

use crate::ert_core::{get_input, get_input_usize, get_string, get_optional_input, chrono_lite, normal_cdf, normal_quantile};
use crate::agnostic::{AgnosticERT, Signal, Arm};

// === CONFIGURATION ===

#[derive(Clone)]
pub struct MultiStateConfig {
    pub state_names: Vec<String>,  // Ordered worst→best
    pub absorbing: Vec<usize>,     // Indices of absorbing states
    pub start_state: usize,
    pub max_days: usize,
}

impl MultiStateConfig {
    pub fn n_states(&self) -> usize {
        self.state_names.len()
    }

    pub fn is_absorbing(&self, state: usize) -> bool {
        self.absorbing.contains(&state)
    }

    // ICU preset: Dead(0), ICU(1), Ward(2), Home(3)
    pub fn icu_preset() -> Self {
        MultiStateConfig {
            state_names: vec!["Dead".into(), "ICU".into(), "Ward".into(), "Home".into()],
            absorbing: vec![0, 3],  // Dead and Home
            start_state: 1,         // ICU
            max_days: 28,
        }
    }
}

// === TRANSITION MATRIX ===

#[derive(Clone)]
pub struct TransitionMatrix {
    pub n_states: usize,
    pub probs: Vec<Vec<f64>>,
}

impl TransitionMatrix {
    pub fn new(n_states: usize) -> Self {
        // Initialize with identity (stay in same state)
        let probs = (0..n_states).map(|i| {
            let mut row = vec![0.0; n_states];
            row[i] = 1.0;
            row
        }).collect();
        TransitionMatrix { n_states, probs }
    }

    pub fn set_row(&mut self, from: usize, probs: Vec<f64>) {
        if from < self.n_states && probs.len() == self.n_states {
            self.probs[from] = probs;
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        for (i, row) in self.probs.iter().enumerate() {
            let sum: f64 = row.iter().sum();
            if (sum - 1.0).abs() > 0.01 {
                return Err(format!("Row {} sums to {:.3}, expected 1.0", i, sum));
            }
        }
        Ok(())
    }

    pub fn sample_next<R: Rng + ?Sized>(&self, rng: &mut R, current: usize) -> usize {
        let r: f64 = rng.gen();
        let mut cumsum = 0.0;
        for (next_state, &prob) in self.probs[current].iter().enumerate() {
            cumsum += prob;
            if r < cumsum {
                return next_state;
            }
        }
        current
    }

    // ICU preset: Control (no treatment effect)
    pub fn icu_control() -> Self {
        let mut m = TransitionMatrix::new(4);
        // Dead (absorbing)
        m.set_row(0, vec![1.0, 0.0, 0.0, 0.0]);
        // ICU -> ICU(91.5%), Ward(7%), Dead(1.5%)
        m.set_row(1, vec![0.015, 0.915, 0.070, 0.000]);
        // Ward -> Dead(2%), ICU(7%), Ward(88%), Home(3%)
        m.set_row(2, vec![0.020, 0.070, 0.880, 0.030]);
        // Home (absorbing)
        m.set_row(3, vec![0.0, 0.0, 0.0, 1.0]);
        m
    }

    // ICU preset: Large treatment effect (OR~1.6, +15% Home)
    pub fn icu_treatment_large() -> Self {
        let mut m = TransitionMatrix::new(4);
        m.set_row(0, vec![1.0, 0.0, 0.0, 0.0]);
        // ICU -> better step-down (9% vs 7%)
        m.set_row(1, vec![0.010, 0.900, 0.090, 0.000]);
        // Ward -> better discharge (5% vs 3%), less death
        m.set_row(2, vec![0.030, 0.050, 0.870, 0.050]);
        m.set_row(3, vec![0.0, 0.0, 0.0, 1.0]);
        m
    }

    // ICU preset: Small treatment effect (OR~1.2, +5% Home)
    pub fn icu_treatment_small() -> Self {
        let mut m = TransitionMatrix::new(4);
        m.set_row(0, vec![1.0, 0.0, 0.0, 0.0]);
        m.set_row(1, vec![0.013, 0.907, 0.080, 0.000]);
        m.set_row(2, vec![0.025, 0.060, 0.875, 0.040]);
        m.set_row(3, vec![0.0, 0.0, 0.0, 1.0]);
        m
    }
}

// === TRANSITION CLASSIFICATION ===

// Good = moving to higher-indexed (better) state
fn is_good_transition(from: usize, to: usize) -> bool {
    to > from
}

// === DATA STRUCTURES ===

struct Transition {
    from: usize,
    to: usize,
    arm: u8,
}

struct Trial {
    stop_n: Option<usize>,
    agnostic_stop_n: Option<usize>,
    stratified_stop_n: Option<usize>,  // Stratified average e-process
    effect_at_stop: f64,
    effect_final: f64,
    min_wealth: f64,
    // Transition diagnostics
    n_transitions: usize,
    n_good_trt: usize,
    n_bad_trt: usize,
    n_good_ctrl: usize,
    n_bad_ctrl: usize,
    // For Markov LRT
    trt_counts: Vec<Vec<usize>>,
    ctrl_counts: Vec<Vec<usize>>,
}

// Per-stratum state for stratified e-process
#[derive(Clone)]
struct Stratum {
    n_good_trt: f64,
    n_total_trt: f64,
    n_good_ctrl: f64,
    n_total_ctrl: f64,
    wealth: f64,
    n_obs: usize,
}

// === PROPORTIONAL ODDS BENCHMARK ===

fn calculate_proportional_or(ctrl: &[f64], trt: &[f64]) -> f64 {
    let n = ctrl.len();
    if n < 2 { return 1.0; }

    // States are ordered worst→best, so cumulative from worst
    let mut cum_ctrl = vec![0.0; n];
    let mut cum_trt = vec![0.0; n];
    cum_ctrl[0] = ctrl[0];
    cum_trt[0] = trt[0];
    for i in 1..n {
        cum_ctrl[i] = cum_ctrl[i-1] + ctrl[i];
        cum_trt[i] = cum_trt[i-1] + trt[i];
    }

    let mut log_ors = Vec::new();
    for i in 0..(n-1) {
        let p_ctrl = cum_ctrl[i].clamp(0.001, 0.999);
        let p_trt = cum_trt[i].clamp(0.001, 0.999);
        let or_j = (p_trt / (1.0 - p_trt)) / (p_ctrl / (1.0 - p_ctrl));
        if or_j > 0.0 {
            log_ors.push(or_j.ln());
        }
    }

    if log_ors.is_empty() { return 1.0; }
    let mean_log_or = log_ors.iter().sum::<f64>() / log_ors.len() as f64;
    (-mean_log_or).exp()  // Flip for "treatment better" interpretation
}

fn calculate_mann_whitney_prob(ctrl: &[f64], trt: &[f64]) -> f64 {
    let n = ctrl.len();
    let mut p_win = 0.0;
    let mut p_tie = 0.0;
    for i in 0..n {
        for j in 0..n {
            let p_joint = trt[i] * ctrl[j];
            if i > j { p_win += p_joint; }  // Higher index = better
            else if i == j { p_tie += p_joint; }
        }
    }
    p_win + 0.5 * p_tie
}

fn proportional_odds_power(or: f64, n_total: usize, n_states: usize, alpha: f64) -> f64 {
    if or <= 1.0 { return alpha; }
    let log_or = or.ln();
    let n = n_total as f64;
    let k = n_states as f64;
    let var_log_or = 4.0 * (k + 1.0) / (3.0 * n);
    let se_log_or = var_log_or.sqrt();
    let z = log_or / se_log_or;
    let z_alpha = normal_quantile(1.0 - alpha / 2.0);
    normal_cdf(z - z_alpha)
}

fn proportional_odds_sample_size(or: f64, n_states: usize, power: f64, alpha: f64) -> usize {
    if or <= 1.0 { return 99999; }
    let log_or = or.ln();
    let z_alpha = normal_quantile(1.0 - alpha / 2.0);
    let z_beta = normal_quantile(power);
    let k = n_states as f64;
    let n = 4.0 * (k + 1.0) / (3.0 * log_or * log_or) * (z_alpha + z_beta).powi(2);
    (n.ceil() as usize).max(10)
}

// === MARKOV MODEL BENCHMARK ===

/// Count transitions by arm into n_states x n_states matrices
fn count_transitions(transitions: &[Transition], n_states: usize) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let mut trt = vec![vec![0usize; n_states]; n_states];
    let mut ctrl = vec![vec![0usize; n_states]; n_states];

    for t in transitions {
        if t.arm == 1 {
            trt[t.from][t.to] += 1;
        } else {
            ctrl[t.from][t.to] += 1;
        }
    }
    (trt, ctrl)
}

/// Log-likelihood of observed transitions given transition probability matrix
fn markov_log_likelihood(counts: &[Vec<usize>], probs: &[Vec<f64>]) -> f64 {
    let mut ll = 0.0;
    for from in 0..counts.len() {
        for to in 0..counts[from].len() {
            let n = counts[from][to];
            if n > 0 && probs[from][to] > 0.0 {
                ll += n as f64 * probs[from][to].ln();
            }
        }
    }
    ll
}

/// Estimate transition probabilities from counts (MLE)
fn estimate_transition_probs(counts: &[Vec<usize>]) -> Vec<Vec<f64>> {
    let n_states = counts.len();
    let mut probs = vec![vec![0.0; n_states]; n_states];

    for from in 0..n_states {
        let row_sum: usize = counts[from].iter().sum();
        if row_sum > 0 {
            for to in 0..n_states {
                probs[from][to] = counts[from][to] as f64 / row_sum as f64;
            }
        }
    }
    probs
}

/// Likelihood ratio test for Markov transition differences
/// Returns (chi2_statistic, p_value, df)
fn markov_lrt(trt_counts: &[Vec<usize>], ctrl_counts: &[Vec<usize>], absorbing: &[usize]) -> (f64, f64, usize) {
    let n_states = trt_counts.len();

    // Pool counts for null hypothesis
    let mut pooled = vec![vec![0usize; n_states]; n_states];
    for from in 0..n_states {
        for to in 0..n_states {
            pooled[from][to] = trt_counts[from][to] + ctrl_counts[from][to];
        }
    }

    // Estimate probabilities
    let p_pooled = estimate_transition_probs(&pooled);
    let p_trt = estimate_transition_probs(trt_counts);
    let p_ctrl = estimate_transition_probs(ctrl_counts);

    // Log-likelihoods
    let ll_null = markov_log_likelihood(trt_counts, &p_pooled)
                + markov_log_likelihood(ctrl_counts, &p_pooled);
    let ll_alt = markov_log_likelihood(trt_counts, &p_trt)
               + markov_log_likelihood(ctrl_counts, &p_ctrl);

    // LRT statistic: -2 * (ll_null - ll_alt)
    let chi2 = -2.0 * (ll_null - ll_alt);
    let chi2 = chi2.max(0.0); // Numerical safety

    // Degrees of freedom: (non-absorbing rows) * (n_states - 1) for the difference
    let n_non_absorbing = (0..n_states).filter(|s| !absorbing.contains(s)).count();
    let df = n_non_absorbing * (n_states - 1);

    // P-value from chi-squared distribution
    let p_value = if df > 0 { 1.0 - chi2_cdf(chi2, df) } else { 1.0 };

    (chi2, p_value, df)
}

/// Chi-squared CDF approximation (Wilson-Hilferty)
fn chi2_cdf(x: f64, df: usize) -> f64 {
    if df == 0 { return 1.0; }
    let k = df as f64;
    // Wilson-Hilferty transformation to normal
    let z = ((x / k).powf(1.0/3.0) - (1.0 - 2.0/(9.0*k))) / (2.0/(9.0*k)).sqrt();
    normal_cdf(z)
}

// === SIMULATION ===

fn simulate_patient<R: Rng + ?Sized>(
    rng: &mut R,
    p: &TransitionMatrix,
    config: &MultiStateConfig,
    arm: u8,
) -> Vec<Transition> {
    let mut state = config.start_state;
    let mut transitions = Vec::new();

    for _ in 1..=config.max_days {
        if config.is_absorbing(state) { break; }
        let new_state = p.sample_next(rng, state);
        if new_state != state {
            transitions.push(Transition { from: state, to: new_state, arm });
        }
        state = new_state;
    }
    transitions
}

fn run_single_trial<R: Rng + ?Sized>(
    rng: &mut R,
    n_patients: usize,
    p_ctrl: &TransitionMatrix,
    p_trt: &TransitionMatrix,
    config: &MultiStateConfig,
    burn_in: usize,
    ramp: usize,
    threshold: f64,
) -> (Trial, Vec<f64>) {
    let mut all_transitions: Vec<Transition> = Vec::new();

    for _ in 0..n_patients {
        let arm: u8 = if rng.gen_bool(0.5) { 1 } else { 0 };
        let p = if arm == 1 { p_trt } else { p_ctrl };
        all_transitions.extend(simulate_patient(rng, p, config, arm));
    }

    let n_states = config.n_states();
    let (trt_counts, ctrl_counts) = count_transitions(&all_transitions, n_states);

    if all_transitions.len() < burn_in {
        return (Trial {
            stop_n: None,
            agnostic_stop_n: None,
            stratified_stop_n: None,
            effect_at_stop: 0.0,
            effect_final: 0.0,
            min_wealth: 1.0,
            n_transitions: all_transitions.len(),
            n_good_trt: 0, n_bad_trt: 0, n_good_ctrl: 0, n_bad_ctrl: 0,
            trt_counts, ctrl_counts,
        }, vec![1.0]);
    }

    // Compute wealth
    let n = all_transitions.len();
    let mut wealth = vec![1.0; n];
    let mut effects = vec![0.0; n];
    let (mut n_good_trt, mut n_total_trt, mut n_good_ctrl, mut n_total_ctrl) = (0.0, 0.0, 0.0, 0.0);
    let (mut cnt_good_trt, mut cnt_bad_trt, mut cnt_good_ctrl, mut cnt_bad_ctrl) = (0usize, 0usize, 0usize, 0usize);

    for (i, trans) in all_transitions.iter().enumerate() {
        let is_good = is_good_transition(trans.from, trans.to);
        let is_trt = trans.arm == 1;

        // Count for diagnostics
        if is_trt {
            if is_good { cnt_good_trt += 1; } else { cnt_bad_trt += 1; }
        } else {
            if is_good { cnt_good_ctrl += 1; } else { cnt_bad_ctrl += 1; }
        }

        let lambda = if i > burn_in && n_total_trt > 0.0 && n_total_ctrl > 0.0 {
            let c_i = (((i - burn_in) as f64) / ramp as f64).clamp(0.0, 1.0);
            let rate_trt = n_good_trt / n_total_trt;
            let rate_ctrl = n_good_ctrl / n_total_ctrl;
            let delta = rate_trt - rate_ctrl;
            if is_good { 0.5 + 0.5 * c_i * delta } else { 0.5 - 0.5 * c_i * delta }
        } else { 0.5 };

        let lambda = lambda.clamp(0.001, 0.999);
        let mult = if is_trt { lambda / 0.5 } else { (1.0 - lambda) / 0.5 };
        wealth[i] = if i > 0 { wealth[i - 1] * mult } else { mult };

        if is_trt { n_total_trt += 1.0; if is_good { n_good_trt += 1.0; } }
        else { n_total_ctrl += 1.0; if is_good { n_good_ctrl += 1.0; } }

        let rate_trt = if n_total_trt > 0.0 { n_good_trt / n_total_trt } else { 0.0 };
        let rate_ctrl = if n_total_ctrl > 0.0 { n_good_ctrl / n_total_ctrl } else { 0.0 };
        effects[i] = rate_trt - rate_ctrl;
    }

    let stop_n = wealth.iter().position(|&w| w >= threshold);
    let effect_at_stop = stop_n.map(|i| effects[i]).unwrap_or(0.0);
    let effect_final = *effects.last().unwrap_or(&0.0);
    let min_wealth = wealth.iter().cloned().fold(f64::INFINITY, f64::min);

    // Agnostic e-RT in parallel
    let mut agnostic = AgnosticERT::new(burn_in, ramp, threshold);
    let mut agnostic_stop_n = None;
    for (i, trans) in all_transitions.iter().enumerate() {
        let signal = Signal {
            arm: if trans.arm == 1 { Arm::Treatment } else { Arm::Control },
            good: is_good_transition(trans.from, trans.to),
        };
        if agnostic.observe(signal) {
            agnostic_stop_n = Some(i + 1);
            break;
        }
    }

    // Stratified e-process: average of per-stratum e-values
    let mut strata: Vec<Stratum> = (0..n_states).map(|_| Stratum {
        n_good_trt: 0.0, n_total_trt: 0.0,
        n_good_ctrl: 0.0, n_total_ctrl: 0.0,
        wealth: 1.0, n_obs: 0,
    }).collect();

    let mut stratified_stop_n = None;
    for (i, trans) in all_transitions.iter().enumerate() {
        let is_good = is_good_transition(trans.from, trans.to);
        let is_trt = trans.arm == 1;
        let from = trans.from;

        let s = &mut strata[from];
        s.n_obs += 1;

        // Compute stratum-specific lambda (use global i for burn-in, stratum counts for rates)
        let s_lambda = if i > burn_in && s.n_total_trt > 0.0 && s.n_total_ctrl > 0.0 {
            let c_i = (((i - burn_in) as f64) / ramp as f64).clamp(0.0, 1.0);
            let rate_trt = s.n_good_trt / s.n_total_trt;
            let rate_ctrl = s.n_good_ctrl / s.n_total_ctrl;
            let delta = rate_trt - rate_ctrl;
            if is_good { 0.5 + 0.5 * c_i * delta } else { 0.5 - 0.5 * c_i * delta }
        } else { 0.5 };

        let s_lambda = s_lambda.clamp(0.001, 0.999);
        let s_mult = if is_trt { s_lambda / 0.5 } else { (1.0 - s_lambda) / 0.5 };
        s.wealth *= s_mult;

        // Update stratum counts
        if is_trt { s.n_total_trt += 1.0; if is_good { s.n_good_trt += 1.0; } }
        else { s.n_total_ctrl += 1.0; if is_good { s.n_good_ctrl += 1.0; } }

        // Average of active strata wealths
        let active: Vec<f64> = strata.iter().filter(|s| s.n_obs > 0).map(|s| s.wealth).collect();
        let avg_wealth = if !active.is_empty() { active.iter().sum::<f64>() / active.len() as f64 } else { 1.0 };

        if stratified_stop_n.is_none() && avg_wealth >= threshold {
            stratified_stop_n = Some(i + 1);
        }
    }

    (Trial {
        stop_n, agnostic_stop_n, stratified_stop_n, effect_at_stop, effect_final, min_wealth,
        n_transitions: n,
        n_good_trt: cnt_good_trt, n_bad_trt: cnt_bad_trt,
        n_good_ctrl: cnt_good_ctrl, n_bad_ctrl: cnt_bad_ctrl,
        trt_counts, ctrl_counts,
    }, wealth)
}

fn compute_day_n<R: Rng + ?Sized>(
    rng: &mut R,
    p: &TransitionMatrix,
    config: &MultiStateConfig,
    n_patients: usize,
) -> Vec<f64> {
    let mut counts = vec![0usize; config.n_states()];

    for _ in 0..n_patients {
        let mut state = config.start_state;
        for _ in 1..=config.max_days {
            if config.is_absorbing(state) { break; }
            state = p.sample_next(rng, state);
        }
        counts[state] += 1;
    }

    let n = n_patients as f64;
    counts.iter().map(|&c| c as f64 / n).collect()
}

// === FUTILITY GRID ===

fn compute_futility_grid(trials: &[Trial], thresholds: &[f64]) -> Vec<(f64, f64, f64)> {
    thresholds.iter().map(|&thresh| {
        let triggered: Vec<_> = trials.iter().filter(|t| t.min_wealth < thresh).collect();
        let n_triggered = triggered.len();
        let n_recovered = triggered.iter().filter(|t| t.stop_n.is_some()).count();
        let pct_triggered = 100.0 * n_triggered as f64 / trials.len() as f64;
        let pct_recovered = if n_triggered > 0 { 100.0 * n_recovered as f64 / n_triggered as f64 } else { 0.0 };
        (thresh, pct_triggered, pct_recovered)
    }).collect()
}

// === INPUT HELPERS ===

fn get_transition_matrix(n_states: usize, name: &str, state_names: &[String], absorbing: &[usize]) -> TransitionMatrix {
    println!("\n--- {} Transition Matrix ---", name);
    println!("Enter probabilities for each row (must sum to 1.0)");
    if !absorbing.is_empty() {
        println!("(Absorbing states auto-filled: stay with p=1.0)");
    }

    let mut m = TransitionMatrix::new(n_states);

    for from in 0..n_states {
        // Auto-fill absorbing states
        if absorbing.contains(&from) {
            let mut probs = vec![0.0; n_states];
            probs[from] = 1.0;
            m.set_row(from, probs);
            println!("\n{} (state {}) is absorbing → stays with p=1.0", state_names[from], from);
            continue;
        }

        println!("\nFrom {} (state {}):", state_names[from], from);
        let mut probs = Vec::new();
        for to in 0..n_states {
            let prompt = format!("  → {} ({}): ", state_names[to], to);
            let p: f64 = get_input(&prompt);
            probs.push(p);
        }
        m.set_row(from, probs);
    }

    if let Err(e) = m.validate() {
        println!("Warning: {}", e);
    }

    m
}

// === MAIN ===

pub fn run() {
    println!("\n==========================================");
    println!("   e-RTms MULTI-STATE SIMULATION");
    println!("==========================================\n");

    // Configuration choice
    println!("Configuration:");
    println!("  1. ICU preset (Dead/ICU/Ward/Home)");
    println!("  2. Custom states");
    let config_choice = get_input_usize("Select (1 or 2): ");

    let (config, p_ctrl, p_trt) = if config_choice == 2 {
        // Custom configuration
        let n_states = get_input_usize("\nNumber of states: ");

        println!("\nEnter state names (ordered WORST to BEST):");
        let mut state_names = Vec::new();
        for i in 0..n_states {
            let name = get_string(&format!("  State {} name: ", i));
            state_names.push(name);
        }

        println!("\nWhich states are absorbing? (comma-separated indices, e.g., 0,{}):", n_states - 1);
        let absorbing_str = get_string("Absorbing states: ");
        let absorbing: Vec<usize> = absorbing_str
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();

        let start_state = get_input_usize(&format!("Starting state (0-{}): ", n_states - 1));
        let max_days = get_input_usize("Follow-up days: ");

        let config = MultiStateConfig {
            state_names: state_names.clone(),
            absorbing: absorbing.clone(),
            start_state,
            max_days,
        };

        let p_ctrl = get_transition_matrix(n_states, "Control", &state_names, &absorbing);
        let p_trt = get_transition_matrix(n_states, "Treatment", &state_names, &absorbing);

        (config, p_ctrl, p_trt)
    } else {
        // ICU preset
        println!("\nUsing ICU preset: Dead(0) < ICU(1) < Ward(2) < Home(3)");
        println!("Absorbing: Dead, Home | Start: ICU | Days: 28\n");

        println!("Effect size:");
        println!("  1. Large (OR~1.6, Home +15%)");
        println!("  2. Small (OR~1.2, Home +5%)");
        let effect_choice = get_input_usize("Select (1 or 2): ");

        let config = MultiStateConfig::icu_preset();
        let p_ctrl = TransitionMatrix::icu_control();
        let p_trt = if effect_choice == 2 {
            println!("Using small effect");
            TransitionMatrix::icu_treatment_small()
        } else {
            println!("Using large effect");
            TransitionMatrix::icu_treatment_large()
        };

        (config, p_ctrl, p_trt)
    };

    // Simulation parameters
    let n_patients = get_input_usize("\nPatients per trial (e.g., 1000): ");
    let n_sims = get_input_usize("Simulations (e.g., 1000): ");
    let threshold: f64 = get_input("Threshold (default 20): ");
    let seed = get_optional_input("Seed (Enter for random): ");

    let burn_in = 30;
    let ramp = 50;

    let mut rng: Box<dyn RngCore> = match seed {
        Some(s) => Box::new(StdRng::seed_from_u64(s)),
        None => Box::new(rand::thread_rng()),
    };

    // Day N distributions
    println!("\nComputing Day {} distributions...", config.max_days);
    let d_ctrl = compute_day_n(&mut *rng, &p_ctrl, &config, 5000);
    let d_trt = compute_day_n(&mut *rng, &p_trt, &config, 5000);

    // Proportional odds benchmark
    let prop_or = calculate_proportional_or(&d_ctrl, &d_trt);
    let mann_whitney = calculate_mann_whitney_prob(&d_ctrl, &d_trt);
    let po_power = proportional_odds_power(prop_or, n_patients, config.n_states(), 0.05);
    let n_80 = proportional_odds_sample_size(prop_or, config.n_states(), 0.80, 0.05);
    let n_90 = proportional_odds_sample_size(prop_or, config.n_states(), 0.90, 0.05);

    // Run simulations
    println!("\n--- Null ---");
    let mut null_trials = Vec::new();
    let mut null_trajectories = Vec::new();
    for sim in 0..n_sims {
        let (trial, wealth) = run_single_trial(&mut *rng, n_patients, &p_ctrl, &p_ctrl, &config, burn_in, ramp, threshold);
        if null_trajectories.len() < 30 { null_trajectories.push(wealth); }
        null_trials.push(trial);
        if (sim + 1) % 100 == 0 { print!("\rSimulation {}/{}", sim + 1, n_sims); io::stdout().flush().unwrap(); }
    }
    println!();

    let null_ert_reject = null_trials.iter().filter(|t| t.stop_n.is_some()).count();
    let null_agn_reject = null_trials.iter().filter(|t| t.agnostic_stop_n.is_some()).count();
    let type1_ert = 100.0 * null_ert_reject as f64 / n_sims as f64;
    let type1_agn = 100.0 * null_agn_reject as f64 / n_sims as f64;

    println!("\n--- Alternative ---");
    let mut alt_trials = Vec::new();
    let mut alt_trajectories = Vec::new();
    for sim in 0..n_sims {
        let (trial, wealth) = run_single_trial(&mut *rng, n_patients, &p_ctrl, &p_trt, &config, burn_in, ramp, threshold);
        if alt_trajectories.len() < 30 { alt_trajectories.push(wealth); }
        alt_trials.push(trial);
        if (sim + 1) % 100 == 0 { print!("\rSimulation {}/{}", sim + 1, n_sims); io::stdout().flush().unwrap(); }
    }
    println!();

    let alt_ert_success = alt_trials.iter().filter(|t| t.stop_n.is_some()).count();
    let alt_agn_success = alt_trials.iter().filter(|t| t.agnostic_stop_n.is_some()).count();
    let alt_strat_success = alt_trials.iter().filter(|t| t.stratified_stop_n.is_some()).count();
    let power_ert = 100.0 * alt_ert_success as f64 / n_sims as f64;
    let power_agn = 100.0 * alt_agn_success as f64 / n_sims as f64;
    let power_strat = 100.0 * alt_strat_success as f64 / n_sims as f64;

    // Type I for stratified
    let null_strat_reject = null_trials.iter().filter(|t| t.stratified_stop_n.is_some()).count();
    let type1_strat = 100.0 * null_strat_reject as f64 / n_sims as f64;

    // Markov LRT power (fairer benchmark - tests transition rate differences)
    let null_markov_reject = null_trials.iter()
        .filter(|t| {
            let (_, p, _) = markov_lrt(&t.trt_counts, &t.ctrl_counts, &config.absorbing);
            p < 0.05
        }).count();
    let alt_markov_success = alt_trials.iter()
        .filter(|t| {
            let (_, p, _) = markov_lrt(&t.trt_counts, &t.ctrl_counts, &config.absorbing);
            p < 0.05
        }).count();
    let type1_markov = 100.0 * null_markov_reject as f64 / n_sims as f64;
    let power_markov = 100.0 * alt_markov_success as f64 / n_sims as f64;

    // Type M
    let effects_at_stop: Vec<f64> = alt_trials.iter().filter(|t| t.stop_n.is_some()).map(|t| t.effect_at_stop).collect();
    let effects_final: Vec<f64> = alt_trials.iter().filter(|t| t.stop_n.is_some()).map(|t| t.effect_final).collect();
    let avg_effect_stop = if !effects_at_stop.is_empty() { effects_at_stop.iter().sum::<f64>() / effects_at_stop.len() as f64 } else { 0.0 };
    let avg_effect_final = if !effects_final.is_empty() { effects_final.iter().sum::<f64>() / effects_final.len() as f64 } else { 0.0 };
    let type_m = if avg_effect_final.abs() > 0.001 { avg_effect_stop / avg_effect_final } else { 1.0 };

    // Futility grid
    let futility_grid = compute_futility_grid(&alt_trials, &[0.1, 0.2, 0.3, 0.4, 0.5]);

    // Console output
    let mut console = String::new();
    console.push_str(&format!("{}\n\n", chrono_lite()));

    // State info
    console.push_str("States (worst→best): ");
    console.push_str(&config.state_names.join(" < "));
    console.push_str("\n");
    console.push_str(&format!("Absorbing: {:?} | Start: {} | Days: {}\n\n",
        config.absorbing.iter().map(|&i| &config.state_names[i]).collect::<Vec<_>>(),
        config.state_names[config.start_state],
        config.max_days));

    // Day N distributions
    console.push_str(&format!("--- Day {} Distribution ---\n", config.max_days));
    console.push_str("Control:   ");
    for (i, p) in d_ctrl.iter().enumerate() {
        console.push_str(&format!("{}={:.1}% ", config.state_names[i], p * 100.0));
    }
    console.push_str("\nTreatment: ");
    for (i, p) in d_trt.iter().enumerate() {
        console.push_str(&format!("{}={:.1}% ", config.state_names[i], p * 100.0));
    }
    console.push_str("\n\n");

    // Transition diagnostics (average across alternative simulations)
    let avg_n_trans = alt_trials.iter().map(|t| t.n_transitions).sum::<usize>() as f64 / n_sims as f64;
    let avg_good_trt = alt_trials.iter().map(|t| t.n_good_trt).sum::<usize>() as f64 / n_sims as f64;
    let avg_bad_trt = alt_trials.iter().map(|t| t.n_bad_trt).sum::<usize>() as f64 / n_sims as f64;
    let avg_good_ctrl = alt_trials.iter().map(|t| t.n_good_ctrl).sum::<usize>() as f64 / n_sims as f64;
    let avg_bad_ctrl = alt_trials.iter().map(|t| t.n_bad_ctrl).sum::<usize>() as f64 / n_sims as f64;
    let trt_good_rate = if avg_good_trt + avg_bad_trt > 0.0 { avg_good_trt / (avg_good_trt + avg_bad_trt) } else { 0.0 };
    let ctrl_good_rate = if avg_good_ctrl + avg_bad_ctrl > 0.0 { avg_good_ctrl / (avg_good_ctrl + avg_bad_ctrl) } else { 0.0 };

    console.push_str("--- Transition Diagnostics ---\n");
    console.push_str(&format!("Avg transitions/trial: {:.0}\n", avg_n_trans));
    console.push_str(&format!("Treatment: {:.0} good, {:.0} bad ({:.1}% good)\n", avg_good_trt, avg_bad_trt, trt_good_rate * 100.0));
    console.push_str(&format!("Control:   {:.0} good, {:.0} bad ({:.1}% good)\n", avg_good_ctrl, avg_bad_ctrl, ctrl_good_rate * 100.0));
    console.push_str(&format!("Rate diff: {:.1}%\n\n", (trt_good_rate - ctrl_good_rate) * 100.0));

    console.push_str("--- Proportional Odds Benchmark ---\n");
    console.push_str(&format!("Proportional OR:     {:.2}\n", prop_or));
    console.push_str(&format!("Mann-Whitney P(T>C): {:.1}%\n", mann_whitney * 100.0));
    console.push_str(&format!("PO Power at N={}:  {:.1}%\n", n_patients, po_power * 100.0));
    console.push_str(&format!("PO N for 80%/90%:    {}/{}\n\n", n_80, n_90));

    console.push_str("--- Markov LRT Benchmark ---\n");
    console.push_str("(Tests transition rate differences - fairer comparison for e-RTms)\n");
    console.push_str(&format!("Markov LRT Power: {:.1}%  (Type I: {:.2}%)\n\n", power_markov, type1_markov));

    console.push_str(&format!("--- Power at N={} ---\n", n_patients));
    console.push_str(&format!("Prop Odds:    {:.1}%\n", po_power * 100.0));
    console.push_str(&format!("Markov LRT:   {:.1}%  (Type I: {:.2}%)\n", power_markov, type1_markov));
    console.push_str(&format!("e-RTms:       {:.1}%  (Type I: {:.2}%)\n", power_ert, type1_ert));
    console.push_str(&format!("e-RTms-strat: {:.1}%  (Type I: {:.2}%)  [averaged across strata]\n", power_strat, type1_strat));
    console.push_str(&format!("e-RTu:        {:.1}%  (Type I: {:.2}%)\n\n", power_agn, type1_agn));

    console.push_str("--- Type M Error ---\n");
    console.push_str(&format!("Effect at stop:  {:.3}\n", avg_effect_stop));
    console.push_str(&format!("Effect at final: {:.3}\n", avg_effect_final));
    console.push_str(&format!("Type M ratio:    {:.2}x\n\n", type_m));

    console.push_str("--- Futility Grid ---\n");
    console.push_str("Threshold  Triggered  Recovered\n");
    for (thresh, triggered, recovered) in &futility_grid {
        console.push_str(&format!("  {:.1}       {:5.1}%     {:5.1}%\n", thresh, triggered, recovered));
    }

    print!("{}", console);

    // HTML Report
    let html = format!(r#"<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>e-RTms Multi-State Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
body{{font-family:system-ui,-apple-system,sans-serif;max-width:1400px;margin:0 auto;padding:20px;background:#fafafa}}
h1{{color:#1a1a2e}}h2,h3{{color:#16213e}}
pre{{background:#fff;padding:15px;border-radius:8px;border:1px solid #ddd;overflow-x:auto;font-size:13px}}
.plot-container{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin:20px 0}}
.plot{{background:#fff;border-radius:8px;padding:10px;box-shadow:0 1px 3px rgba(0,0,0,0.1)}}
</style></head><body>
<h1>e-RTms Multi-State Report</h1>
<h2>Console Output</h2>
<pre>{}</pre>
<h2>Visualizations</h2>
<div class="plot-container">
<div class="plot"><div id="p1" style="height:350px"></div></div>
<div class="plot"><div id="p2" style="height:350px"></div></div>
</div>
<script>
var t_null={:?};var t_alt={:?};var threshold={};
Plotly.newPlot('p1',t_null.map((y,i)=>({{type:'scatter',y:y,line:{{color:'rgba(150,150,150,0.4)'}},showlegend:false}})),{{
  title:'Null Hypothesis',yaxis:{{type:'log',title:'e-value',range:[-1,Math.log10(threshold)+1]}},xaxis:{{title:'Transition'}},
  shapes:[{{type:'line',x0:0,x1:1,xref:'paper',y0:threshold,y1:threshold,line:{{color:'green',dash:'dash',width:2}}}}]}});
Plotly.newPlot('p2',t_alt.map((y,i)=>({{type:'scatter',y:y,line:{{color:'rgba(70,130,180,0.5)'}},showlegend:false}})),{{
  title:'Alternative Hypothesis',yaxis:{{type:'log',title:'e-value',range:[-1,Math.log10(threshold)+1]}},xaxis:{{title:'Transition'}},
  shapes:[{{type:'line',x0:0,x1:1,xref:'paper',y0:threshold,y1:threshold,line:{{color:'green',dash:'dash',width:2}}}}]}});
</script></body></html>"#, console, null_trajectories, alt_trajectories, threshold);

    File::create("multistate_report.html").unwrap().write_all(html.as_bytes()).unwrap();
    println!("\n>> Saved: multistate_report.html");
}
