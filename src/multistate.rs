use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::RngCore;
use std::fs::File;
use std::io::{self, Write};

use crate::ert_core::{get_input, get_input_usize, get_bool, get_optional_input, chrono_lite, normal_cdf, normal_quantile};
use crate::agnostic::{AgnosticERT, Signal, Arm};

// === DATA STRUCTURES ===

const STATE_WARD: usize = 0;
const STATE_ICU: usize = 1;
const STATE_HOME: usize = 2;
const STATE_DEAD: usize = 3;

// Ordinal scores for proportional odds (higher = better)
// Dead < ICU < Ward < Home
#[allow(dead_code)]
const ORDINAL_SCORES: [usize; 4] = [2, 1, 3, 0]; // Ward=2, ICU=1, Home=3, Dead=0

// === PROPORTIONAL ODDS BENCHMARK ===

/// Calculate proportional odds ratio from Day 28 distributions
/// Uses the relationship between Mann-Whitney and proportional OR
fn calculate_proportional_or(ctrl: [f64; 4], trt: [f64; 4]) -> f64 {
    // Convert state indices to ordinal scores
    // State order: Ward(0), ICU(1), Home(2), Dead(3)
    // Ordinal order: Dead < ICU < Ward < Home (0, 1, 2, 3)

    // Reorder distributions by ordinal score (worst to best)
    // Dead(3), ICU(1), Ward(0), Home(2)
    let ctrl_ord = [ctrl[3], ctrl[1], ctrl[0], ctrl[2]]; // Dead, ICU, Ward, Home
    let trt_ord = [trt[3], trt[1], trt[0], trt[2]];

    // Calculate cumulative probabilities (P(Y <= j))
    let mut cum_ctrl = [0.0; 4];
    let mut cum_trt = [0.0; 4];
    cum_ctrl[0] = ctrl_ord[0];
    cum_trt[0] = trt_ord[0];
    for i in 1..4 {
        cum_ctrl[i] = cum_ctrl[i-1] + ctrl_ord[i];
        cum_trt[i] = cum_trt[i-1] + trt_ord[i];
    }

    // Calculate log-OR at each cut-point (except last which is always 1)
    // OR_j = [P(Y<=j|trt)/(1-P(Y<=j|trt))] / [P(Y<=j|ctrl)/(1-P(Y<=j|ctrl))]
    // Under proportional odds, this should be constant
    let mut log_ors = Vec::new();
    for i in 0..3 {
        let p_ctrl = cum_ctrl[i].clamp(0.001, 0.999);
        let p_trt = cum_trt[i].clamp(0.001, 0.999);

        let odds_ctrl = p_ctrl / (1.0 - p_ctrl);
        let odds_trt = p_trt / (1.0 - p_trt);

        // OR < 1 means treatment is better (less likely to be in worse states)
        let or_j = odds_trt / odds_ctrl;
        if or_j > 0.0 {
            log_ors.push(or_j.ln());
        }
    }

    // Geometric mean of ORs (arithmetic mean of log-ORs)
    if log_ors.is_empty() { return 1.0; }
    let mean_log_or = log_ors.iter().sum::<f64>() / log_ors.len() as f64;

    // Return OR for treatment benefit (invert so OR > 1 = treatment better)
    (-mean_log_or).exp()
}

/// Calculate Mann-Whitney probability (P(Trt > Ctrl))
fn calculate_mann_whitney_prob(ctrl: [f64; 4], trt: [f64; 4]) -> f64 {
    // Reorder by ordinal score (worst to best): Dead, ICU, Ward, Home
    let ctrl_ord = [ctrl[3], ctrl[1], ctrl[0], ctrl[2]];
    let trt_ord = [trt[3], trt[1], trt[0], trt[2]];

    let mut p_win = 0.0;
    let mut p_tie = 0.0;

    for i in 0..4 {
        for j in 0..4 {
            let p_joint = trt_ord[i] * ctrl_ord[j];
            if i > j {
                p_win += p_joint; // Treatment in better state
            } else if i == j {
                p_tie += p_joint;
            }
        }
    }

    // P(Trt > Ctrl) + 0.5 * P(Trt = Ctrl) for ties
    p_win + 0.5 * p_tie
}

/// Power calculation for proportional odds model (Whitehead 1993 approximation)
/// N = total sample size for given power
fn proportional_odds_power(or: f64, n_total: usize, alpha: f64) -> f64 {
    if or <= 1.0 { return alpha; } // No effect

    let log_or = or.ln();
    let n = n_total as f64;

    // Whitehead formula for ordinal outcomes (K categories)
    // Variance ≈ (K+1)/(3*n*p*(1-p)) where p=0.5 for 1:1 randomization
    // Simplified: Var(log-OR) ≈ 4*(K+1)/(3*n) for balanced design with K=4
    let k = 4.0;
    let var_log_or = 4.0 * (k + 1.0) / (3.0 * n);
    let se_log_or = var_log_or.sqrt();

    // z-statistic
    let z = log_or / se_log_or;

    // Critical value for two-sided alpha
    let z_alpha = normal_quantile(1.0 - alpha / 2.0);

    // Power = P(Z > z_alpha - effect/SE)
    normal_cdf(z - z_alpha)
}

/// Sample size for proportional odds model to achieve target power
fn proportional_odds_sample_size(or: f64, power: f64, alpha: f64) -> usize {
    if or <= 1.0 { return 99999; }

    let log_or = or.ln();
    let z_alpha = normal_quantile(1.0 - alpha / 2.0);
    let z_beta = normal_quantile(power);

    // Whitehead formula inverted: n = 4*(K+1)/(3*log_or²) * (z_α + z_β)²
    let k = 4.0;
    let n = 4.0 * (k + 1.0) / (3.0 * log_or * log_or) * (z_alpha + z_beta).powi(2);

    (n.ceil() as usize).max(10)
}


/// Results from proportional odds benchmark
struct PropOddsBenchmark {
    or: f64,
    mann_whitney: f64,
    power_at_n: f64,
    n_for_80: usize,
    n_for_90: usize,
}

#[derive(Clone)]
struct TransitionMatrix {
    probs: [[f64; 4]; 4],
}

impl TransitionMatrix {
    fn new(probs: [[f64; 4]; 4]) -> Self { TransitionMatrix { probs } }

    fn default_control() -> Self {
        TransitionMatrix::new([
            [0.880, 0.070, 0.030, 0.020], // Ward
            [0.070, 0.915, 0.000, 0.015], // ICU
            [0.000, 0.000, 1.000, 0.000], // Home
            [0.000, 0.000, 0.000, 1.000], // Dead
        ])
    }

    fn default_treatment() -> Self {
        TransitionMatrix::new([
            [0.870, 0.050, 0.050, 0.030], // Ward
            [0.090, 0.900, 0.000, 0.010], // ICU
            [0.000, 0.000, 1.000, 0.000], // Home
            [0.000, 0.000, 0.000, 1.000], // Dead
        ])
    }

    fn sample_next<R: Rng + ?Sized>(&self, rng: &mut R, current: usize) -> usize {
        let r: f64 = rng.gen();
        let mut cumsum = 0.0;
        for (next_state, &prob) in self.probs[current].iter().enumerate() {
            cumsum += prob;
            if r < cumsum { return next_state; }
        }
        current
    }
}

struct Transition {
    from: usize,
    to: usize,
    _day: usize,
    arm: u8,
}

struct PatientResult {
    final_state: usize,
    transitions: Vec<Transition>,
}

struct TrialResult {
    stopped_at: Option<usize>,
    success: bool,
    effect_at_stop: Option<f64>,  // Good rate diff when threshold crossed
    effect_at_final: f64,         // Good rate diff at end of trial
}

// === SIMULATION ===

fn simulate_patient<R: Rng + ?Sized>(
    rng: &mut R, p: &TransitionMatrix, arm: u8, max_days: usize, start_state: usize,
) -> PatientResult {
    let mut state = start_state;
    let mut transitions = Vec::new();

    for day in 1..=max_days {
        if state == STATE_HOME || state == STATE_DEAD { break; }
        let new_state = p.sample_next(rng, state);
        if new_state != state {
            transitions.push(Transition { from: state, to: new_state, _day: day, arm });
        }
        state = new_state;
    }
    PatientResult { final_state: state, transitions }
}

fn is_good_transition(from: usize, to: usize) -> bool {
    (from == STATE_ICU && to == STATE_WARD) || (from == STATE_WARD && to == STATE_HOME)
}

fn run_single_trial<R: Rng + ?Sized>(
    rng: &mut R, n_patients: usize, p_ctrl: &TransitionMatrix, p_trt: &TransitionMatrix,
    max_days: usize, start_state: usize, burn_in: usize, ramp: usize, threshold: f64,
) -> (TrialResult, Vec<f64>, Vec<Transition>) {
    let mut all_transitions: Vec<Transition> = Vec::new();

    for _ in 0..n_patients {
        let arm: u8 = if rng.gen_bool(0.5) { 1 } else { 0 };
        let p = if arm == 1 { p_trt } else { p_ctrl };
        let result = simulate_patient(rng, p, arm, max_days, start_state);
        all_transitions.extend(result.transitions);
    }

    if all_transitions.len() < burn_in {
        return (TrialResult { stopped_at: None, success: false, effect_at_stop: None, effect_at_final: 0.0 }, vec![1.0], all_transitions);
    }

    // Compute wealth and track effect sizes at each point
    let (wealth, effects) = compute_e_rtms_with_effects(&all_transitions, burn_in, ramp);
    let crossing = wealth.iter().position(|&w| w >= threshold);

    let effect_at_stop = crossing.map(|idx| effects[idx]);
    let effect_at_final = *effects.last().unwrap_or(&0.0);

    (TrialResult {
        stopped_at: crossing,
        success: crossing.is_some(),
        effect_at_stop,
        effect_at_final,
    }, wealth, all_transitions)
}

/// Compute e-value wealth AND track effect size (good rate diff) at each step
fn compute_e_rtms_with_effects(transitions: &[Transition], burn_in: usize, ramp: usize) -> (Vec<f64>, Vec<f64>) {
    let n = transitions.len();
    if n == 0 { return (vec![1.0], vec![0.0]); }

    let mut wealth = vec![1.0; n];
    let mut effects = vec![0.0; n];
    let mut n_good_trt: f64 = 0.0;
    let mut n_total_trt: f64 = 0.0;
    let mut n_good_ctrl: f64 = 0.0;
    let mut n_total_ctrl: f64 = 0.0;

    for (i, trans) in transitions.iter().enumerate() {
        let is_good = is_good_transition(trans.from, trans.to);
        let is_trt = trans.arm == 1;

        let lambda = if i > burn_in && n_total_trt > 0.0 && n_total_ctrl > 0.0 {
            let c_i = (((i - burn_in) as f64) / ramp as f64).clamp(0.0, 1.0);
            let rate_trt = n_good_trt / n_total_trt;
            let rate_ctrl = n_good_ctrl / n_total_ctrl;
            let delta = rate_trt - rate_ctrl;
            if is_good { 0.5 + 0.5 * c_i * delta } else { 0.5 - 0.5 * c_i * delta }
        } else { 0.5 };

        let lambda = lambda.clamp(0.01, 0.99);
        let mult = if is_trt { lambda / 0.5 } else { (1.0 - lambda) / 0.5 };
        wealth[i] = if i > 0 { wealth[i - 1] * mult } else { mult };

        // Update counts
        if is_trt {
            n_total_trt += 1.0;
            if is_good { n_good_trt += 1.0; }
        } else {
            n_total_ctrl += 1.0;
            if is_good { n_good_ctrl += 1.0; }
        }

        // Track effect (good rate difference) at this point
        let rate_trt = if n_total_trt > 0.0 { n_good_trt / n_total_trt } else { 0.0 };
        let rate_ctrl = if n_total_ctrl > 0.0 { n_good_ctrl / n_total_ctrl } else { 0.0 };
        effects[i] = rate_trt - rate_ctrl;
    }
    (wealth, effects)
}

/// Compute agnostic e-RT for multistate transitions
/// Signal: good = is_good_transition, arm = treatment/control
fn compute_agnostic_multistate(
    transitions: &[Transition],
    burn_in: usize,
    ramp: usize,
    threshold: f64,
) -> (bool, Option<usize>) {
    let mut agnostic = AgnosticERT::new(burn_in, ramp, threshold);

    for (i, trans) in transitions.iter().enumerate() {
        let is_good = is_good_transition(trans.from, trans.to);
        let is_trt = trans.arm == 1;

        let signal = Signal {
            arm: if is_trt { Arm::Treatment } else { Arm::Control },
            good: is_good,
        };

        if agnostic.observe(signal) {
            return (true, Some(i + 1));
        }
    }
    (false, None)
}

// === SIMULATION RUNNER ===

struct SimResults {
    type1_error: f64,
    success_count: usize,
    _avg_stop_trans: f64,
    _median_transitions: f64,
    trajectories: Vec<Vec<f64>>,
    #[allow(dead_code)]
    _stop_times: Vec<f64>,
    // Type M error tracking
    _avg_effect_at_stop: f64,
    _avg_effect_at_final: f64,
    _type_m_error: f64,
    // Agnostic tracking
    agnostic_successes: usize,
    #[allow(dead_code)]
    _agnostic_stop_times: Vec<f64>,
}

fn run_simulation<R: Rng + ?Sized>(
    rng: &mut R, n_patients: usize, n_sims: usize, p_ctrl: &TransitionMatrix, p_trt: &TransitionMatrix,
    max_days: usize, start_state: usize, burn_in: usize, ramp: usize, threshold: f64, is_null: bool,
) -> SimResults {
    let mut success_count = 0;
    let mut stop_times: Vec<f64> = Vec::new();
    let mut trajectories: Vec<Vec<f64>> = Vec::new();
    let mut trans_counts: Vec<f64> = Vec::new();

    // Type M tracking
    let mut effects_at_stop: Vec<f64> = Vec::new();
    let mut effects_at_final: Vec<f64> = Vec::new();

    // Agnostic tracking
    let mut agnostic_successes = 0;
    let mut agnostic_stop_times: Vec<f64> = Vec::new();

    let p_actual = if is_null { p_ctrl } else { p_trt };

    for sim in 0..n_sims {
        let (result, wealth, transitions) = run_single_trial(rng, n_patients, p_ctrl, p_actual, max_days, start_state, burn_in, ramp, threshold);
        trans_counts.push(wealth.len() as f64);

        if result.success {
            success_count += 1;
            if let Some(stop) = result.stopped_at { stop_times.push(stop as f64); }
            // Track effects for Type M calculation (only for successful trials)
            if let Some(eff_stop) = result.effect_at_stop {
                effects_at_stop.push(eff_stop);
                effects_at_final.push(result.effect_at_final);
            }
        }

        // Run agnostic on same transitions
        let (agn_success, agn_stop) = compute_agnostic_multistate(&transitions, burn_in, ramp, threshold);
        if agn_success {
            agnostic_successes += 1;
            if let Some(stop) = agn_stop {
                agnostic_stop_times.push(stop as f64);
            }
        }

        if trajectories.len() < 100 { trajectories.push(wealth); }
        if (sim + 1) % 100 == 0 { print!("\rSimulation {}/{}", sim + 1, n_sims); io::stdout().flush().unwrap(); }
    }
    println!();

    let type1_error = if is_null { (success_count as f64 / n_sims as f64) * 100.0 } else { 0.0 };
    let avg_stop_trans = if !stop_times.is_empty() { stop_times.iter().sum::<f64>() / stop_times.len() as f64 } else { 0.0 };
    let median_transitions = { let mut s = trans_counts.clone(); s.sort_by(|a, b| a.partial_cmp(b).unwrap()); s[s.len() / 2] };

    // Calculate Type M error (magnification ratio)
    let (avg_effect_at_stop, avg_effect_at_final, type_m_error) = if !effects_at_stop.is_empty() {
        let avg_stop = effects_at_stop.iter().sum::<f64>() / effects_at_stop.len() as f64;
        let avg_final = effects_at_final.iter().sum::<f64>() / effects_at_final.len() as f64;
        let type_m = if avg_final.abs() > 0.001 { avg_stop / avg_final } else { 1.0 };
        (avg_stop, avg_final, type_m)
    } else {
        (0.0, 0.0, 1.0)
    };

    SimResults {
        type1_error, success_count,
        _avg_stop_trans: avg_stop_trans,
        _median_transitions: median_transitions,
        trajectories,
        _stop_times: stop_times,
        _avg_effect_at_stop: avg_effect_at_stop,
        _avg_effect_at_final: avg_effect_at_final,
        _type_m_error: type_m_error,
        agnostic_successes,
        _agnostic_stop_times: agnostic_stop_times,
    }
}

fn compute_day28<R: Rng + ?Sized>(rng: &mut R, p: &TransitionMatrix, n: usize, max_days: usize, start: usize) -> [f64; 4] {
    let mut counts = [0usize; 4];
    for _ in 0..n { counts[simulate_patient(rng, p, 0, max_days, start).final_state] += 1; }
    let n = n as f64;
    [counts[0] as f64 / n, counts[1] as f64 / n, counts[2] as f64 / n, counts[3] as f64 / n]
}

// === HTML REPORT ===

fn build_html(_n_patients: usize, _n_sims: usize, threshold: f64, null: &SimResults, alt: &SimResults,
              _d28_c: [f64; 4], _d28_t: [f64; 4], bench: &PropOddsBenchmark, ert_power: f64,
              agnostic_power: f64, _agnostic_null_rate: f64) -> String {

    format!(r#"<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>e-RTms Multi-State</title>
<script src="https://cdn.plot.ly/plotly-2.12.1.min.js"></script>
<style>body{{font-family:monospace;max-width:1200px;margin:0 auto;padding:20px}}pre{{background:#f5f5f5;padding:10px}}</style>
</head><body>
<h1>e-RTms Multi-State</h1>
<pre>
{}
Type I: {:.2}%  |  Power: {:.1}%  |  Prop Odds: {:.1}%  |  e-RTu: {:.1}%
</pre>
<div id="p1" style="height:400px"></div>
<div id="p2" style="height:400px"></div>
<script>
var t_null={:?};var t_alt={:?};var threshold={};
Plotly.newPlot('p1',t_null.slice(0,30).map((y,i)=>({{type:'scatter',y:y,line:{{color:'rgba(150,150,150,0.4)'}},showlegend:false}})),{{
  yaxis:{{type:'log',title:'e-value',range:[-1,Math.log10(threshold)+1]}},xaxis:{{title:'Transition'}},
  shapes:[{{type:'line',x0:0,x1:1,xref:'paper',y0:threshold,y1:threshold,line:{{color:'green',dash:'dash',width:2}}}}]}});
Plotly.newPlot('p2',t_alt.slice(0,30).map((y,i)=>({{type:'scatter',y:y,line:{{color:'rgba(70,130,180,0.5)'}},showlegend:false}})),{{
  yaxis:{{type:'log',title:'e-value',range:[-1,Math.log10(threshold)+1]}},xaxis:{{title:'Transition'}},
  shapes:[{{type:'line',x0:0,x1:1,xref:'paper',y0:threshold,y1:threshold,line:{{color:'green',dash:'dash',width:2}}}}]}});
</script></body></html>"#,
        chrono_lite(), null.type1_error, ert_power, bench.power_at_n * 100.0, agnostic_power,
        null.trajectories, alt.trajectories, threshold)
}

// === MAIN ===

pub fn run() {
    println!("\n==========================================");
    println!("   e-RTms MULTI-STATE SIMULATION");
    println!("==========================================\n");

    println!("Model: Ward, ICU, Home (abs), Dead (abs)");
    println!("Start: ICU | Follow-up: 28 days\n");

    let use_default = get_bool("Use default transition matrices?");
    let (p_ctrl, p_trt) = if use_default {
        println!("Using defaults: Ctrl ICU->Ward 7%, Trt ICU->Ward 9%");
        (TransitionMatrix::default_control(), TransitionMatrix::default_treatment())
    } else {
        println!("Custom not implemented, using defaults.");
        (TransitionMatrix::default_control(), TransitionMatrix::default_treatment())
    };

    let n_patients = get_input_usize("\nPatients per trial (e.g., 1000): ");
    let n_sims = get_input_usize("Simulations (e.g., 1000): ");
    let threshold = get_input("Threshold (default 20): ");
    let seed = get_optional_input("Seed (Enter for random): ");

    let burn_in = 30; let ramp = 50; let max_days = 28; let start = STATE_ICU;

    let mut rng: Box<dyn RngCore> = match seed {
        Some(s) => Box::new(StdRng::seed_from_u64(s)),
        None => Box::new(rand::thread_rng()),
    };

    println!("\nComputing Day 28...");
    let d28_c = compute_day28(&mut *rng, &p_ctrl, 5000, max_days, start);
    let d28_t = compute_day28(&mut *rng, &p_trt, 5000, max_days, start);
    println!("Ctrl: Ward={:.1}% ICU={:.1}% Home={:.1}% Dead={:.1}%", d28_c[0]*100.0, d28_c[1]*100.0, d28_c[2]*100.0, d28_c[3]*100.0);
    println!("Trt:  Ward={:.1}% ICU={:.1}% Home={:.1}% Dead={:.1}%", d28_t[0]*100.0, d28_t[1]*100.0, d28_t[2]*100.0, d28_t[3]*100.0);

    // === PROPORTIONAL ODDS BENCHMARK ===
    println!("\n--- Proportional Odds Benchmark ---");
    println!("Ordinal scale: Dead < ICU < Ward < Home");

    let prop_or = calculate_proportional_or(d28_c, d28_t);
    let mann_whitney = calculate_mann_whitney_prob(d28_c, d28_t);
    let po_power = proportional_odds_power(prop_or, n_patients, 0.05);
    let n_80 = proportional_odds_sample_size(prop_or, 0.80, 0.05);
    let n_90 = proportional_odds_sample_size(prop_or, 0.90, 0.05);

    let benchmark = PropOddsBenchmark {
        or: prop_or,
        mann_whitney,
        power_at_n: po_power,
        n_for_80: n_80,
        n_for_90: n_90,
    };

    println!("Proportional OR:     {:.2} (treatment benefit)", benchmark.or);
    println!("Mann-Whitney P(T>C): {:.1}%", benchmark.mann_whitney * 100.0);
    println!("PO Power at N={}:  {:.1}%", n_patients, benchmark.power_at_n * 100.0);
    println!("PO N for 80% power:  {}", benchmark.n_for_80);
    println!("PO N for 90% power:  {}", benchmark.n_for_90);

    println!("\n--- Null ---");
    let null = run_simulation(&mut *rng, n_patients, n_sims, &p_ctrl, &p_ctrl, max_days, start, burn_in, ramp, threshold, true);
    println!("Type I Error: {:.2}%", null.type1_error);

    println!("\n--- Alternative ---");
    let alt = run_simulation(&mut *rng, n_patients, n_sims, &p_ctrl, &p_trt, max_days, start, burn_in, ramp, threshold, false);
    let ert_power = (alt.success_count as f64 / n_sims as f64) * 100.0;
    let agnostic_power = (alt.agnostic_successes as f64 / n_sims as f64) * 100.0;
    let agnostic_null_rate = (null.agnostic_successes as f64 / n_sims as f64) * 100.0;
    println!("e-RTms Power:    {:.1}%", ert_power);
    println!("e-RTu Power:     {:.1}%", agnostic_power);
    println!("e-RTu Type I:    {:.2}%", agnostic_null_rate);

    // Three-tier power comparison
    println!("\n--- Power Comparison (Three-Tier Hierarchy) ---");
    println!("Prop. Odds (fixed):  {:.1}%  <- ceiling (traditional)", benchmark.power_at_n * 100.0);
    println!("e-RTms (sequential): {:.1}%  <- domain-aware sequential", ert_power);
    println!("e-RTu:               {:.1}%  <- floor (universal)", agnostic_power);
    println!("\nDomain knowledge:    {:+.1}%", ert_power - agnostic_power);
    println!("Sequential cost:     -{:.1}%", benchmark.power_at_n * 100.0 - ert_power);

    // Type M error
    println!("\n--- Type M Error (Magnification) ---");
    println!("Effect at stop:  {:.3}", alt._avg_effect_at_stop);
    println!("Effect at final: {:.3}", alt._avg_effect_at_final);
    println!("Type M ratio:    {:.2}x", alt._type_m_error);

    let html = build_html(n_patients, n_sims, threshold, &null, &alt, d28_c, d28_t, &benchmark, ert_power, agnostic_power, agnostic_null_rate);
    File::create("multistate_report.html").unwrap().write_all(html.as_bytes()).unwrap();
    println!("\n>> Saved: multistate_report.html");
}

#[allow(dead_code)]
fn estimate_n_for_power<R: Rng + ?Sized>(
    _rng: &mut R, target: f64, _p_ctrl: &TransitionMatrix, _p_trt: &TransitionMatrix,
    _max_days: usize, _start: usize, _burn_in: usize, _ramp: usize, _threshold: f64, current_n: usize,
) -> String {
    // Quick heuristic based on current results
    // More accurate would require running simulations at multiple N
    // For now, use scaling relationship: power ∝ sqrt(N)
    let current_power = 0.5; // Assume we're roughly at 50% if unsure
    if current_power >= target {
        return format!("<{}", current_n);
    }

    // Very rough: just indicate it's similar to PO or needs more exploration
    "≈PO".to_string()
}