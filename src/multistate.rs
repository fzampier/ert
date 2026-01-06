use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::RngCore;
use std::fs::File;
use std::io::{self, Write};

use crate::ert_core::{get_input, get_input_usize, get_optional_input, chrono_lite, normal_cdf, normal_quantile};
use crate::agnostic::{AgnosticERT, Signal, Arm};

// === STATES ===

const STATE_WARD: usize = 0;
const STATE_ICU: usize = 1;
const STATE_HOME: usize = 2;
const STATE_DEAD: usize = 3;

// === DATA ===

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
            [0.000, 0.000, 1.000, 0.000], // Home (absorbing)
            [0.000, 0.000, 0.000, 1.000], // Dead (absorbing)
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

    // Small realistic effect: ~5% absolute improvement in Home at day 28
    fn small_treatment() -> Self {
        TransitionMatrix::new([
            [0.875, 0.060, 0.040, 0.025], // Ward: slightly better discharge
            [0.080, 0.907, 0.000, 0.013], // ICU: 8% vs 7% step-down rate
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
    arm: u8,
}

struct Trial {
    stop_n: Option<usize>,
    agnostic_stop_n: Option<usize>,
    effect_at_stop: f64,
    effect_final: f64,
    min_wealth: f64,
}

// === PROPORTIONAL ODDS BENCHMARK ===

fn calculate_proportional_or(ctrl: [f64; 4], trt: [f64; 4]) -> f64 {
    // Reorder by ordinal score (worst to best): Dead, ICU, Ward, Home
    let ctrl_ord = [ctrl[3], ctrl[1], ctrl[0], ctrl[2]];
    let trt_ord = [trt[3], trt[1], trt[0], trt[2]];

    let mut cum_ctrl = [0.0; 4];
    let mut cum_trt = [0.0; 4];
    cum_ctrl[0] = ctrl_ord[0];
    cum_trt[0] = trt_ord[0];
    for i in 1..4 {
        cum_ctrl[i] = cum_ctrl[i-1] + ctrl_ord[i];
        cum_trt[i] = cum_trt[i-1] + trt_ord[i];
    }

    let mut log_ors = Vec::new();
    for i in 0..3 {
        let p_ctrl = cum_ctrl[i].clamp(0.001, 0.999);
        let p_trt = cum_trt[i].clamp(0.001, 0.999);
        let or_j = (p_trt / (1.0 - p_trt)) / (p_ctrl / (1.0 - p_ctrl));
        if or_j > 0.0 { log_ors.push(or_j.ln()); }
    }

    if log_ors.is_empty() { return 1.0; }
    let mean_log_or = log_ors.iter().sum::<f64>() / log_ors.len() as f64;
    (-mean_log_or).exp()
}

fn calculate_mann_whitney_prob(ctrl: [f64; 4], trt: [f64; 4]) -> f64 {
    let ctrl_ord = [ctrl[3], ctrl[1], ctrl[0], ctrl[2]];
    let trt_ord = [trt[3], trt[1], trt[0], trt[2]];

    let mut p_win = 0.0;
    let mut p_tie = 0.0;
    for i in 0..4 {
        for j in 0..4 {
            let p_joint = trt_ord[i] * ctrl_ord[j];
            if i > j { p_win += p_joint; }
            else if i == j { p_tie += p_joint; }
        }
    }
    p_win + 0.5 * p_tie
}

fn proportional_odds_power(or: f64, n_total: usize, alpha: f64) -> f64 {
    if or <= 1.0 { return alpha; }
    let log_or = or.ln();
    let n = n_total as f64;
    let k = 4.0;
    let var_log_or = 4.0 * (k + 1.0) / (3.0 * n);
    let se_log_or = var_log_or.sqrt();
    let z = log_or / se_log_or;
    let z_alpha = normal_quantile(1.0 - alpha / 2.0);
    normal_cdf(z - z_alpha)
}

fn proportional_odds_sample_size(or: f64, power: f64, alpha: f64) -> usize {
    if or <= 1.0 { return 99999; }
    let log_or = or.ln();
    let z_alpha = normal_quantile(1.0 - alpha / 2.0);
    let z_beta = normal_quantile(power);
    let k = 4.0;
    let n = 4.0 * (k + 1.0) / (3.0 * log_or * log_or) * (z_alpha + z_beta).powi(2);
    (n.ceil() as usize).max(10)
}

// === SIMULATE ===

fn simulate_patient<R: Rng + ?Sized>(
    rng: &mut R, p: &TransitionMatrix, arm: u8, max_days: usize, start_state: usize,
) -> Vec<Transition> {
    let mut state = start_state;
    let mut transitions = Vec::new();

    for _ in 1..=max_days {
        if state == STATE_HOME || state == STATE_DEAD { break; }
        let new_state = p.sample_next(rng, state);
        if new_state != state {
            transitions.push(Transition { from: state, to: new_state, arm });
        }
        state = new_state;
    }
    transitions
}

fn is_good_transition(from: usize, to: usize) -> bool {
    (from == STATE_ICU && to == STATE_WARD) || (from == STATE_WARD && to == STATE_HOME)
}

fn run_single_trial<R: Rng + ?Sized>(
    rng: &mut R, n_patients: usize, p_ctrl: &TransitionMatrix, p_trt: &TransitionMatrix,
    max_days: usize, start_state: usize, burn_in: usize, ramp: usize, threshold: f64,
) -> (Trial, Vec<f64>) {
    let mut all_transitions: Vec<Transition> = Vec::new();

    for _ in 0..n_patients {
        let arm: u8 = if rng.gen_bool(0.5) { 1 } else { 0 };
        let p = if arm == 1 { p_trt } else { p_ctrl };
        all_transitions.extend(simulate_patient(rng, p, arm, max_days, start_state));
    }

    if all_transitions.len() < burn_in {
        return (Trial { stop_n: None, agnostic_stop_n: None, effect_at_stop: 0.0, effect_final: 0.0, min_wealth: 1.0 }, vec![1.0]);
    }

    // Compute wealth
    let n = all_transitions.len();
    let mut wealth = vec![1.0; n];
    let mut effects = vec![0.0; n];
    let (mut n_good_trt, mut n_total_trt, mut n_good_ctrl, mut n_total_ctrl) = (0.0, 0.0, 0.0, 0.0);

    for (i, trans) in all_transitions.iter().enumerate() {
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

    // Agnostic e-RT
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

    (Trial { stop_n, agnostic_stop_n, effect_at_stop, effect_final, min_wealth }, wealth)
}

fn compute_day28<R: Rng + ?Sized>(rng: &mut R, p: &TransitionMatrix, n: usize, max_days: usize, start: usize) -> [f64; 4] {
    let mut counts = [0usize; 4];
    for _ in 0..n {
        let mut state = start;
        for _ in 1..=max_days {
            if state == STATE_HOME || state == STATE_DEAD { break; }
            state = p.sample_next(rng, state);
        }
        counts[state] += 1;
    }
    let n = n as f64;
    [counts[0] as f64 / n, counts[1] as f64 / n, counts[2] as f64 / n, counts[3] as f64 / n]
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

// === MAIN ===

pub fn run() {
    println!("\n==========================================");
    println!("   e-RTms MULTI-STATE SIMULATION");
    println!("==========================================\n");

    println!("Model: Ward, ICU, Home (abs), Dead (abs)");
    println!("Start: ICU | Follow-up: 28 days\n");

    println!("Effect size:");
    println!("  1. Large (OR~1.6, Home +15%)");
    println!("  2. Small (OR~1.2, Home +5%) - more realistic");
    let effect_choice = get_input_usize("Select (1 or 2): ");
    let (p_ctrl, p_trt) = if effect_choice == 2 {
        println!("Using small effect: ICU->Ward 7%→8%");
        (TransitionMatrix::default_control(), TransitionMatrix::small_treatment())
    } else {
        println!("Using large effect: ICU->Ward 7%→9%");
        (TransitionMatrix::default_control(), TransitionMatrix::default_treatment())
    };

    let n_patients = get_input_usize("\nPatients per trial (e.g., 1000): ");
    let n_sims = get_input_usize("Simulations (e.g., 1000): ");
    let threshold = get_input("Threshold (default 20): ");
    let seed = get_optional_input("Seed (Enter for random): ");

    let burn_in = 30;
    let ramp = 50;
    let max_days = 28;
    let start = STATE_ICU;

    let mut rng: Box<dyn RngCore> = match seed {
        Some(s) => Box::new(StdRng::seed_from_u64(s)),
        None => Box::new(rand::thread_rng()),
    };

    // Day 28 distributions
    println!("\nComputing Day 28...");
    let d28_c = compute_day28(&mut *rng, &p_ctrl, 5000, max_days, start);
    let d28_t = compute_day28(&mut *rng, &p_trt, 5000, max_days, start);

    // Proportional odds benchmark
    let prop_or = calculate_proportional_or(d28_c, d28_t);
    let mann_whitney = calculate_mann_whitney_prob(d28_c, d28_t);
    let po_power = proportional_odds_power(prop_or, n_patients, 0.05);
    let n_80 = proportional_odds_sample_size(prop_or, 0.80, 0.05);
    let n_90 = proportional_odds_sample_size(prop_or, 0.90, 0.05);

    // Run simulations
    println!("\n--- Null ---");
    let mut null_trials = Vec::new();
    let mut null_trajectories = Vec::new();
    for sim in 0..n_sims {
        let (trial, wealth) = run_single_trial(&mut *rng, n_patients, &p_ctrl, &p_ctrl, max_days, start, burn_in, ramp, threshold);
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
        let (trial, wealth) = run_single_trial(&mut *rng, n_patients, &p_ctrl, &p_trt, max_days, start, burn_in, ramp, threshold);
        if alt_trajectories.len() < 30 { alt_trajectories.push(wealth); }
        alt_trials.push(trial);
        if (sim + 1) % 100 == 0 { print!("\rSimulation {}/{}", sim + 1, n_sims); io::stdout().flush().unwrap(); }
    }
    println!();

    let alt_ert_success = alt_trials.iter().filter(|t| t.stop_n.is_some()).count();
    let alt_agn_success = alt_trials.iter().filter(|t| t.agnostic_stop_n.is_some()).count();
    let power_ert = 100.0 * alt_ert_success as f64 / n_sims as f64;
    let power_agn = 100.0 * alt_agn_success as f64 / n_sims as f64;

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
    console.push_str(&format!("Model: Ward, ICU, Home (abs), Dead (abs)\n"));
    console.push_str(&format!("Start: ICU | Follow-up: 28 days\n\n"));
    console.push_str(&format!("Ctrl: Ward={:.1}% ICU={:.1}% Home={:.1}% Dead={:.1}%\n", d28_c[0]*100.0, d28_c[1]*100.0, d28_c[2]*100.0, d28_c[3]*100.0));
    console.push_str(&format!("Trt:  Ward={:.1}% ICU={:.1}% Home={:.1}% Dead={:.1}%\n\n", d28_t[0]*100.0, d28_t[1]*100.0, d28_t[2]*100.0, d28_t[3]*100.0));
    console.push_str(&format!("--- Proportional Odds Benchmark ---\n"));
    console.push_str(&format!("Proportional OR:     {:.2}\n", prop_or));
    console.push_str(&format!("Mann-Whitney P(T>C): {:.1}%\n", mann_whitney * 100.0));
    console.push_str(&format!("PO Power at N={}:  {:.1}%\n", n_patients, po_power * 100.0));
    console.push_str(&format!("PO N for 80%/90%:    {}/{}\n\n", n_80, n_90));
    console.push_str(&format!("--- Power at N={} ---\n", n_patients));
    console.push_str(&format!("Prop Odds:  {:.1}%\n", po_power * 100.0));
    console.push_str(&format!("e-RTms:     {:.1}%  (Type I: {:.2}%)\n", power_ert, type1_ert));
    console.push_str(&format!("e-RTu:      {:.1}%  (Type I: {:.2}%)\n\n", power_agn, type1_agn));
    console.push_str(&format!("--- Type M Error ---\n"));
    console.push_str(&format!("Effect at stop:  {:.3}\n", avg_effect_stop));
    console.push_str(&format!("Effect at final: {:.3}\n", avg_effect_final));
    console.push_str(&format!("Type M ratio:    {:.2}x\n\n", type_m));
    console.push_str(&format!("--- Futility Grid ---\n"));
    console.push_str(&format!("Threshold  Triggered  Recovered\n"));
    for (thresh, triggered, recovered) in &futility_grid {
        console.push_str(&format!("  {:.1}       {:5.1}%     {:5.1}%\n", thresh, triggered, recovered));
    }

    print!("{}", console);

    // HTML
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
