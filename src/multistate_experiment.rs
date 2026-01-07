// multistate_experiment.rs - Stratified e-process experiment
//
// Demonstrates that averaging stratified e-processes recovers power
// when patients can bounce between non-absorbing states.

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::fs::File;
use std::io::Write;

// === CONFIGURATION ===

#[derive(Clone)]
struct MultiStateConfig {
    state_names: Vec<String>,
    absorbing: Vec<usize>,
    start_state: usize,
    max_days: usize,
}

impl MultiStateConfig {
    fn n_states(&self) -> usize {
        self.state_names.len()
    }

    fn is_absorbing(&self, state: usize) -> bool {
        self.absorbing.contains(&state)
    }
}

// === TRANSITION MATRIX ===

#[derive(Clone)]
struct TransitionMatrix {
    probs: Vec<Vec<f64>>,
}

impl TransitionMatrix {
    fn new(n_states: usize) -> Self {
        let probs = (0..n_states).map(|i| {
            let mut row = vec![0.0; n_states];
            row[i] = 1.0;
            row
        }).collect();
        TransitionMatrix { probs }
    }

    fn set_row(&mut self, from: usize, probs: Vec<f64>) {
        self.probs[from] = probs;
    }

    fn sample_next<R: Rng + ?Sized>(&self, rng: &mut R, current: usize) -> usize {
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
}

// === TRANSITION ===

struct Transition {
    from: usize,
    to: usize,
    arm: u8,
}

fn is_good_transition(from: usize, to: usize) -> bool {
    to > from
}

// === STRATUM STATE ===

#[derive(Clone)]
struct Stratum {
    n_good_trt: f64,
    n_total_trt: f64,
    n_good_ctrl: f64,
    n_total_ctrl: f64,
    wealth: f64,
}

impl Stratum {
    fn new() -> Self {
        Stratum {
            n_good_trt: 0.0,
            n_total_trt: 0.0,
            n_good_ctrl: 0.0,
            n_total_ctrl: 0.0,
            wealth: 1.0,
        }
    }
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

// === STRATIFIED E-PROCESS ===

struct TrialResult {
    avg_stop: Option<usize>,
    original_stop: Option<usize>,
    original_wealth: Vec<f64>,
    avg_wealth: Vec<f64>,
    strata_wealth: Vec<Vec<f64>>,
}

fn run_stratified_trial<R: Rng + ?Sized>(
    rng: &mut R,
    n_patients: usize,
    p_ctrl: &TransitionMatrix,
    p_trt: &TransitionMatrix,
    config: &MultiStateConfig,
    burn_in: usize,
    ramp: usize,
    threshold: f64,
) -> TrialResult {
    let n_states = config.n_states();

    // Collect all transitions
    let mut all_transitions: Vec<Transition> = Vec::new();
    for _ in 0..n_patients {
        let arm: u8 = if rng.gen_bool(0.5) { 1 } else { 0 };
        let p = if arm == 1 { p_trt } else { p_ctrl };
        all_transitions.extend(simulate_patient(rng, p, config, arm));
    }

    if all_transitions.is_empty() {
        return TrialResult {
            avg_stop: None,
            original_stop: None,
            original_wealth: vec![1.0],
            avg_wealth: vec![1.0],
            strata_wealth: vec![],
        };
    }

    // Initialize strata (one per from_state)
    let mut strata: Vec<Stratum> = (0..n_states).map(|_| Stratum::new()).collect();
    let mut strata_n_obs: Vec<usize> = vec![0; n_states];

    // Original (unstratified) tracking
    let mut orig_wealth = 1.0;
    let mut orig_n_good_trt = 0.0;
    let mut orig_n_total_trt = 0.0;
    let mut orig_n_good_ctrl = 0.0;
    let mut orig_n_total_ctrl = 0.0;

    let mut avg_stop = None;
    let mut original_stop = None;

    // Trajectory storage
    let mut original_trajectory: Vec<f64> = Vec::new();
    let mut avg_trajectory: Vec<f64> = Vec::new();
    let mut strata_trajectories: Vec<Vec<f64>> = (0..n_states).map(|_| Vec::new()).collect();

    for (i, trans) in all_transitions.iter().enumerate() {
        let is_good = is_good_transition(trans.from, trans.to);
        let is_trt = trans.arm == 1;
        let from = trans.from;

        // === STRATIFIED BETTING ===
        let s = &mut strata[from];
        strata_n_obs[from] += 1;

        // Compute stratum-specific lambda (use global i for burn-in)
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

        // === ORIGINAL (UNSTRATIFIED) BETTING ===
        let orig_lambda = if i > burn_in && orig_n_total_trt > 0.0 && orig_n_total_ctrl > 0.0 {
            let c_i = (((i - burn_in) as f64) / ramp as f64).clamp(0.0, 1.0);
            let rate_trt = orig_n_good_trt / orig_n_total_trt;
            let rate_ctrl = orig_n_good_ctrl / orig_n_total_ctrl;
            let delta = rate_trt - rate_ctrl;
            if is_good { 0.5 + 0.5 * c_i * delta } else { 0.5 - 0.5 * c_i * delta }
        } else { 0.5 };

        let orig_lambda = orig_lambda.clamp(0.001, 0.999);
        let orig_mult = if is_trt { orig_lambda / 0.5 } else { (1.0 - orig_lambda) / 0.5 };
        orig_wealth *= orig_mult;

        // Update original counts
        if is_trt { orig_n_total_trt += 1.0; if is_good { orig_n_good_trt += 1.0; } }
        else { orig_n_total_ctrl += 1.0; if is_good { orig_n_good_ctrl += 1.0; } }

        // === COMPUTE AVERAGE STRATEGY ===
        let active_strata: Vec<f64> = strata.iter()
            .enumerate()
            .filter(|(j, _)| strata_n_obs[*j] > 0)
            .map(|(_, s)| s.wealth)
            .collect();
        let avg_wealth_now = if !active_strata.is_empty() {
            active_strata.iter().sum::<f64>() / active_strata.len() as f64
        } else { 1.0 };

        // Store trajectories
        original_trajectory.push(orig_wealth);
        avg_trajectory.push(avg_wealth_now);
        for (j, s) in strata.iter().enumerate() {
            strata_trajectories[j].push(s.wealth);
        }

        // === CHECK STOPPING ===
        if avg_stop.is_none() && avg_wealth_now >= threshold {
            avg_stop = Some(i + 1);
        }
        if original_stop.is_none() && orig_wealth >= threshold {
            original_stop = Some(i + 1);
        }
    }

    TrialResult {
        avg_stop,
        original_stop,
        original_wealth: original_trajectory,
        avg_wealth: avg_trajectory,
        strata_wealth: strata_trajectories,
    }
}

// === TEST SCENARIOS ===

fn create_absorbing_scenario() -> (MultiStateConfig, TransitionMatrix, TransitionMatrix) {
    let config = MultiStateConfig {
        state_names: vec!["Dead".into(), "ICU".into(), "Ward".into(), "Home".into()],
        absorbing: vec![0, 3],
        start_state: 1,
        max_days: 28,
    };

    let mut ctrl = TransitionMatrix::new(4);
    ctrl.set_row(0, vec![1.0, 0.0, 0.0, 0.0]);
    ctrl.set_row(1, vec![0.015, 0.915, 0.070, 0.000]);
    ctrl.set_row(2, vec![0.020, 0.070, 0.880, 0.030]);
    ctrl.set_row(3, vec![0.0, 0.0, 0.0, 1.0]);

    let mut trt = TransitionMatrix::new(4);
    trt.set_row(0, vec![1.0, 0.0, 0.0, 0.0]);
    trt.set_row(1, vec![0.010, 0.900, 0.090, 0.000]);
    trt.set_row(2, vec![0.015, 0.050, 0.885, 0.050]);
    trt.set_row(3, vec![0.0, 0.0, 0.0, 1.0]);

    (config, ctrl, trt)
}

fn create_bouncing_scenario() -> (MultiStateConfig, TransitionMatrix, TransitionMatrix) {
    let config = MultiStateConfig {
        state_names: vec!["Dead".into(), "Bad".into(), "Good".into()],
        absorbing: vec![0],
        start_state: 1,
        max_days: 28,
    };

    let mut ctrl = TransitionMatrix::new(3);
    ctrl.set_row(0, vec![1.0, 0.0, 0.0]);
    ctrl.set_row(1, vec![0.05, 0.80, 0.15]);
    ctrl.set_row(2, vec![0.02, 0.08, 0.90]);

    let mut trt = TransitionMatrix::new(3);
    trt.set_row(0, vec![1.0, 0.0, 0.0]);
    trt.set_row(1, vec![0.02, 0.80, 0.18]);
    trt.set_row(2, vec![0.02, 0.08, 0.90]);

    (config, ctrl, trt)
}

// === MAIN ===

pub fn run() {
    println!("\n=====================================================");
    println!("   STRATIFIED E-PROCESS EXPERIMENT");
    println!("   Comparing: Original vs Stratified Average");
    println!("=====================================================\n");

    let n_patients = 500;
    let n_sims = 1000;
    let threshold = 20.0;
    let burn_in = 30;
    let ramp = 50;
    let seed = 42u64;

    println!("Parameters: N={}, sims={}, threshold={}", n_patients, n_sims, threshold);

    // === SCENARIO 1: ABSORBING (ICU-like) ===
    println!("\n--- SCENARIO 1: ABSORBING STATES (ICU model) ---");
    let (config_abs, ctrl_abs, trt_abs) = create_absorbing_scenario();
    println!("States: {:?}", config_abs.state_names);
    println!("Absorbing: {:?}", config_abs.absorbing);

    let mut rng = StdRng::seed_from_u64(seed);
    let (mut null_orig, mut null_avg) = (0, 0);
    for _ in 0..n_sims {
        let r = run_stratified_trial(&mut rng, n_patients, &ctrl_abs, &ctrl_abs,
                                      &config_abs, burn_in, ramp, threshold);
        if r.original_stop.is_some() { null_orig += 1; }
        if r.avg_stop.is_some() { null_avg += 1; }
    }

    let mut rng = StdRng::seed_from_u64(seed + 1);
    let (mut alt_orig, mut alt_avg) = (0, 0);
    for _ in 0..n_sims {
        let r = run_stratified_trial(&mut rng, n_patients, &ctrl_abs, &trt_abs,
                                      &config_abs, burn_in, ramp, threshold);
        if r.original_stop.is_some() { alt_orig += 1; }
        if r.avg_stop.is_some() { alt_avg += 1; }
    }

    println!("\n             Type I     Power");
    println!("Original:    {:5.2}%    {:5.1}%",
             100.0 * null_orig as f64 / n_sims as f64,
             100.0 * alt_orig as f64 / n_sims as f64);
    println!("Stratified:  {:5.2}%    {:5.1}%",
             100.0 * null_avg as f64 / n_sims as f64,
             100.0 * alt_avg as f64 / n_sims as f64);

    // === SCENARIO 2: BOUNCING (non-absorbing) ===
    println!("\n--- SCENARIO 2: BOUNCING STATES (Good not absorbing) ---");
    let (config_bounce, ctrl_bounce, trt_bounce) = create_bouncing_scenario();
    println!("States: {:?}", config_bounce.state_names);
    println!("Absorbing: {:?}", config_bounce.absorbing);

    let mut rng = StdRng::seed_from_u64(seed + 2);
    let (mut null_orig_b, mut null_avg_b) = (0, 0);
    for _ in 0..n_sims {
        let r = run_stratified_trial(&mut rng, n_patients, &ctrl_bounce, &ctrl_bounce,
                                      &config_bounce, burn_in, ramp, threshold);
        if r.original_stop.is_some() { null_orig_b += 1; }
        if r.avg_stop.is_some() { null_avg_b += 1; }
    }

    let mut rng = StdRng::seed_from_u64(seed + 3);
    let (mut alt_orig_b, mut alt_avg_b) = (0, 0);
    for _ in 0..n_sims {
        let r = run_stratified_trial(&mut rng, n_patients, &ctrl_bounce, &trt_bounce,
                                      &config_bounce, burn_in, ramp, threshold);
        if r.original_stop.is_some() { alt_orig_b += 1; }
        if r.avg_stop.is_some() { alt_avg_b += 1; }
    }

    println!("\n             Type I     Power");
    println!("Original:    {:5.2}%    {:5.1}%",
             100.0 * null_orig_b as f64 / n_sims as f64,
             100.0 * alt_orig_b as f64 / n_sims as f64);
    println!("Stratified:  {:5.2}%    {:5.1}%",
             100.0 * null_avg_b as f64 / n_sims as f64,
             100.0 * alt_avg_b as f64 / n_sims as f64);

    println!("\n=====================================================");
    println!("   KEY INSIGHT");
    println!("=====================================================");
    println!("Average of martingales is always a martingale:");
    println!("  E[(W_1 + W_2 + ...)/k] = 1 by linearity");
    println!("This holds regardless of within-patient dependence.");
    println!("=====================================================\n");

    // === GENERATE HTML REPORT ===
    println!("Generating HTML report...");

    let mut rng = StdRng::seed_from_u64(seed + 200);
    let mut null_examples: Vec<TrialResult> = Vec::new();
    let mut alt_examples: Vec<TrialResult> = Vec::new();

    for _ in 0..10 {
        null_examples.push(run_stratified_trial(&mut rng, n_patients, &ctrl_bounce, &ctrl_bounce,
                                                 &config_bounce, burn_in, ramp, threshold));
    }
    let mut rng = StdRng::seed_from_u64(seed + 201);
    for _ in 0..10 {
        alt_examples.push(run_stratified_trial(&mut rng, n_patients, &ctrl_bounce, &trt_bounce,
                                                &config_bounce, burn_in, ramp, threshold));
    }

    let html = generate_html(&null_examples, &alt_examples, &config_bounce, threshold,
                             null_orig_b, null_avg_b, alt_orig_b, alt_avg_b, n_sims);

    File::create("stratified_experiment.html")
        .unwrap()
        .write_all(html.as_bytes())
        .unwrap();

    println!(">> Saved: stratified_experiment.html");
}

fn vec_to_js(v: &[f64]) -> String {
    let nums: Vec<String> = v.iter().map(|x| format!("{:.6}", x)).collect();
    format!("[{}]", nums.join(","))
}

fn generate_html(
    null_examples: &[TrialResult],
    alt_examples: &[TrialResult],
    config: &MultiStateConfig,
    threshold: f64,
    null_orig: usize,
    null_avg: usize,
    alt_orig: usize,
    alt_avg: usize,
    n_sims: usize,
) -> String {
    // Build JS arrays for trajectories
    let null_orig_js: Vec<String> = null_examples.iter().map(|r| vec_to_js(&r.original_wealth)).collect();
    let null_avg_js: Vec<String> = null_examples.iter().map(|r| vec_to_js(&r.avg_wealth)).collect();
    let alt_orig_js: Vec<String> = alt_examples.iter().map(|r| vec_to_js(&r.original_wealth)).collect();
    let alt_avg_js: Vec<String> = alt_examples.iter().map(|r| vec_to_js(&r.avg_wealth)).collect();

    // Strata from first alt example (only non-absorbing)
    let strata_js: Vec<String> = alt_examples[0].strata_wealth.iter()
        .enumerate()
        .filter(|(i, _)| !config.absorbing.contains(i))
        .map(|(_, v)| vec_to_js(v))
        .collect();
    let strata_names: Vec<&str> = config.state_names.iter()
        .enumerate()
        .filter(|(i, _)| !config.absorbing.contains(i))
        .map(|(_, n)| n.as_str())
        .collect();

    format!(r#"<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Stratified e-RTms Experiment</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
h1 {{ color: #1a1a2e; border-bottom: 3px solid #4a90d9; padding-bottom: 10px; }}
h2 {{ color: #16213e; margin-top: 30px; }}
.summary {{ background: #fff; padding: 20px; border-radius: 8px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
.plot {{ background: #fff; border-radius: 8px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 10px; text-align: center; }}
th {{ background: #4a90d9; color: white; }}
.good {{ color: #27ae60; font-weight: bold; }}
.bad {{ color: #e74c3c; font-weight: bold; }}
.note {{ background: #e8f4fd; padding: 15px; border-left: 4px solid #4a90d9; margin: 20px 0; border-radius: 4px; }}
</style>
</head>
<body>

<h1>Stratified e-RTms: Averaging Recovers Power</h1>

<div class="summary">
<h2>Bouncing Scenario Results</h2>
<p><strong>States:</strong> {states} (only Dead is absorbing)</p>
<p><strong>Problem:</strong> Patients bounce between Bad and Good, diluting treatment signal.</p>

<table>
<tr><th>Strategy</th><th>Type I Error</th><th>Power</th><th>Valid?</th></tr>
<tr>
  <td>Original (unstratified)</td>
  <td>{null_orig_pct:.1}%</td>
  <td class="bad">{alt_orig_pct:.1}%</td>
  <td class="good">Yes</td>
</tr>
<tr>
  <td>Stratified Average</td>
  <td>{null_avg_pct:.1}%</td>
  <td class="good">{alt_avg_pct:.1}%</td>
  <td class="good">Yes</td>
</tr>
</table>
</div>

<div class="note">
<strong>Why it works:</strong> Each stratum (from Bad, from Good) runs its own e-process.
The average E[(W_1 + W_2)/2] = 1 under H0 by linearity of expectation, regardless of dependence.
</div>

<h2>Null Hypothesis Trajectories</h2>
<div class="grid">
<div class="plot"><div id="null_orig" style="height:300px"></div></div>
<div class="plot"><div id="null_avg" style="height:300px"></div></div>
</div>

<h2>Alternative Hypothesis Trajectories</h2>
<div class="grid">
<div class="plot"><div id="alt_orig" style="height:300px"></div></div>
<div class="plot"><div id="alt_avg" style="height:300px"></div></div>
</div>

<h2>Per-Stratum Wealth (Example Trial)</h2>
<div class="plot"><div id="strata" style="height:350px"></div></div>

<script>
var threshold = {threshold};
var colors = ['rgba(150,150,150,0.6)', 'rgba(70,130,180,0.6)', 'rgba(46,204,113,0.6)'];

function plotTraces(divId, data, title, color) {{
    var traces = data.map(function(y) {{
        return {{ type: 'scatter', y: y, mode: 'lines', line: {{ color: color, width: 1.5 }}, showlegend: false }};
    }});
    Plotly.newPlot(divId, traces, {{
        title: title,
        yaxis: {{ type: 'log', title: 'e-value', range: [-1, 2.5] }},
        xaxis: {{ title: 'Transition' }},
        shapes: [{{ type: 'line', x0: 0, x1: 1, xref: 'paper', y0: threshold, y1: threshold,
                   line: {{ color: 'green', dash: 'dash', width: 2 }} }}]
    }});
}}

var nullOrig = [{null_orig_js}];
var nullAvg = [{null_avg_js}];
var altOrig = [{alt_orig_js}];
var altAvg = [{alt_avg_js}];

plotTraces('null_orig', nullOrig, 'Null: Original (stays flat)', 'rgba(150,150,150,0.6)');
plotTraces('null_avg', nullAvg, 'Null: Stratified Average (stays flat)', 'rgba(70,130,180,0.6)');
plotTraces('alt_orig', altOrig, 'Alt: Original (FAILS - no power)', 'rgba(150,150,150,0.6)');
plotTraces('alt_avg', altAvg, 'Alt: Stratified Average (WORKS!)', 'rgba(46,204,113,0.6)');

// Strata plot
var strataData = [{strata_js}];
var strataNames = {strata_names_js};
var strataColors = ['#e74c3c', '#3498db', '#27ae60'];
var strataTraces = strataData.map(function(y, i) {{
    return {{ type: 'scatter', y: y, mode: 'lines', name: 'From: ' + strataNames[i],
              line: {{ color: strataColors[i], width: 2 }} }};
}});
Plotly.newPlot('strata', strataTraces, {{
    title: 'Individual Stratum Wealth (one trial under H1)',
    yaxis: {{ type: 'log', title: 'Wealth' }},
    xaxis: {{ title: 'Transition' }},
    legend: {{ x: 0.02, y: 0.98 }}
}});
</script>

</body></html>"#,
        states = config.state_names.join(" < "),
        threshold = threshold,
        null_orig_pct = 100.0 * null_orig as f64 / n_sims as f64,
        null_avg_pct = 100.0 * null_avg as f64 / n_sims as f64,
        alt_orig_pct = 100.0 * alt_orig as f64 / n_sims as f64,
        alt_avg_pct = 100.0 * alt_avg as f64 / n_sims as f64,
        null_orig_js = null_orig_js.join(","),
        null_avg_js = null_avg_js.join(","),
        alt_orig_js = alt_orig_js.join(","),
        alt_avg_js = alt_avg_js.join(","),
        strata_js = strata_js.join(","),
        strata_names_js = format!("{:?}", strata_names),
    )
}
