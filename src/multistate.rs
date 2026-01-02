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
    if trimmed.is_empty() { None } else { trimmed.parse::<u64>().ok() }
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

// === DATA STRUCTURES ===

const STATE_WARD: usize = 0;
const STATE_ICU: usize = 1;
const STATE_HOME: usize = 2;
const STATE_DEAD: usize = 3;

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
    _good_rate_diff_at_stop: Option<f64>,
    _final_good_rate_diff: f64,
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

fn compute_e_rtms(transitions: &[Transition], burn_in: usize, ramp: usize) -> Vec<f64> {
    let n = transitions.len();
    if n == 0 { return vec![1.0]; }

    let mut wealth = vec![1.0; n];
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

        if is_trt {
            n_total_trt += 1.0;
            if is_good { n_good_trt += 1.0; }
        } else {
            n_total_ctrl += 1.0;
            if is_good { n_good_ctrl += 1.0; }
        }
    }
    wealth
}

fn run_single_trial<R: Rng + ?Sized>(
    rng: &mut R, n_patients: usize, p_ctrl: &TransitionMatrix, p_trt: &TransitionMatrix,
    max_days: usize, start_state: usize, burn_in: usize, ramp: usize, threshold: f64,
) -> (TrialResult, Vec<f64>) {
    let mut all_transitions: Vec<Transition> = Vec::new();

    for _ in 0..n_patients {
        let arm: u8 = if rng.gen_bool(0.5) { 1 } else { 0 };
        let p = if arm == 1 { p_trt } else { p_ctrl };
        let result = simulate_patient(rng, p, arm, max_days, start_state);
        all_transitions.extend(result.transitions);
    }

    if all_transitions.len() < burn_in {
        return (TrialResult { stopped_at: None, success: false, _good_rate_diff_at_stop: None, _final_good_rate_diff: 0.0 }, vec![1.0]);
    }

    let wealth = compute_e_rtms(&all_transitions, burn_in, ramp);
    let crossing = wealth.iter().position(|&w| w >= threshold);

    (TrialResult { stopped_at: crossing, success: crossing.is_some(), _good_rate_diff_at_stop: None, _final_good_rate_diff: 0.0 }, wealth)
}

// === SIMULATION RUNNER ===

struct SimResults {
    type1_error: f64,
    success_count: usize,
    avg_stop_trans: f64,
    median_transitions: f64,
    trajectories: Vec<Vec<f64>>,
    stop_times: Vec<f64>,
}

fn run_simulation<R: Rng + ?Sized>(
    rng: &mut R, n_patients: usize, n_sims: usize, p_ctrl: &TransitionMatrix, p_trt: &TransitionMatrix,
    max_days: usize, start_state: usize, burn_in: usize, ramp: usize, threshold: f64, is_null: bool,
) -> SimResults {
    let mut success_count = 0;
    let mut stop_times: Vec<f64> = Vec::new();
    let mut trajectories: Vec<Vec<f64>> = Vec::new();
    let mut trans_counts: Vec<f64> = Vec::new();

    let p_actual = if is_null { p_ctrl } else { p_trt };

    for sim in 0..n_sims {
        let (result, wealth) = run_single_trial(rng, n_patients, p_ctrl, p_actual, max_days, start_state, burn_in, ramp, threshold);
        trans_counts.push(wealth.len() as f64);

        if result.success {
            success_count += 1;
            if let Some(stop) = result.stopped_at { stop_times.push(stop as f64); }
        }

        if trajectories.len() < 100 { trajectories.push(wealth); }
        if (sim + 1) % 100 == 0 { print!("\rSimulation {}/{}", sim + 1, n_sims); io::stdout().flush().unwrap(); }
    }
    println!();

    let type1_error = if is_null { (success_count as f64 / n_sims as f64) * 100.0 } else { 0.0 };
    let avg_stop_trans = if !stop_times.is_empty() { stop_times.iter().sum::<f64>() / stop_times.len() as f64 } else { 0.0 };
    let median_transitions = { let mut s = trans_counts.clone(); s.sort_by(|a, b| a.partial_cmp(b).unwrap()); s[s.len() / 2] };

    SimResults { type1_error, success_count, avg_stop_trans, median_transitions, trajectories, stop_times }
}

fn compute_day28<R: Rng + ?Sized>(rng: &mut R, p: &TransitionMatrix, n: usize, max_days: usize, start: usize) -> [f64; 4] {
    let mut counts = [0usize; 4];
    for _ in 0..n { counts[simulate_patient(rng, p, 0, max_days, start).final_state] += 1; }
    let n = n as f64;
    [counts[0] as f64 / n, counts[1] as f64 / n, counts[2] as f64 / n, counts[3] as f64 / n]
}

// === HTML REPORT ===

fn build_html(n_patients: usize, n_sims: usize, threshold: f64, null: &SimResults, alt: &SimResults, d28_c: [f64; 4], d28_t: [f64; 4]) -> String {
    let power = (alt.success_count as f64 / n_sims as f64) * 100.0;
    format!(r#"<!DOCTYPE html><html><head><meta charset="utf-8"><title>e-RTms Report</title>
<script src="https://cdn.plot.ly/plotly-2.12.1.min.js"></script>
<style>body{{font-family:sans-serif;max-width:1000px;margin:0 auto;padding:20px}}h1{{color:#2c3e50}}table{{border-collapse:collapse;margin:15px 0}}td{{padding:8px 16px;border-bottom:1px solid #eee}}.hl{{background:#e8f4f8;font-weight:bold}}</style></head><body>
<h1>e-RTms Multi-State Report</h1>
<p>Generated: {}</p>
<h2>Model</h2><p>States: Ward, ICU, Home (abs), Dead (abs) | Start: ICU | 28 days</p>
<p>Good: ICU→Ward, Ward→Home | Bad: all others</p>
<h2>Parameters</h2><table><tr><td>N:</td><td>{}</td></tr><tr><td>Sims:</td><td>{}</td></tr><tr><td>Threshold:</td><td>{}</td></tr></table>
<h2>Day 28</h2><table><tr><th></th><th>Ward</th><th>ICU</th><th>Home</th><th>Dead</th></tr>
<tr><td>Ctrl:</td><td>{:.1}%</td><td>{:.1}%</td><td>{:.1}%</td><td>{:.1}%</td></tr>
<tr><td>Trt:</td><td>{:.1}%</td><td>{:.1}%</td><td>{:.1}%</td><td>{:.1}%</td></tr></table>
<h2>Results</h2><table><tr class="hl"><td>Type I Error:</td><td>{:.2}%</td></tr><tr class="hl"><td>Power:</td><td>{:.1}%</td></tr>
<tr><td>Median trans:</td><td>{:.0}</td></tr><tr><td>Avg stop:</td><td>{:.0}</td></tr></table>
<h2>Trajectories</h2><div id="p1" style="height:400px"></div><div id="p2" style="height:400px"></div>
<script>
var t_null={:?};var t_alt={:?};var stops={:?};
Plotly.newPlot('p1',t_null.slice(0,30).map((y,i)=>({{type:'scatter',y:y,line:{{color:'rgba(150,150,150,0.3)'}},showlegend:false}})),{{yaxis:{{type:'log',title:'e-value'}},xaxis:{{title:'Transition'}},shapes:[{{type:'line',x0:0,x1:1,xref:'paper',y0:{},y1:{},line:{{color:'red',dash:'dash'}}}}],title:'Null'}});
Plotly.newPlot('p2',t_alt.slice(0,30).map((y,i)=>({{type:'scatter',y:y,line:{{color:'rgba(70,130,180,0.3)'}},showlegend:false}})),{{yaxis:{{type:'log',title:'e-value'}},xaxis:{{title:'Transition'}},shapes:[{{type:'line',x0:0,x1:1,xref:'paper',y0:{},y1:{},line:{{color:'red',dash:'dash'}}}}],title:'Alternative'}});
</script></body></html>"#,
        chrono_lite(), n_patients, n_sims, threshold,
        d28_c[0]*100.0, d28_c[1]*100.0, d28_c[2]*100.0, d28_c[3]*100.0,
        d28_t[0]*100.0, d28_t[1]*100.0, d28_t[2]*100.0, d28_t[3]*100.0,
        null.type1_error, power, alt.median_transitions, alt.avg_stop_trans,
        null.trajectories, alt.trajectories, alt.stop_times,
        threshold, threshold, threshold, threshold)
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

    println!("\n--- Null ---");
    let null = run_simulation(&mut *rng, n_patients, n_sims, &p_ctrl, &p_ctrl, max_days, start, burn_in, ramp, threshold, true);
    println!("Type I Error: {:.2}%", null.type1_error);

    println!("\n--- Alternative ---");
    let alt = run_simulation(&mut *rng, n_patients, n_sims, &p_ctrl, &p_trt, max_days, start, burn_in, ramp, threshold, false);
    println!("Power: {:.1}%", (alt.success_count as f64 / n_sims as f64) * 100.0);

    let html = build_html(n_patients, n_sims, threshold, &null, &alt, d28_c, d28_t);
    File::create("multistate_report.html").unwrap().write_all(html.as_bytes()).unwrap();
    println!("\n>> Saved: multistate_report.html");
}