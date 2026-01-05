// agnostic.rs - Universal "Bet on Sight" e-RT
//
// Core abstraction: The e-process only sees (arm, good/bad) signals.
// It doesn't know what generated them - binary, continuous, multistate,
// survival, kpop streams, Embraer jets - doesn't matter.

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

// === CORE: THE UNIVERSAL SIGNAL ===

#[derive(Clone, Copy, PartialEq)]
pub enum Arm {
    Treatment,
    Control,
}

#[derive(Clone, Copy)]
pub struct Signal {
    pub arm: Arm,
    pub good: bool,
}

impl Signal {
    pub fn new(arm: Arm, good: bool) -> Self {
        Signal { arm, good }
    }

    pub fn treatment(good: bool) -> Self {
        Signal { arm: Arm::Treatment, good }
    }

    pub fn control(good: bool) -> Self {
        Signal { arm: Arm::Control, good }
    }
}

// === THE AGNOSTIC E-PROCESS ===

pub struct AgnosticERT {
    // Counts
    n_trt: f64,
    n_ctrl: f64,
    good_trt: f64,
    good_ctrl: f64,

    // State
    wealth: f64,
    wealth_history: Vec<f64>,
    effect_history: Vec<f64>,
    signal_count: usize,

    // Parameters
    burn_in: usize,
    ramp: usize,
    threshold: f64,

    // Result
    stopped: bool,
    stopped_at: Option<usize>,
}

impl AgnosticERT {
    pub fn new(burn_in: usize, ramp: usize, threshold: f64) -> Self {
        AgnosticERT {
            n_trt: 0.0,
            n_ctrl: 0.0,
            good_trt: 0.0,
            good_ctrl: 0.0,
            wealth: 1.0,
            wealth_history: vec![1.0],
            effect_history: vec![0.0],
            signal_count: 0,
            burn_in,
            ramp,
            threshold,
            stopped: false,
            stopped_at: None,
        }
    }

    /// Feed a signal. Returns true if threshold crossed.
    pub fn observe(&mut self, signal: Signal) -> bool {
        if self.stopped {
            return true;
        }

        self.signal_count += 1;

        // Compute wager (lambda)
        let lambda = if self.signal_count > self.burn_in
            && self.n_trt > 0.0
            && self.n_ctrl > 0.0
        {
            let rate_trt = self.good_trt / self.n_trt;
            let rate_ctrl = self.good_ctrl / self.n_ctrl;
            let delta = rate_trt - rate_ctrl;

            let c_i = (((self.signal_count - self.burn_in) as f64) / self.ramp as f64).clamp(0.0, 1.0);

            // Bet direction based on observed effect and signal type
            let base = if signal.good {
                0.5 + 0.5 * c_i * delta
            } else {
                0.5 - 0.5 * c_i * delta
            };
            base.clamp(0.01, 0.99)
        } else {
            0.5 // No information yet, neutral bet
        };

        // Update wealth
        let mult = match signal.arm {
            Arm::Treatment => lambda / 0.5,
            Arm::Control => (1.0 - lambda) / 0.5,
        };
        self.wealth *= mult;

        // Update counts (AFTER betting, for next round)
        match signal.arm {
            Arm::Treatment => {
                self.n_trt += 1.0;
                if signal.good {
                    self.good_trt += 1.0;
                }
            }
            Arm::Control => {
                self.n_ctrl += 1.0;
                if signal.good {
                    self.good_ctrl += 1.0;
                }
            }
        }

        // Track history
        self.wealth_history.push(self.wealth);
        let rate_trt = if self.n_trt > 0.0 { self.good_trt / self.n_trt } else { 0.0 };
        let rate_ctrl = if self.n_ctrl > 0.0 { self.good_ctrl / self.n_ctrl } else { 0.0 };
        self.effect_history.push(rate_trt - rate_ctrl);

        // Check threshold
        if self.wealth >= self.threshold {
            self.stopped = true;
            self.stopped_at = Some(self.signal_count);
        }

        self.stopped
    }

    // Getters
    pub fn wealth(&self) -> f64 {
        self.wealth
    }
    pub fn stopped(&self) -> bool {
        self.stopped
    }
    pub fn stopped_at(&self) -> Option<usize> {
        self.stopped_at
    }
    pub fn history(&self) -> &[f64] {
        &self.wealth_history
    }
    pub fn effect_history(&self) -> &[f64] {
        &self.effect_history
    }
    pub fn rates(&self) -> (f64, f64) {
        let r_t = if self.n_trt > 0.0 { self.good_trt / self.n_trt } else { 0.0 };
        let r_c = if self.n_ctrl > 0.0 { self.good_ctrl / self.n_ctrl } else { 0.0 };
        (r_t, r_c)
    }
    pub fn counts(&self) -> (f64, f64, f64, f64) {
        (self.n_trt, self.good_trt, self.n_ctrl, self.good_ctrl)
    }
    pub fn effect_at_stop(&self) -> Option<f64> {
        self.stopped_at.map(|idx| self.effect_history[idx])
    }
    pub fn effect_at_final(&self) -> f64 {
        *self.effect_history.last().unwrap_or(&0.0)
    }
}

// === SIMULATION RESULTS ===

struct SimResults {
    rejection_rate: f64,
    avg_stop: f64,
    median_signals: f64,
    trajectories: Vec<Vec<f64>>,
    // Type M
    avg_effect_at_stop: f64,
    avg_effect_at_final: f64,
    type_m: f64,
}

// === DEMONSTRATION: BINARY SIGNAL GENERATOR ===

fn generate_binary_signal<R: Rng + ?Sized>(rng: &mut R, p_trt: f64, p_ctrl: f64) -> Signal {
    let is_trt = rng.gen_bool(0.5);
    let p = if is_trt { p_trt } else { p_ctrl };
    let good = rng.gen_bool(p);
    Signal {
        arm: if is_trt { Arm::Treatment } else { Arm::Control },
        good,
    }
}

// === SIMULATION ===

fn run_simulation<R: Rng + ?Sized>(
    rng: &mut R,
    n_signals: usize,
    n_sims: usize,
    p_trt: f64,
    p_ctrl: f64,
    burn_in: usize,
    ramp: usize,
    threshold: f64,
) -> SimResults {
    let mut rejections = 0;
    let mut stop_times: Vec<f64> = Vec::new();
    let mut signal_counts: Vec<f64> = Vec::new();
    let mut trajectories: Vec<Vec<f64>> = Vec::new();

    // Type M tracking
    let mut effects_at_stop: Vec<f64> = Vec::new();
    let mut effects_at_final: Vec<f64> = Vec::new();

    for sim in 0..n_sims {
        let mut ert = AgnosticERT::new(burn_in, ramp, threshold);

        for _ in 0..n_signals {
            let signal = generate_binary_signal(rng, p_trt, p_ctrl);
            if ert.observe(signal) {
                break;
            }
        }

        signal_counts.push(ert.signal_count as f64);

        if ert.stopped() {
            rejections += 1;
            if let Some(t) = ert.stopped_at() {
                stop_times.push(t as f64);
            }
            if let Some(eff) = ert.effect_at_stop() {
                effects_at_stop.push(eff);
                effects_at_final.push(ert.effect_at_final());
            }
        }

        if trajectories.len() < 50 {
            trajectories.push(ert.history().to_vec());
        }

        if (sim + 1) % 100 == 0 {
            print!("\rSimulation {}/{}", sim + 1, n_sims);
            io::stdout().flush().unwrap();
        }
    }
    println!();

    let rejection_rate = rejections as f64 / n_sims as f64;
    let avg_stop = if !stop_times.is_empty() {
        stop_times.iter().sum::<f64>() / stop_times.len() as f64
    } else {
        0.0
    };
    let median_signals = {
        let mut s = signal_counts.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s[s.len() / 2]
    };

    // Type M
    let (avg_effect_at_stop, avg_effect_at_final, type_m) = if !effects_at_stop.is_empty() {
        let avg_stop = effects_at_stop.iter().sum::<f64>() / effects_at_stop.len() as f64;
        let avg_final = effects_at_final.iter().sum::<f64>() / effects_at_final.len() as f64;
        let tm = if avg_final.abs() > 0.001 { avg_stop / avg_final } else { 1.0 };
        (avg_stop, avg_final, tm)
    } else {
        (0.0, 0.0, 1.0)
    };

    SimResults {
        rejection_rate,
        avg_stop,
        median_signals,
        trajectories,
        avg_effect_at_stop,
        avg_effect_at_final,
        type_m,
    }
}

// === HTML REPORT ===

fn build_html(
    n_signals: usize,
    n_sims: usize,
    threshold: f64,
    p_trt: f64,
    p_ctrl: f64,
    null: &SimResults,
    alt: &SimResults,
) -> String {
    format!(
        r#"<!DOCTYPE html><html><head><meta charset="utf-8"><title>Agnostic e-RT Report</title>
<script src="https://cdn.plot.ly/plotly-2.12.1.min.js"></script>
<style>
body{{font-family:sans-serif;max-width:1000px;margin:0 auto;padding:20px;background:#f8f9fa}}
h1{{color:#2c3e50}}
h2{{color:#34495e;border-bottom:2px solid #9b59b6;padding-bottom:5px;margin-top:30px}}
table{{border-collapse:collapse;margin:15px 0;background:white;box-shadow:0 1px 3px rgba(0,0,0,0.1)}}
th,td{{padding:10px 16px;border-bottom:1px solid #eee;text-align:left}}
.hl{{background:#f3e8ff;font-weight:bold}}
.note{{font-size:0.9em;color:#7f8c8d;margin-top:10px}}
</style></head><body>
<h1>Agnostic e-RT Report</h1>
<p>Universal "Bet on Sight" Sequential Test</p>
<p>Generated: {}</p>

<h2>Design</h2>
<table>
<tr><td>Signals per trial:</td><td>{}</td></tr>
<tr><td>Simulations:</td><td>{}</td></tr>
<tr><td>Threshold (1/α):</td><td>{}</td></tr>
<tr><td>Burn-in:</td><td>50</td></tr>
<tr><td>Ramp:</td><td>100</td></tr>
</table>

<h2>Signal Generator (Demo: Binary)</h2>
<table>
<tr><td>P(good | treatment):</td><td>{:.1}%</td></tr>
<tr><td>P(good | control):</td><td>{:.1}%</td></tr>
<tr><td>True effect (δ):</td><td>{:.1}%</td></tr>
</table>

<h2>Results</h2>
<table>
<tr class="hl"><td>Type I Error:</td><td>{:.2}%</td></tr>
<tr class="hl"><td>Power:</td><td>{:.1}%</td></tr>
<tr><td>Median signals:</td><td>{:.0}</td></tr>
<tr><td>Avg stop (when rejected):</td><td>{:.0}</td></tr>
</table>

<h2>Type M Error</h2>
<table>
<tr><td>Effect at stop:</td><td>{:.3}</td></tr>
<tr><td>Effect at final:</td><td>{:.3}</td></tr>
<tr class="hl"><td>Type M ratio:</td><td>{:.2}x</td></tr>
</table>

<h2>e-Value Trajectories</h2>
<div id="p1" style="height:400px"></div>
<div id="p2" style="height:400px"></div>

<script>
var t_null={:?};
var t_alt={:?};
var threshold={};

Plotly.newPlot('p1',t_null.slice(0,30).map((y,i)=>({{type:'scatter',y:y,line:{{color:'rgba(150,150,150,0.4)'}},showlegend:false}})),{{
    yaxis:{{type:'log',title:'e-value',range:[-1, Math.log10(threshold)+1]}},
    xaxis:{{title:'Signal'}},
    shapes:[{{type:'line',x0:0,x1:1,xref:'paper',y0:threshold,y1:threshold,line:{{color:'red',dash:'dash',width:2}}}}],
    title:'Null (no effect)'
}});

Plotly.newPlot('p2',t_alt.slice(0,30).map((y,i)=>({{type:'scatter',y:y,line:{{color:'rgba(155,89,182,0.5)'}},showlegend:false}})),{{
    yaxis:{{type:'log',title:'e-value',range:[-1, Math.log10(threshold)+1]}},
    xaxis:{{title:'Signal'}},
    shapes:[{{type:'line',x0:0,x1:1,xref:'paper',y0:threshold,y1:threshold,line:{{color:'red',dash:'dash',width:2}}}}],
    title:'Alternative (treatment works)'
}});
</script>

</body></html>"#,
        chrono_lite(),
        n_signals,
        n_sims,
        threshold,
        p_trt * 100.0,
        p_ctrl * 100.0,
        (p_trt - p_ctrl) * 100.0,
        null.rejection_rate * 100.0,
        alt.rejection_rate * 100.0,
        alt.median_signals,
        alt.avg_stop,
        alt.avg_effect_at_stop,
        alt.avg_effect_at_final,
        alt.type_m,
        null.trajectories,
        alt.trajectories,
        threshold,
    )
}

// === MAIN ===

pub fn run() {
    println!("\n==========================================");
    println!("   AGNOSTIC e-RT (Bet on Sight)");
    println!("==========================================\n");

    println!("The universal e-process. Sees only: (arm, good/bad)");
    println!("Doesn't care what generates the signals.\n");

    println!("Demo mode: Binary signal generator");
    println!("─────────────────────────────────────────\n");

    let p_ctrl = get_input("P(good | control), e.g. 0.30: ");
    let p_trt = get_input("P(good | treatment), e.g. 0.40: ");

    let n_signals = get_input_usize("\nSignals per trial (e.g., 500): ");
    let n_sims = get_input_usize("Simulations (e.g., 1000): ");
    let threshold = get_input("Threshold (default 20): ");
    let seed = get_optional_input("Seed (Enter for random): ");

    let burn_in = 50;
    let ramp = 100;

    let mut rng: Box<dyn RngCore> = match seed {
        Some(s) => Box::new(StdRng::seed_from_u64(s)),
        None => Box::new(rand::thread_rng()),
    };

    println!("\nTrue effect: {:.1}%", (p_trt - p_ctrl) * 100.0);

    println!("\n--- Null (both arms = control) ---");
    let null = run_simulation(&mut *rng, n_signals, n_sims, p_ctrl, p_ctrl, burn_in, ramp, threshold);
    println!("Type I Error: {:.2}%", null.rejection_rate * 100.0);

    println!("\n--- Alternative (treatment works) ---");
    let alt = run_simulation(&mut *rng, n_signals, n_sims, p_trt, p_ctrl, burn_in, ramp, threshold);
    println!("Power: {:.1}%", alt.rejection_rate * 100.0);

    println!("\n--- Type M Error ---");
    println!("Effect at stop:  {:.3}", alt.avg_effect_at_stop);
    println!("Effect at final: {:.3}", alt.avg_effect_at_final);
    println!("Type M ratio:    {:.2}x", alt.type_m);

    let html = build_html(n_signals, n_sims, threshold, p_trt, p_ctrl, &null, &alt);
    File::create("agnostic_report.html")
        .unwrap()
        .write_all(html.as_bytes())
        .unwrap();
    println!("\n>> Saved: agnostic_report.html");
}
