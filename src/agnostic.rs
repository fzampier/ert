// agnostic.rs - Universal "Bet on Sight" e-RT
//
// Core abstraction: The e-process only sees (arm, good/bad) signals.
// It doesn't know what generated them - binary, continuous, multistate,
// survival, kpop streams, Embraer jets - doesn't matter.

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::RngCore;
use std::fs::File;
use std::io::{self, Write};

use crate::ert_core::{get_input, get_input_usize, get_optional_input, chrono_lite};

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

#[allow(dead_code)]
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
    #[allow(dead_code)]
    pub fn wealth(&self) -> f64 { self.wealth }
    pub fn stopped(&self) -> bool { self.stopped }
    pub fn stopped_at(&self) -> Option<usize> { self.stopped_at }
    pub fn history(&self) -> &[f64] { &self.wealth_history }
    #[allow(dead_code)]
    pub fn effect_history(&self) -> &[f64] { &self.effect_history }
    #[allow(dead_code)]
    pub fn rates(&self) -> (f64, f64) {
        let r_t = if self.n_trt > 0.0 { self.good_trt / self.n_trt } else { 0.0 };
        let r_c = if self.n_ctrl > 0.0 { self.good_ctrl / self.n_ctrl } else { 0.0 };
        (r_t, r_c)
    }
    #[allow(dead_code)]
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
    _avg_stop: f64,
    _median_signals: f64,
    trajectories: Vec<Vec<f64>>,
    // Type M
    _avg_effect_at_stop: f64,
    _avg_effect_at_final: f64,
    _type_m: f64,
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
        _avg_stop: avg_stop,
        _median_signals: median_signals,
        trajectories,
        _avg_effect_at_stop: avg_effect_at_stop,
        _avg_effect_at_final: avg_effect_at_final,
        _type_m: type_m,
    }
}

// === HTML REPORT ===

fn build_html(
    _n_signals: usize,
    _n_sims: usize,
    threshold: f64,
    _p_trt: f64,
    _p_ctrl: f64,
    null: &SimResults,
    alt: &SimResults,
) -> String {
    format!(
        r#"<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>e-RTu Universal Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
body{{font-family:system-ui,-apple-system,sans-serif;max-width:1400px;margin:0 auto;padding:20px;background:#fafafa}}
h1{{color:#1a1a2e}}h2,h3{{color:#16213e}}
pre{{background:#fff;padding:15px;border-radius:8px;border:1px solid #ddd;overflow-x:auto;font-size:13px}}
.plot-container{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin:20px 0}}
.plot{{background:#fff;border-radius:8px;padding:10px;box-shadow:0 1px 3px rgba(0,0,0,0.1)}}
</style></head><body>
<h1>e-RTu Universal Report</h1>
<h2>Console Output</h2>
<pre>{}
Type I: {:.2}%  |  Power: {:.1}%</pre>
<h2>Visualizations</h2>
<div class="plot-container">
<div class="plot"><div id="p1" style="height:350px"></div></div>
<div class="plot"><div id="p2" style="height:350px"></div></div>
</div>
<script>
var t_null={:?};var t_alt={:?};var threshold={};
Plotly.newPlot('p1',t_null.slice(0,30).map((y,i)=>({{type:'scatter',y:y,line:{{color:'rgba(150,150,150,0.4)'}},showlegend:false}})),{{
  title:'Null Hypothesis',yaxis:{{type:'log',title:'e-value',range:[-1,Math.log10(threshold)+1]}},xaxis:{{title:'Signal'}},
  shapes:[{{type:'line',x0:0,x1:1,xref:'paper',y0:threshold,y1:threshold,line:{{color:'green',dash:'dash',width:2}}}}]}});
Plotly.newPlot('p2',t_alt.slice(0,30).map((y,i)=>({{type:'scatter',y:y,line:{{color:'rgba(155,89,182,0.5)'}},showlegend:false}})),{{
  title:'Alternative Hypothesis',yaxis:{{type:'log',title:'e-value',range:[-1,Math.log10(threshold)+1]}},xaxis:{{title:'Signal'}},
  shapes:[{{type:'line',x0:0,x1:1,xref:'paper',y0:threshold,y1:threshold,line:{{color:'green',dash:'dash',width:2}}}}]}});
</script></body></html>"#,
        chrono_lite(), null.rejection_rate * 100.0, alt.rejection_rate * 100.0,
        null.trajectories, alt.trajectories, threshold,
    )
}

// === MAIN ===

pub fn run() {
    println!("\n==========================================");
    println!("   e-RTu (Universal)");
    println!("==========================================\n");

    println!("The universal e-process. Sees only: (arm, good/bad)");
    println!("Domain-agnostic. Works on any signal source.\n");

    println!("Demo mode: Binary signal generator");
    println!("------------------------------------------\n");

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
    println!("Effect at stop:  {:.3}", alt._avg_effect_at_stop);
    println!("Effect at final: {:.3}", alt._avg_effect_at_final);
    println!("Type M ratio:    {:.2}x", alt._type_m);

    let html = build_html(n_signals, n_sims, threshold, p_trt, p_ctrl, &null, &alt);
    File::create("agnostic_report.html")
        .unwrap()
        .write_all(html.as_bytes())
        .unwrap();
    println!("\n>> Saved: agnostic_report.html");
}
