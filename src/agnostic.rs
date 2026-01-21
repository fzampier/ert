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

impl Signal {
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
    signal_count: usize,

    // Parameters
    burn_in: usize,
    ramp: usize,
    threshold: f64,

    // Result
    stopped: bool,
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
            signal_count: 0,
            burn_in,
            ramp,
            threshold,
            stopped: false,
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
            base.clamp(0.001, 0.999)
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

        // Check threshold
        if self.wealth >= self.threshold {
            self.stopped = true;
        }

        self.stopped
    }

    pub fn stopped(&self) -> bool { self.stopped }
    pub fn wealth(&self) -> f64 { self.wealth }
    pub fn history(&self) -> &[f64] { &self.wealth_history }
}

// === SIMULATION RESULTS ===

struct SimResults {
    rejection_rate: f64,
    pos_trajectories: Vec<Vec<f64>>,  // Crossed threshold
    neg_trajectories: Vec<Vec<f64>>,  // Did not cross
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
    let mut pos_trajectories: Vec<Vec<f64>> = Vec::new();
    let mut neg_trajectories: Vec<Vec<f64>> = Vec::new();

    for sim in 0..n_sims {
        let mut ert = AgnosticERT::new(burn_in, ramp, threshold);

        for _ in 0..n_signals {
            let signal = generate_binary_signal(rng, p_trt, p_ctrl);
            if ert.observe(signal) {
                break;
            }
        }

        let stopped = ert.stopped();
        if stopped {
            rejections += 1;
        }

        // Collect positive/negative trajectories separately
        let need_traj = pos_trajectories.len() < 30 || neg_trajectories.len() < 30;
        if need_traj {
            if stopped {
                if pos_trajectories.len() < 30 { pos_trajectories.push(ert.history().to_vec()); }
            } else if neg_trajectories.len() < 30 {
                neg_trajectories.push(ert.history().to_vec());
            }
        }

        if (sim + 1) % 100 == 0 {
            print!("\rSimulation {}/{}", sim + 1, n_sims);
            io::stdout().flush().unwrap();
        }
    }
    println!();

    SimResults {
        rejection_rate: rejections as f64 / n_sims as f64,
        pos_trajectories,
        neg_trajectories,
    }
}

// === HTML REPORT ===

fn build_html(threshold: f64, null: &SimResults, alt: &SimResults) -> String {
    // Build representative samples: proportion of positive samples matches rate
    let null_rate = null.rejection_rate;
    let n_pos_null = ((null_rate) * 30.0).round() as usize;
    let n_neg_null = 30 - n_pos_null;
    let mut null_trajs: Vec<Vec<f64>> = Vec::new();
    null_trajs.extend(null.pos_trajectories.iter().take(n_pos_null).cloned());
    null_trajs.extend(null.neg_trajectories.iter().take(n_neg_null).cloned());

    let alt_rate = alt.rejection_rate;
    let n_pos_alt = ((alt_rate) * 30.0).round() as usize;
    let n_neg_alt = 30 - n_pos_alt;
    let mut alt_trajs: Vec<Vec<f64>> = Vec::new();
    alt_trajs.extend(alt.pos_trajectories.iter().take(n_pos_alt).cloned());
    alt_trajs.extend(alt.neg_trajectories.iter().take(n_neg_alt).cloned());

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
        null_trajs, alt_trajs, threshold,
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

    let html = build_html(threshold, &null, &alt);
    File::create("agnostic_report.html")
        .unwrap()
        .write_all(html.as_bytes())
        .unwrap();
    println!("\n>> Saved: agnostic_report.html");
}
