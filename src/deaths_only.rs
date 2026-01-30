// deaths_only.rs - e-RTd: Deaths-Only Sequential Monitoring
//
// A simplified e-process that monitors only death events, without requiring
// survivor data or denominator tracking. Based on the insight that under 1:1
// randomization, P(death from treatment | death occurred) = 0.5 under null.
//
// Key advantage: No day-28 survival reporting needed - just log deaths.
// Trade-off: ~2.5x sample size inflation vs frequentist power.

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::RngCore;
use std::fs::File;
use std::io::{self, Write};

use crate::ert_core::{get_input, get_input_usize, get_optional_input, get_bool, chrono_lite};

// === CORE E-PROCESS ===

#[derive(Clone, Copy, PartialEq)]
pub enum Arm {
    Treatment,
    Control,
}

/// Deaths-only e-process for sequential monitoring.
///
/// Bets on the proportion of deaths from each arm.
/// Under null (equal mortality): p = 0.5
/// Under alternative (treatment helps): p < 0.5
pub struct DeathsOnlyERT {
    // Death counts
    d_trt: u64,
    d_ctrl: u64,

    // State
    wealth: f64,
    wealth_history: Vec<f64>,
    death_count: usize,

    // Parameters
    burn_in: usize,
    ramp: usize,
    threshold: f64,

    // Result
    stopped: bool,
    stop_time: Option<usize>,
}

impl DeathsOnlyERT {
    pub fn new(burn_in: usize, ramp: usize, threshold: f64) -> Self {
        DeathsOnlyERT {
            d_trt: 0,
            d_ctrl: 0,
            wealth: 1.0,
            wealth_history: vec![1.0],
            death_count: 0,
            burn_in,
            ramp,
            threshold,
            stopped: false,
            stop_time: None,
        }
    }

    /// Process a death event. Returns true if threshold has been crossed (ever).
    /// Continues tracking wealth even after crossing for visualization.
    pub fn observe(&mut self, arm: Arm) -> bool {
        self.death_count += 1;

        // Compute p_obs from PREVIOUS deaths (before updating counts)
        let total = self.d_trt + self.d_ctrl;
        let p_obs = if total > 0 {
            self.d_trt as f64 / total as f64
        } else {
            0.5
        };

        // Compute lambda (wager)
        let lambda = if self.death_count > self.burn_in && total > 0 {
            let c_i = ((self.death_count - self.burn_in) as f64 / self.ramp as f64).clamp(0.0, 1.0);
            (0.5 + c_i * (p_obs - 0.5)).clamp(0.001, 0.999)
        } else {
            0.5 // Neutral bet during burn-in
        };

        // Update wealth
        let mult = match arm {
            Arm::Treatment => lambda / 0.5,
            Arm::Control => (1.0 - lambda) / 0.5,
        };
        self.wealth *= mult;

        // Update counts AFTER betting
        match arm {
            Arm::Treatment => self.d_trt += 1,
            Arm::Control => self.d_ctrl += 1,
        }

        self.wealth_history.push(self.wealth);

        // Check threshold (only record FIRST crossing)
        if !self.stopped && self.wealth >= self.threshold {
            self.stopped = true;
            self.stop_time = Some(self.death_count);
        }

        self.stopped
    }

    pub fn wealth(&self) -> f64 { self.wealth }
    #[allow(dead_code)]
    pub fn stopped(&self) -> bool { self.stopped }
    #[allow(dead_code)]
    pub fn stop_time(&self) -> Option<usize> { self.stop_time }
    pub fn history(&self) -> &[f64] { &self.wealth_history }
    #[allow(dead_code)]
    pub fn death_count(&self) -> usize { self.death_count }

    /// Observed proportion of deaths from treatment arm
    pub fn death_proportion_trt(&self) -> f64 {
        let total = self.d_trt + self.d_ctrl;
        if total > 0 {
            self.d_trt as f64 / total as f64
        } else {
            0.5
        }
    }

    /// Mortality rate ratio: RR = p / (1 - p)
    /// Where p = P(death from treatment | death)
    pub fn mortality_rate_ratio(&self) -> f64 {
        let p = self.death_proportion_trt();
        if p >= 0.999 { return f64::INFINITY; }
        if p <= 0.001 { return 0.0; }
        p / (1.0 - p)
    }

    /// Anytime-valid confidence sequence for p = P(death from trt | death)
    pub fn confidence_sequence_p(&self, alpha: f64) -> (f64, f64) {
        let n = (self.d_trt + self.d_ctrl) as f64;
        if n < 2.0 {
            return (0.0, 1.0);
        }

        let p = self.death_proportion_trt();

        // Time-uniform critical value (Robbins mixture)
        let log_factor = (2.0 / alpha).ln() + n.ln().ln().max(0.0);
        let crit = (2.0 * log_factor).sqrt();

        let se = (p * (1.0 - p) / n).sqrt();
        let margin = crit * se;

        ((p - margin).clamp(0.0, 1.0), (p + margin).clamp(0.0, 1.0))
    }

    /// Anytime-valid confidence sequence for mortality rate ratio
    /// RR = p / (1 - p), where p = P(death from trt | death)
    pub fn confidence_sequence_rr(&self, alpha: f64) -> (f64, f64) {
        let (p_lo, p_hi) = self.confidence_sequence_p(alpha);

        // Clamp away from 0 and 1 to avoid infinity
        let p_lo = p_lo.max(0.001);
        let p_hi = p_hi.min(0.999);

        (p_lo / (1.0 - p_lo), p_hi / (1.0 - p_hi))
    }

    pub fn get_counts(&self) -> (u64, u64) {
        (self.d_trt, self.d_ctrl)
    }
}

// === SAMPLE SIZE CALCULATION ===

/// Calculate frequentist sample size for two-proportion z-test
fn frequentist_sample_size(p_ctrl: f64, p_trt: f64, power: f64) -> usize {
    let z_alpha: f64 = 1.96; // two-sided alpha=0.05
    let z_beta: f64 = if power > 0.85 { 1.28 } else { 0.84 };

    let p_bar = (p_ctrl + p_trt) / 2.0;
    let delta = (p_ctrl - p_trt).abs();

    let term1 = 4.0 * (z_alpha + z_beta).powi(2);
    let term2 = p_bar * (1.0 - p_bar);
    let term3 = delta.powi(2);

    ((term1 * term2) / term3).ceil() as usize
}

/// Calculate required sample size for e-RTd to achieve target power.
///
/// e-RTd requires ~2.5x the frequentist sample size for equivalent power.
/// This is the cost of not tracking survivors.
pub fn calculate_n_deaths_only(p_ctrl: f64, p_trt: f64, power: f64) -> (usize, usize, usize) {
    let n_freq = frequentist_sample_size(p_ctrl, p_trt, power);

    // e-RTd inflation factor: ~2.5x based on simulation studies
    let inflation = 2.5;
    let n_ertd = (n_freq as f64 * inflation).ceil() as usize;

    // Expected deaths
    let expected_deaths = (n_ertd as f64 / 2.0 * (p_ctrl + p_trt)).ceil() as usize;

    (n_freq, n_ertd, expected_deaths)
}

// === SIMULATION ===

struct SimResults {
    rejection_rate: f64,
    trajectories: Vec<Vec<f64>>,  // All trajectories run to completion
    median_stop_time: Option<usize>,
}

/// Generate a stream of deaths for simulation.
/// p_trt_given_death = P(death from treatment | death occurred)
fn simulate_deaths<R: Rng + ?Sized>(
    rng: &mut R,
    n_deaths: usize,
    p_trt_given_death: f64,
) -> Vec<Arm> {
    (0..n_deaths)
        .map(|_| {
            if rng.gen_bool(p_trt_given_death) {
                Arm::Treatment
            } else {
                Arm::Control
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn run_simulation<R: Rng + ?Sized>(
    rng: &mut R,
    n_deaths: usize,
    n_sims: usize,
    p_trt_given_death: f64,
    burn_in: usize,
    ramp: usize,
    threshold: f64,
) -> SimResults {
    let mut rejections = 0;
    let mut trajectories: Vec<Vec<f64>> = Vec::new();
    let mut stop_times: Vec<usize> = Vec::new();

    for sim in 0..n_sims {
        let deaths = simulate_deaths(rng, n_deaths, p_trt_given_death);
        let mut ert = DeathsOnlyERT::new(burn_in, ramp, threshold);
        let mut first_cross: Option<usize> = None;

        // Run simulation to the END (don't stop at threshold)
        for (i, arm) in deaths.iter().enumerate() {
            ert.observe(*arm);

            // Track first crossing
            if first_cross.is_none() && ert.wealth() >= threshold {
                first_cross = Some(i + 1);
            }
        }

        if first_cross.is_some() {
            rejections += 1;
            stop_times.push(first_cross.unwrap());
        }

        // Collect trajectories (up to 30 for visualization)
        if trajectories.len() < 30 {
            trajectories.push(ert.history().to_vec());
        }

        if (sim + 1) % 100 == 0 {
            print!("\rSimulation {}/{}", sim + 1, n_sims);
            io::stdout().flush().unwrap();
        }
    }
    println!();

    // Compute median stop time
    let median_stop_time = if !stop_times.is_empty() {
        stop_times.sort();
        Some(stop_times[stop_times.len() / 2])
    } else {
        None
    };

    SimResults {
        rejection_rate: rejections as f64 / n_sims as f64,
        trajectories,
        median_stop_time,
    }
}

// === HTML REPORT ===

fn build_html(
    threshold: f64,
    null: &SimResults,
    alt: &SimResults,
    p_ctrl: f64,
    p_trt: f64,
    n_freq: usize,
    n_ertd: usize,
) -> String {
    // Use all collected trajectories (now run to completion)
    let null_trajs = &null.trajectories;
    let alt_trajs = &alt.trajectories;

    // Compute p values for display
    let _p_null = 0.5;
    let p_alt = p_trt / (p_trt + p_ctrl);

    format!(
        r#"<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>e-RTd Deaths-Only Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
body{{font-family:system-ui,-apple-system,sans-serif;max-width:1400px;margin:0 auto;padding:20px;background:#fafafa}}
h1{{color:#1a1a2e}}h2,h3{{color:#16213e}}
pre{{background:#fff;padding:15px;border-radius:8px;border:1px solid #ddd;overflow-x:auto;font-size:13px}}
.plot-container{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin:20px 0}}
.plot{{background:#fff;border-radius:8px;padding:10px;box-shadow:0 1px 3px rgba(0,0,0,0.1)}}
.note{{font-size:0.9em;color:#666;margin-top:10px}}
</style></head><body>
<h1>e-RTd Deaths-Only Report</h1>
<h2>Summary</h2>
<pre>{}

Mortality rates: Control {:.1}%, Treatment {:.1}%
ARR: {:.1}%

Sample size for 80% power:
  Frequentist z-test: N = {}
  e-RTd (deaths-only): N = {} (~2.5x inflation)

P(death from trt | death): Null={:.3}, Alt={:.3}

Type I: {:.2}%  |  Power: {:.1}%{}
</pre>
<p class="note">e-RTd bets only on death events. No survivor tracking required.</p>

<h2>Visualizations</h2>
<div class="plot-container">
<div class="plot"><div id="p1" style="height:350px"></div></div>
<div class="plot"><div id="p2" style="height:350px"></div></div>
</div>
<script>
var t_null={:?};var t_alt={:?};var threshold={};
Plotly.newPlot('p1',t_null.slice(0,30).map((y,i)=>({{type:'scatter',y:y,line:{{color:'rgba(150,150,150,0.4)'}},showlegend:false}})),{{
  title:'Null Hypothesis (p=0.5)',yaxis:{{type:'log',title:'e-value',range:[-1,Math.log10(threshold)+1]}},xaxis:{{title:'Deaths'}},
  shapes:[{{type:'line',x0:0,x1:1,xref:'paper',y0:threshold,y1:threshold,line:{{color:'green',dash:'dash',width:2}}}}]}});
Plotly.newPlot('p2',t_alt.slice(0,30).map((y,i)=>({{type:'scatter',y:y,line:{{color:'rgba(155,89,182,0.5)'}},showlegend:false}})),{{
  title:'Alternative Hypothesis (p={:.3})',yaxis:{{type:'log',title:'e-value',range:[-1,Math.log10(threshold)+1]}},xaxis:{{title:'Deaths'}},
  shapes:[{{type:'line',x0:0,x1:1,xref:'paper',y0:threshold,y1:threshold,line:{{color:'green',dash:'dash',width:2}}}}]}});
</script></body></html>"#,
        chrono_lite(),
        p_ctrl * 100.0, p_trt * 100.0,
        (p_ctrl - p_trt) * 100.0,
        n_freq, n_ertd,
        0.5, p_alt,
        null.rejection_rate * 100.0, alt.rejection_rate * 100.0,
        alt.median_stop_time.map_or(String::new(), |t| format!("  |  Median stop: {} deaths", t)),
        null_trajs, alt_trajs, threshold, p_alt,
    )
}

// === MAIN INTERACTIVE ===

pub fn run() {
    println!("\n==========================================");
    println!("   e-RTd (Deaths Only)");
    println!("==========================================\n");

    println!("Deaths-only e-process. Bets on which arm each death came from.");
    println!("No survivor tracking required - just log deaths.\n");

    println!("Trade-off: ~2.5x sample size vs frequentist power.");
    println!("Best for: Large pragmatic trials, early signal detection.\n");

    println!("Enter mortality rates:");
    let p_ctrl = get_input("Control mortality (e.g., 0.30): ");
    let p_trt = get_input("Treatment mortality (e.g., 0.25): ");

    // p = P(death from treatment | death)
    let p_alt = p_trt / (p_trt + p_ctrl);

    // Sample size calculation (matching binary.rs pattern)
    let (total_patients, n_freq, target_power) = if get_bool("\nCalculate Sample Size automatically?") {
        let power: f64 = get_input("Target Power (e.g. 0.80): ");
        let power = power.min(0.99);
        let (n_freq, _, _) = calculate_n_deaths_only(p_ctrl, p_trt, power);
        println!("\nFrequentist N (Power {:.0}%): {}", power * 100.0, n_freq);

        let final_n = if get_bool("Add e-RTd correction? (recommended: 150% for ~2.5x total)") {
            let buf: f64 = get_input("Buffer percentage (e.g. 150): ");
            let buffered = (n_freq as f64 * (1.0 + buf / 100.0)).ceil() as usize;
            println!("Corrected N: {} ({:.1}x frequentist)", buffered, buffered as f64 / n_freq as f64);
            buffered
        } else {
            println!("Note: Without correction, expect ~40% of frequentist power");
            n_freq
        };
        (final_n, n_freq, Some(power))
    } else {
        let n = get_input_usize("Enter Number of Patients: ");
        let (n_freq, _, _) = calculate_n_deaths_only(p_ctrl, p_trt, 0.80);
        (n, n_freq, None)
    };

    let expected_deaths = (total_patients as f64 / 2.0 * (p_ctrl + p_trt)).ceil() as usize;

    println!("\nP(death from trt | death): {:.3} (null: 0.500)", p_alt);
    println!("Expected deaths: ~{}", expected_deaths);

    let n_sims = get_input_usize("\nSimulations (e.g., 1000): ");
    let threshold = get_input("Threshold (default 20): ");
    let seed = get_optional_input("Seed (Enter for random): ");

    let burn_in: usize = 30;
    let ramp: usize = 50;

    let mut rng: Box<dyn RngCore> = match seed {
        Some(s) => Box::new(StdRng::seed_from_u64(s)),
        None => Box::new(rand::thread_rng()),
    };

    println!("\n--- Null (both arms = control mortality) ---");
    let null = run_simulation(
        &mut *rng,
        expected_deaths,
        n_sims,
        0.5, // p = 0.5 under null
        burn_in,
        ramp,
        threshold,
    );
    println!("Type I Error: {:.2}%", null.rejection_rate * 100.0);

    println!("\n--- Alternative (treatment reduces mortality) ---");
    let alt = run_simulation(
        &mut *rng,
        expected_deaths,
        n_sims,
        p_alt,
        burn_in,
        ramp,
        threshold,
    );
    println!("Power: {:.1}%", alt.rejection_rate * 100.0);
    if let Some(t) = alt.median_stop_time {
        println!("Median stopping time: {} deaths", t);
    }

    // Summary
    println!("\n==========================================");
    println!("   SUMMARY");
    println!("==========================================");
    println!("Patients: {}", total_patients);
    println!("Expected deaths: ~{}", expected_deaths);
    println!("Mortality: Control {:.1}%, Treatment {:.1}%", p_ctrl * 100.0, p_trt * 100.0);
    println!("ARR: {:.1}%", (p_ctrl - p_trt) * 100.0);
    if let Some(p) = target_power {
        println!("Target power: {:.0}%", p * 100.0);
    }
    println!("Frequentist N: {}", n_freq);
    println!("Type I: {:.2}%  |  Power: {:.1}%", null.rejection_rate * 100.0, alt.rejection_rate * 100.0);

    let html = build_html(threshold, &null, &alt, p_ctrl, p_trt, n_freq, total_patients);
    File::create("deaths_only_report.html")
        .unwrap()
        .write_all(html.as_bytes())
        .unwrap();
    println!("\n>> Saved: deaths_only_report.html");
}
