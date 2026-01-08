use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::RngCore;
use std::io::{self, Write};
use std::fs::File;

use crate::ert_core::{
    get_input, get_input_usize, get_bool, get_optional_input, get_choice,
    median, mad, calculate_n_continuous, chrono_lite, t_test_power_continuous,
};
use crate::agnostic::{AgnosticERT, Signal, Arm};

// === STRUCTS ===

#[derive(Clone, Copy, PartialEq)]
enum Method { LinearERT, MAD }

#[derive(Clone, Copy, PartialEq)]
enum Distribution { Uniform, Normal, VFDMixture }

#[derive(Clone, Copy)]
struct VFDParams {
    p_death_ctrl: f64,
    p_death_trt: f64,
    survivor_mean_ctrl: f64,  // mean VFD among survivors (e.g., 18)
    survivor_mean_trt: f64,   // mean VFD among survivors (e.g., 21)
    max_val: f64,             // typically 28
}

struct Trial {
    stop_n: Option<usize>,
    effect_at_stop: f64,
    effect_final: f64,
    min_wealth: f64,
    t_significant: bool,
    agnostic_stopped: bool,
}

// === LinearERT PROCESS ===

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
        Self { wealth: 1.0, burn_in, ramp, min_val, max_val, sum_trt: 0.0, n_trt: 0.0, sum_ctrl: 0.0, n_ctrl: 0.0 }
    }

    fn update(&mut self, i: usize, outcome: f64, is_trt: bool) {
        let mean_trt = if self.n_trt > 0.0 { self.sum_trt / self.n_trt } else { (self.min_val + self.max_val) / 2.0 };
        let mean_ctrl = if self.n_ctrl > 0.0 { self.sum_ctrl / self.n_ctrl } else { (self.min_val + self.max_val) / 2.0 };
        let delta_hat = mean_trt - mean_ctrl;

        if is_trt { self.n_trt += 1.0; self.sum_trt += outcome; }
        else { self.n_ctrl += 1.0; self.sum_ctrl += outcome; }

        if i > self.burn_in {
            let c_i = (((i - self.burn_in) as f64) / self.ramp as f64).clamp(0.0, 1.0);
            let x = (outcome - self.min_val) / (self.max_val - self.min_val);
            let scalar = 2.0 * x - 1.0;
            let delta_norm = delta_hat / (self.max_val - self.min_val);
            let lambda = (0.5 + 0.5 * c_i * delta_norm * scalar).clamp(0.001, 0.999);
            self.wealth *= if is_trt { lambda / 0.5 } else { (1.0 - lambda) / 0.5 };
        }
    }

    fn current_effect(&self) -> f64 {
        let mean_trt = if self.n_trt > 0.0 { self.sum_trt / self.n_trt } else { 0.0 };
        let mean_ctrl = if self.n_ctrl > 0.0 { self.sum_ctrl / self.n_ctrl } else { 0.0 };
        mean_trt - mean_ctrl
    }
}

// === MAD-based PROCESS ===

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
        Self { wealth: 1.0, burn_in, ramp, c_max, outcomes: Vec::new(), treatments: Vec::new() }
    }

    fn update(&mut self, i: usize, outcome: f64, is_trt: bool) {
        // Continuous direction: standardized effect estimate (not binary!)
        let direction = if !self.outcomes.is_empty() {
            let (mut sum_t, mut ss_t, mut n_t) = (0.0, 0.0, 0.0);
            let (mut sum_c, mut ss_c, mut n_c) = (0.0, 0.0, 0.0);
            for (o, &t) in self.outcomes.iter().zip(self.treatments.iter()) {
                if t { sum_t += o; ss_t += o * o; n_t += 1.0; }
                else { sum_c += o; ss_c += o * o; n_c += 1.0; }
            }
            if n_t > 1.0 && n_c > 1.0 {
                let m_t = sum_t / n_t;
                let m_c = sum_c / n_c;
                let var_t = (ss_t - sum_t * sum_t / n_t) / (n_t - 1.0);
                let var_c = (ss_c - sum_c * sum_c / n_c) / (n_c - 1.0);
                let pooled_sd = ((var_t + var_c) / 2.0).sqrt().max(0.001);
                let delta = (m_t - m_c) / pooled_sd;  // standardized effect
                delta.clamp(-1.0, 1.0)  // bounded continuous direction
            } else { 0.0 }
        } else { 0.0 };

        self.outcomes.push(outcome);
        self.treatments.push(is_trt);

        if i > self.burn_in && self.outcomes.len() > 1 {
            let past: Vec<f64> = self.outcomes[..self.outcomes.len()-1].to_vec();
            let med = median(&past);
            let mad_val = mad(&past);
            let s = if mad_val > 0.0 { mad_val } else { 1.0 };
            let r = (outcome - med) / s;
            let g = r / (1.0 + r.abs());
            let c_i = (((i - self.burn_in) as f64) / self.ramp as f64).clamp(0.0, 1.0);
            let lambda = (0.5 + c_i * self.c_max * g * direction).clamp(0.001, 0.999);
            self.wealth *= if is_trt { lambda / 0.5 } else { (1.0 - lambda) / 0.5 };
        }
    }

    fn current_effect(&self, sd: f64) -> f64 {
        let (mut sum_t, mut n_t, mut sum_c, mut n_c) = (0.0, 0.0, 0.0, 0.0);
        for (o, &t) in self.outcomes.iter().zip(self.treatments.iter()) {
            if t { sum_t += o; n_t += 1.0; } else { sum_c += o; n_c += 1.0; }
        }
        let m_t = if n_t > 0.0 { sum_t / n_t } else { 0.0 };
        let m_c = if n_c > 0.0 { sum_c / n_c } else { 0.0 };
        (m_t - m_c) / sd
    }
}

// === T-TEST ===

fn t_test_significant(outcomes: &[(f64, bool)], alpha: f64) -> bool {
    let (mut sum_t, mut ss_t, mut n_t) = (0.0, 0.0, 0.0);
    let (mut sum_c, mut ss_c, mut n_c) = (0.0, 0.0, 0.0);
    for &(o, is_trt) in outcomes {
        if is_trt { sum_t += o; ss_t += o * o; n_t += 1.0; }
        else { sum_c += o; ss_c += o * o; n_c += 1.0; }
    }
    if n_t < 2.0 || n_c < 2.0 { return false; }
    let mean_t = sum_t / n_t;
    let mean_c = sum_c / n_c;
    let var_t = (ss_t - sum_t * sum_t / n_t) / (n_t - 1.0);
    let var_c = (ss_c - sum_c * sum_c / n_c) / (n_c - 1.0);
    let se = (var_t / n_t + var_c / n_c).sqrt();
    if se < 1e-10 { return false; }
    let t = (mean_t - mean_c).abs() / se;
    let df = n_t + n_c - 2.0;
    let p = 2.0 * (1.0 - t_cdf(t, df));
    p < alpha
}

fn t_cdf(t: f64, df: f64) -> f64 {
    let x = df / (df + t * t);
    1.0 - 0.5 * incomplete_beta(df / 2.0, 0.5, x)
}

fn incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    if x >= 1.0 { return 1.0; }
    let mut sum = 0.0;
    let mut term = 1.0;
    for n in 0..200 {
        if n > 0 { term *= (n as f64 - b) * x / n as f64; }
        let add = term / (a + n as f64);
        sum += add;
        if add.abs() < 1e-10 { break; }
    }
    x.powf(a) * (1.0 - x).powf(b) * sum / beta(a, b)
}

fn beta(a: f64, b: f64) -> f64 {
    (gamma_ln(a) + gamma_ln(b) - gamma_ln(a + b)).exp()
}

fn gamma_ln(x: f64) -> f64 {
    let c = [76.18009172947146, -86.50532032941677, 24.01409824083091,
             -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
    let mut y = x;
    let mut tmp = x + 5.5;
    tmp -= (x + 0.5) * tmp.ln();
    let mut ser = 1.000000000190015;
    for &cj in &c { y += 1.0; ser += cj / y; }
    -tmp + (2.5066282746310005 * ser / x).ln()
}

// === VFD MIXTURE SAMPLING ===

/// Sample from Beta(alpha, beta) using rejection sampling
fn sample_beta<R: Rng + ?Sized>(rng: &mut R, alpha: f64, beta: f64) -> f64 {
    // Use Gamma sampling method: X ~ Gamma(alpha), Y ~ Gamma(beta), then X/(X+Y) ~ Beta(alpha, beta)
    let x = sample_gamma(rng, alpha);
    let y = sample_gamma(rng, beta);
    if x + y > 0.0 { x / (x + y) } else { 0.5 }
}

/// Sample from Gamma(shape, 1) using Marsaglia and Tsang's method
fn sample_gamma<R: Rng + ?Sized>(rng: &mut R, shape: f64) -> f64 {
    if shape < 1.0 {
        // For shape < 1, use: Gamma(shape) = Gamma(shape+1) * U^(1/shape)
        let u: f64 = rng.gen();
        sample_gamma(rng, shape + 1.0) * u.powf(1.0 / shape)
    } else {
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x: f64 = sample_standard_normal(rng);
            let v = (1.0 + c * x).powi(3);
            if v > 0.0 {
                let u: f64 = rng.gen();
                if u < 1.0 - 0.0331 * x.powi(4) || u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                    return d * v;
                }
            }
        }
    }
}

/// Box-Muller transform for standard normal
fn sample_standard_normal<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    let u1: f64 = rng.gen();
    let u2: f64 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Sample from VFD mixture model
fn sample_vfd<R: Rng + ?Sized>(rng: &mut R, p_death: f64, survivor_mean: f64, max_val: f64) -> f64 {
    if rng.gen::<f64>() < p_death {
        0.0  // Death
    } else {
        // Survivor: Beta distribution scaled to [1, max_val]
        // Map survivor_mean (on 1..max_val) to Beta mean (on 0..1)
        let beta_mean = (survivor_mean - 1.0) / (max_val - 1.0);
        let beta_mean = beta_mean.clamp(0.05, 0.95);  // Avoid extreme parameters

        // Use concentration parameter to control variance
        // Higher concentration = tighter distribution around mean
        let concentration = 8.0;  // Reasonable spread
        let alpha = beta_mean * concentration;
        let beta = (1.0 - beta_mean) * concentration;

        // Sample and scale to [1, max_val]
        let x = sample_beta(rng, alpha, beta);
        1.0 + x * (max_val - 1.0)
    }
}

/// Calculate mean and SD for VFD mixture (for sample size calculation)
fn vfd_mixture_moments(p_death: f64, survivor_mean: f64, max_val: f64) -> (f64, f64) {
    // E[X] = (1 - p_death) * survivor_mean + p_death * 0
    let mean = (1.0 - p_death) * survivor_mean;

    // Var[X] = E[X^2] - E[X]^2
    // E[X^2] = (1-p_death) * E[X^2 | survivor] + p_death * 0
    // For Beta on [1, max_val]: E[X^2|survivor] ≈ survivor_mean^2 + variance_survivor
    let beta_mean = (survivor_mean - 1.0) / (max_val - 1.0);
    let concentration = 8.0;
    let alpha = beta_mean.clamp(0.05, 0.95) * concentration;
    let beta_param = (1.0 - beta_mean.clamp(0.05, 0.95)) * concentration;
    let beta_var = (alpha * beta_param) / ((alpha + beta_param).powi(2) * (alpha + beta_param + 1.0));
    let survivor_var = beta_var * (max_val - 1.0).powi(2);

    // Total variance using law of total variance
    let e_x2_survivor = survivor_mean.powi(2) + survivor_var;
    let e_x2 = (1.0 - p_death) * e_x2_survivor;
    let var = e_x2 - mean.powi(2) + p_death * (1.0 - p_death) * survivor_mean.powi(2);

    (mean, var.sqrt())
}

// === SIMULATION ===

fn run_simulation<R: Rng + ?Sized>(
    rng: &mut R, method: Method, n_pts: usize, n_sims: usize,
    mu_ctrl: f64, mu_trt: f64, sd: f64, min_val: f64, max_val: f64,
    _design_effect: f64, threshold: f64, _fut_watch: f64,
    burn_in: usize, ramp: usize, c_max: f64,
    distribution: Distribution, vfd_params: Option<VFDParams>,
) -> (Vec<Trial>, Vec<Vec<f64>>, Vec<usize>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let method_name = if method == Method::LinearERT { "e-RTo" } else { "e-RTc" };
    let alpha = 1.0 / threshold;

    // Sample indices for trajectories
    let sample_indices: Vec<usize> = (0..30.min(n_sims)).collect();
    let step = (n_pts / 100).max(1);
    let steps: Vec<usize> = (1..=n_pts).filter(|&i| i % step == 0 || i == n_pts).collect();

    let mut trials: Vec<Trial> = Vec::with_capacity(n_sims);
    let mut sample_trajs: Vec<Vec<f64>> = vec![Vec::with_capacity(steps.len()); sample_indices.len()];
    let mut y_lo: Vec<f64> = vec![0.0; steps.len()];
    let mut y_med: Vec<f64> = vec![0.0; steps.len()];
    let mut y_hi: Vec<f64> = vec![0.0; steps.len()];

    // Running percentile buffers
    let mut step_vals: Vec<Vec<f64>> = vec![Vec::with_capacity(n_sims); steps.len()];

    print!("  {} Power + Futility... ", method_name);
    io::stdout().flush().unwrap();
    let pb_interval = (n_sims / 20).max(1);

    for sim in 0..n_sims {
        if sim % pb_interval == 0 { print!("."); io::stdout().flush().unwrap(); }

        let mut stop_n: Option<usize> = None;
        let mut effect_at_stop = 0.0;
        let mut min_wealth = 1.0f64;
        let mut outcomes: Vec<(f64, bool)> = Vec::with_capacity(n_pts);
        let mut all_outcomes: Vec<f64> = Vec::with_capacity(n_pts); // for running median

        // Agnostic tracker
        let mut agnostic = AgnosticERT::new(burn_in, ramp, threshold);
        let mut agnostic_stopped = false;

        let is_sample = sample_indices.contains(&sim);
        let mut step_idx = 0;

        if method == Method::LinearERT {
            let mut proc = LinearERTProcess::new(burn_in, ramp, min_val, max_val);

            for i in 1..=n_pts {
                let is_trt = rng.gen_bool(0.5);

                // Sample outcome based on distribution
                let outcome = match distribution {
                    Distribution::VFDMixture => {
                        let vfd = vfd_params.unwrap();
                        let (p_death, surv_mean) = if is_trt {
                            (vfd.p_death_trt, vfd.survivor_mean_trt)
                        } else {
                            (vfd.p_death_ctrl, vfd.survivor_mean_ctrl)
                        };
                        sample_vfd(rng, p_death, surv_mean, vfd.max_val)
                    }
                    Distribution::Normal => {
                        let mu = if is_trt { mu_trt } else { mu_ctrl };
                        let z = sample_standard_normal(rng);
                        (mu + z * sd).clamp(min_val, max_val)
                    }
                    Distribution::Uniform => {
                        let mu = if is_trt { mu_trt } else { mu_ctrl };
                        ((rng.gen::<f64>() * 2.0 - 1.0) * sd * 1.5 + mu).clamp(min_val, max_val)
                    }
                };
                outcomes.push((outcome, is_trt));

                proc.update(i, outcome, is_trt);
                min_wealth = min_wealth.min(proc.wealth);
                all_outcomes.push(outcome);

                // Agnostic with running median
                if !agnostic_stopped {
                    let running_med = if all_outcomes.len() > 1 {
                        let mut sorted = all_outcomes.clone();
                        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        sorted[sorted.len() / 2]
                    } else { outcome };
                    let signal = Signal { arm: if is_trt { Arm::Treatment } else { Arm::Control }, good: outcome > running_med };
                    if agnostic.observe(signal) { agnostic_stopped = true; }
                }

                // Track at steps
                if step_idx < steps.len() && i == steps[step_idx] {
                    step_vals[step_idx].push(proc.wealth);
                    if is_sample {
                        let idx = sample_indices.iter().position(|&x| x == sim).unwrap();
                        sample_trajs[idx].push(proc.wealth);
                    }
                    step_idx += 1;
                }

                if stop_n.is_none() && proc.wealth > threshold {
                    stop_n = Some(i);
                    effect_at_stop = proc.current_effect();
                }
            }

            let effect_final = proc.current_effect();
            let t_significant = t_test_significant(&outcomes, alpha);
            trials.push(Trial { stop_n, effect_at_stop, effect_final, min_wealth, t_significant, agnostic_stopped });
        } else {
            let mut proc = MADProcess::new(burn_in, ramp, c_max);

            for i in 1..=n_pts {
                let is_trt = rng.gen_bool(0.5);
                let mu = if is_trt { mu_trt } else { mu_ctrl };

                // Sample outcome (MAD is unbounded, so Normal or Uniform)
                let outcome = match distribution {
                    Distribution::Normal | Distribution::VFDMixture => {
                        mu + sample_standard_normal(rng) * sd
                    }
                    Distribution::Uniform => {
                        rng.gen::<f64>() * sd * 2.0 - sd + mu
                    }
                };
                outcomes.push((outcome, is_trt));

                proc.update(i, outcome, is_trt);
                min_wealth = min_wealth.min(proc.wealth);
                all_outcomes.push(outcome);

                // Agnostic with running median
                if !agnostic_stopped {
                    let running_med = if all_outcomes.len() > 1 {
                        let mut sorted = all_outcomes.clone();
                        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        sorted[sorted.len() / 2]
                    } else { outcome };
                    let signal = Signal { arm: if is_trt { Arm::Treatment } else { Arm::Control }, good: outcome > running_med };
                    if agnostic.observe(signal) { agnostic_stopped = true; }
                }

                // Track at steps
                if step_idx < steps.len() && i == steps[step_idx] {
                    step_vals[step_idx].push(proc.wealth);
                    if is_sample {
                        let idx = sample_indices.iter().position(|&x| x == sim).unwrap();
                        sample_trajs[idx].push(proc.wealth);
                    }
                    step_idx += 1;
                }

                if stop_n.is_none() && proc.wealth > threshold {
                    stop_n = Some(i);
                    effect_at_stop = proc.current_effect(sd);
                }
            }

            let effect_final = proc.current_effect(sd);
            let t_significant = t_test_significant(&outcomes, alpha);
            trials.push(Trial { stop_n, effect_at_stop, effect_final, min_wealth, t_significant, agnostic_stopped });
        }
    }
    println!(" done");

    // Compute percentiles
    for (i, vals) in step_vals.iter_mut().enumerate() {
        if vals.is_empty() { continue; }
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = vals.len();
        y_lo[i] = vals[(n as f64 * 0.05) as usize];
        y_med[i] = vals[n / 2];
        y_hi[i] = vals[((n as f64 * 0.95) as usize).min(n - 1)];
    }

    let x: Vec<usize> = steps;
    (trials, sample_trajs, x, y_lo, y_med, y_hi)
}

fn run_type1<R: Rng + ?Sized>(
    rng: &mut R, method: Method, n_pts: usize, n_sims: usize,
    mu_ctrl: f64, sd: f64, min_val: f64, max_val: f64,
    threshold: f64, burn_in: usize, ramp: usize, c_max: f64,
    distribution: Distribution, vfd_params: Option<VFDParams>,
) -> f64 {
    let method_name = if method == Method::LinearERT { "e-RTo" } else { "e-RTc" };
    print!("  {} Type I Error... ", method_name);
    io::stdout().flush().unwrap();

    let mut rejections = 0;
    for _ in 0..n_sims {
        if method == Method::LinearERT {
            let mut proc = LinearERTProcess::new(burn_in, ramp, min_val, max_val);
            for i in 1..=n_pts {
                let is_trt = rng.gen_bool(0.5);
                // Under null, both arms have same distribution (use control params)
                let outcome = match distribution {
                    Distribution::VFDMixture => {
                        let vfd = vfd_params.unwrap();
                        sample_vfd(rng, vfd.p_death_ctrl, vfd.survivor_mean_ctrl, vfd.max_val)
                    }
                    Distribution::Normal => {
                        (mu_ctrl + sample_standard_normal(rng) * sd).clamp(min_val, max_val)
                    }
                    Distribution::Uniform => {
                        ((rng.gen::<f64>() * 2.0 - 1.0) * sd * 1.5 + mu_ctrl).clamp(min_val, max_val)
                    }
                };
                proc.update(i, outcome, is_trt);
                if proc.wealth > threshold { rejections += 1; break; }
            }
        } else {
            let mut proc = MADProcess::new(burn_in, ramp, c_max);
            for i in 1..=n_pts {
                let is_trt = rng.gen_bool(0.5);
                let outcome = match distribution {
                    Distribution::Normal | Distribution::VFDMixture => {
                        mu_ctrl + sample_standard_normal(rng) * sd
                    }
                    Distribution::Uniform => {
                        rng.gen::<f64>() * sd * 2.0 - sd + mu_ctrl
                    }
                };
                proc.update(i, outcome, is_trt);
                if proc.wealth > threshold { rejections += 1; break; }
            }
        }
    }

    let type1 = (rejections as f64 / n_sims as f64) * 100.0;
    println!("{:.2}%", type1);
    type1
}

// === HTML REPORT ===

fn build_report(
    console: &str, method_name: &str, n_pts: usize, _n_sims: usize,
    threshold: f64, fut_watch: f64,
    x: &[usize], y_lo: &[f64], y_med: &[f64], y_hi: &[f64],
    trajs: &[Vec<f64>], stops: &[f64], min_wealths: &[f64],
    grid: &[(f64, usize, usize, usize, f64)],
) -> String {
    let x_json = format!("{:?}", x);
    let lo_json = format!("{:?}", y_lo);
    let med_json = format!("{:?}", y_med);
    let hi_json = format!("{:?}", y_hi);

    let mut sample_traces = String::new();
    for traj in trajs.iter().take(30) {
        sample_traces.push_str(&format!(
            "{{type:'scatter',mode:'lines',x:{},y:{:?},line:{{color:'rgba(100,100,100,0.3)',width:1}},showlegend:false}},",
            x_json, traj
        ));
    }

    // ECDF for stops
    let mut sorted_stops = stops.to_vec();
    sorted_stops.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let ecdf_x: Vec<f64> = sorted_stops.clone();
    let ecdf_y: Vec<f64> = (1..=sorted_stops.len()).map(|i| i as f64 / sorted_stops.len() as f64).collect();

    // ECDF for min wealth
    let mut sorted_mw = min_wealths.to_vec();
    sorted_mw.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mw_x: Vec<f64> = sorted_mw.clone();
    let mw_y: Vec<f64> = (1..=sorted_mw.len()).map(|i| i as f64 / sorted_mw.len() as f64).collect();


    // Grid line plot data
    let grid_th: Vec<f64> = grid.iter().map(|(th, _, _, _, _)| *th).collect();
    let grid_t: Vec<f64> = grid.iter().map(|(_, n_trig, n_t, _, _)| if *n_trig > 0 { (*n_t as f64 / *n_trig as f64) * 100.0 } else { 0.0 }).collect();
    let grid_e: Vec<f64> = grid.iter().map(|(_, n_trig, _, n_e, _)| if *n_trig > 0 { (*n_e as f64 / *n_trig as f64) * 100.0 } else { 0.0 }).collect();

    format!(r#"<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{} Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
body{{font-family:system-ui,-apple-system,sans-serif;max-width:1400px;margin:0 auto;padding:20px;background:#fafafa}}
h1{{color:#1a1a2e}}h2,h3{{color:#16213e}}
pre{{background:#fff;padding:15px;border-radius:8px;border:1px solid #ddd;overflow-x:auto;font-size:13px}}
table{{margin:10px 0}}th,td{{padding:8px 12px;text-align:right}}th{{background:#f0f0f0}}
.plot-container{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin:20px 0}}
.plot{{background:#fff;border-radius:8px;padding:10px;box-shadow:0 1px 3px rgba(0,0,0,0.1)}}
</style></head><body>
<h1>{} Simulation Report</h1>
<h2>Console Output</h2>
<pre>{}</pre>
<h2>Visualizations</h2>
<div class="plot-container">
<div class="plot"><div id="p1" style="height:350px"></div></div>
<div class="plot"><div id="p2" style="height:350px"></div></div>
<div class="plot"><div id="p3" style="height:350px"></div></div>
<div class="plot"><div id="p4" style="height:350px"></div></div>
<div class="plot"><div id="p5" style="height:350px"></div></div>
<div class="plot"><div id="p6" style="height:350px"></div></div>
</div>
<script>
Plotly.newPlot('p1',[
  {{type:'scatter',x:{},y:{},line:{{width:0}},showlegend:false}},
  {{type:'scatter',x:{},y:{},fill:'tonexty',fillcolor:'rgba(70,130,180,0.3)',line:{{width:0}},showlegend:false}},
  {{type:'scatter',x:{},y:{},line:{{color:'steelblue',width:2}},name:'Median'}}
],{{title:'Wealth Trajectory (5-95% CI)',yaxis:{{type:'log',title:'e-value'}},xaxis:{{title:'Patients'}},
shapes:[{{type:'line',x0:0,x1:1,xref:'paper',y0:{},y1:{},line:{{color:'green',width:2,dash:'dash'}}}}]}});
Plotly.newPlot('p2',[{}
  {{type:'scatter',x:[0,{}],y:[{},{}],line:{{color:'green',width:2,dash:'dash'}},name:'Success'}},
  {{type:'scatter',x:[0,{}],y:[{},{}],line:{{color:'coral',width:2,dash:'dot'}},name:'Futility'}}
],{{title:'Sample Trajectories (n=30)',yaxis:{{type:'log',title:'e-value'}},xaxis:{{title:'Patients'}}}});
Plotly.newPlot('p3',[{{type:'scatter',mode:'lines',x:{:?},y:{:?},line:{{color:'steelblue',width:2}}}}],
{{title:'Stopping Time ECDF',xaxis:{{title:'Patient #'}},yaxis:{{title:'Cumulative Proportion'}}}});
Plotly.newPlot('p4',[{{type:'scatter',mode:'lines',x:{:?},y:{:?},line:{{color:'coral',width:2}}}}],
{{title:'Minimum Wealth ECDF',xaxis:{{title:'Min Wealth'}},yaxis:{{title:'Cumulative Proportion'}},
shapes:[{{type:'line',x0:{},x1:{},y0:0,y1:1,line:{{color:'red',width:1,dash:'dash'}}}}]}});
Plotly.newPlot('p5',[
  {{type:'scatter',mode:'lines+markers',x:{:?},y:{:?},name:'t-test+',line:{{color:'steelblue'}}}},
  {{type:'scatter',mode:'lines+markers',x:{:?},y:{:?},name:'e-RT+',line:{{color:'coral'}}}}
],{{title:'Recovery Rate by Threshold',xaxis:{{title:'Futility Threshold'}},yaxis:{{title:'% Would Succeed'}}}});
Plotly.newPlot('p6',[{{type:'histogram',x:{:?},nbinsx:30,marker:{{color:'steelblue'}}}}],
{{title:'Final Effect Distribution',xaxis:{{title:'Effect Size'}},yaxis:{{title:'Count'}}}});
</script></body></html>"#,
        method_name, method_name, console,
        // p1: CI plot
        x_json, lo_json, x_json, hi_json, x_json, med_json, threshold, threshold,
        // p2: sample traces
        sample_traces, n_pts, threshold, threshold, n_pts, fut_watch, fut_watch,
        // p3: stop ECDF
        ecdf_x, ecdf_y,
        // p4: min wealth ECDF
        mw_x, mw_y, fut_watch, fut_watch,
        // p5: grid lines
        grid_th, grid_t, grid_th, grid_e,
        // p6: effect histogram
        stops
    )
}

// === MAIN ===

pub fn run() {
    println!("\n==========================================");
    println!("   CONTINUOUS e-RT SIMULATION");
    println!("==========================================\n");

    let method_choice = get_choice("Select method:", &[
        "e-RTo (ordinal/bounded, e.g., VFD 0-28)",
        "e-RTc (continuous/unbounded)",
    ]);
    let method = if method_choice == 1 { Method::LinearERT } else { Method::MAD };
    let method_name = if method == Method::LinearERT { "e-RTo" } else { "e-RTc" };

    // Distribution selection for e-RTo
    let (distribution, vfd_params) = if method == Method::LinearERT {
        let dist_choice = get_choice("Outcome distribution:", &[
            "VFD Mixture (bimodal: mortality spike + survivor distribution)",
            "Normal",
            "Uniform",
        ]);
        match dist_choice {
            1 => {
                println!("\n--- VFD Mixture Parameters ---");
                let p_death_ctrl: f64 = get_input("Control mortality rate (e.g., 0.30): ");
                let p_death_trt: f64 = get_input("Treatment mortality rate (e.g., 0.25): ");
                let survivor_mean_ctrl: f64 = get_input("Control survivor mean VFD (e.g., 18): ");
                let survivor_mean_trt: f64 = get_input("Treatment survivor mean VFD (e.g., 20): ");
                let max_val: f64 = get_input("Max VFD (e.g., 28): ");
                let vfd = VFDParams { p_death_ctrl, p_death_trt, survivor_mean_ctrl, survivor_mean_trt, max_val };
                (Distribution::VFDMixture, Some(vfd))
            }
            2 => (Distribution::Normal, None),
            _ => (Distribution::Uniform, None),
        }
    } else {
        let dist_choice = get_choice("Outcome distribution:", &["Normal", "Uniform"]);
        if dist_choice == 1 { (Distribution::Normal, None) } else { (Distribution::Uniform, None) }
    };

    // Get means (computed from VFD params if mixture)
    let (mu_ctrl, mu_trt, sd, min_val, max_val) = if distribution == Distribution::VFDMixture {
        let vfd = vfd_params.unwrap();
        let (mc, sc) = vfd_mixture_moments(vfd.p_death_ctrl, vfd.survivor_mean_ctrl, vfd.max_val);
        let (mt, st) = vfd_mixture_moments(vfd.p_death_trt, vfd.survivor_mean_trt, vfd.max_val);
        let pooled_sd = ((sc.powi(2) + st.powi(2)) / 2.0).sqrt();
        println!("\nDerived from mixture model:");
        println!("  Control mean: {:.2}, SD: {:.2}", mc, sc);
        println!("  Treatment mean: {:.2}, SD: {:.2}", mt, st);
        println!("  Pooled SD: {:.2}", pooled_sd);
        (mc, mt, pooled_sd, 0.0, vfd.max_val)
    } else {
        let mu_ctrl: f64 = get_input("Control Mean (e.g., 14): ");
        let mu_trt: f64 = get_input("Treatment Mean (e.g., 18): ");
        let (min_val, max_val) = if method == Method::LinearERT {
            (get_input("Min bound (e.g., 0): "), get_input("Max bound (e.g., 28): "))
        } else { (f64::MIN, f64::MAX) };
        let sd: f64 = get_input("Standard Deviation (e.g., 8): ");
        (mu_ctrl, mu_trt, sd, min_val, max_val)
    };

    let design_effect = if method == Method::LinearERT {
        (mu_trt - mu_ctrl).abs()
    } else {
        ((mu_trt - mu_ctrl) / sd).abs()
    };

    let n_pts = if get_bool("Calculate Sample Size automatically?") {
        let power: f64 = get_input("Target Power (e.g., 0.90): ");
        let cohen_d = ((mu_trt - mu_ctrl) / sd).abs();
        let n = calculate_n_continuous(cohen_d, power);
        println!("\nFrequentist N (Power {:.0}%, d={:.2}): {}", power * 100.0, cohen_d, n);
        if get_bool("Add buffer?") {
            let buf: f64 = get_input("Buffer percentage (e.g., 15): ");
            let buffered = (n as f64 * (1.0 + buf / 100.0)).ceil() as usize;
            println!("Buffered N: {}", buffered);
            buffered
        } else { n }
    } else {
        get_input_usize("Enter Number of Patients: ")
    };

    let n_sims = get_input_usize("Number of simulations (e.g., 2000): ");
    let threshold: f64 = get_input("Success threshold (default 20): ");
    let fut_watch: f64 = get_input("Futility watch (default 0.2): ");
    let seed = get_optional_input("Seed (Enter for random): ");

    let burn_in: usize = 20;
    let ramp: usize = 50;
    let c_max: f64 = 0.6;

    // Console output capture
    let mut console = String::new();
    console.push_str(&format!("{}\n", chrono_lite()));
    console.push_str(&format!("\n==========================================\n"));
    console.push_str(&format!("   PARAMETERS\n"));
    console.push_str(&format!("==========================================\n"));
    console.push_str(&format!("Method:      {}\n", method_name));
    let dist_name = match distribution {
        Distribution::VFDMixture => "VFD Mixture",
        Distribution::Normal => "Normal",
        Distribution::Uniform => "Uniform",
    };
    console.push_str(&format!("Distribution: {}\n", dist_name));
    if let Some(vfd) = vfd_params {
        console.push_str(&format!("  Ctrl mortality: {:.0}%, survivor mean: {:.1}\n",
            vfd.p_death_ctrl * 100.0, vfd.survivor_mean_ctrl));
        console.push_str(&format!("  Trt mortality:  {:.0}%, survivor mean: {:.1}\n",
            vfd.p_death_trt * 100.0, vfd.survivor_mean_trt));
    }
    console.push_str(&format!("Control:     {:.2}\n", mu_ctrl));
    console.push_str(&format!("Treatment:   {:.2}\n", mu_trt));
    if method == Method::LinearERT {
        console.push_str(&format!("Bounds:      [{:.0}, {:.0}]\n", min_val, max_val));
        console.push_str(&format!("Design Δ:    {:.2}\n", design_effect));
    } else {
        console.push_str(&format!("SD:          {:.2}\n", sd));
        console.push_str(&format!("Cohen's d:   {:.2}\n", design_effect));
    }
    console.push_str(&format!("N:           {}\n", n_pts));
    console.push_str(&format!("Simulations: {}\n", n_sims));
    console.push_str(&format!("Threshold:   {} (α={:.3})\n", threshold, 1.0/threshold));
    console.push_str(&format!("Futility:    {}\n", fut_watch));

    let mut rng: Box<dyn RngCore> = match seed {
        Some(s) => { console.push_str(&format!("Seed:        {}\n", s)); Box::new(StdRng::seed_from_u64(s)) }
        None => { console.push_str("Seed:        random\n"); Box::new(rand::thread_rng()) }
    };

    println!("\n==========================================");
    println!("   RUNNING SIMULATIONS");
    println!("==========================================\n");

    // Type I Error
    let type1 = run_type1(&mut *rng, method, n_pts, n_sims, mu_ctrl, sd, min_val, max_val,
        threshold, burn_in, ramp, c_max, distribution, vfd_params);

    // Power simulation
    let (trials, trajs, x, y_lo, y_med, y_hi) = run_simulation(
        &mut *rng, method, n_pts, n_sims, mu_ctrl, mu_trt, sd, min_val, max_val,
        design_effect, threshold, fut_watch, burn_in, ramp, c_max, distribution, vfd_params
    );

    // Compute statistics
    let successes: Vec<&Trial> = trials.iter().filter(|t| t.stop_n.is_some()).collect();
    let power = (successes.len() as f64 / n_sims as f64) * 100.0;
    let agn_power = (trials.iter().filter(|t| t.agnostic_stopped).count() as f64 / n_sims as f64) * 100.0;
    let t_test_power = t_test_power_continuous(mu_trt - mu_ctrl, sd, n_pts, 1.0/threshold) * 100.0;

    let (avg_stop, avg_eff_stop, avg_eff_final, type_m) = if !successes.is_empty() {
        let avg_n = successes.iter().map(|t| t.stop_n.unwrap() as f64).sum::<f64>() / successes.len() as f64;
        let avg_s = successes.iter().map(|t| t.effect_at_stop.abs()).sum::<f64>() / successes.len() as f64;
        let avg_f = successes.iter().map(|t| t.effect_final.abs()).sum::<f64>() / successes.len() as f64;
        (avg_n, avg_s, avg_f, if avg_f > 0.0 { avg_s / avg_f } else { 1.0 })
    } else { (0.0, 0.0, 0.0, 1.0) };

    // Futility grid (single pass)
    let thresholds = [0.1, 0.2, 0.3, 0.4, 0.5];
    let mut grid: Vec<(f64, usize, usize, usize, f64)> = thresholds.iter()
        .map(|&th| (th, 0usize, 0usize, 0usize, 0.0f64)).collect();

    for t in &trials {
        for (th, n_trig, n_t, n_e, sum_eff) in &mut grid {
            if t.min_wealth < *th {
                *n_trig += 1;
                if t.t_significant { *n_t += 1; }
                if t.stop_n.is_some() { *n_e += 1; }
                *sum_eff += t.effect_final.abs();
            }
        }
    }

    // Console results
    console.push_str(&format!("\n==========================================\n"));
    console.push_str(&format!("   RESULTS\n"));
    console.push_str(&format!("==========================================\n"));
    console.push_str(&format!("Type I Error:  {:.2}%\n", type1));
    console.push_str(&format!("\n--- Power at N={} ---\n", n_pts));
    console.push_str(&format!("t-test:    {:.1}%\n", t_test_power));
    console.push_str(&format!("{}:      {:.1}%\n", method_name, power));
    console.push_str(&format!("e-RTu:     {:.1}%\n", agn_power));

    if !successes.is_empty() {
        console.push_str(&format!("\n--- Stopping ---\n"));
        console.push_str(&format!("Avg stop:      {:.0} ({:.0}%)\n", avg_stop, avg_stop / n_pts as f64 * 100.0));
        console.push_str(&format!("Effect @ stop: {:.2}\n", avg_eff_stop));
        console.push_str(&format!("Effect @ end:  {:.2}\n", avg_eff_final));
        console.push_str(&format!("Type M:        {:.2}x\n", type_m));
    }

    console.push_str(&format!("\n--- Futility Grid ---\n"));
    console.push_str(&format!("{:<10} {:>10} {:>10} {:>10}\n", "Threshold", "Triggered", "t-test+", "e-RT+"));
    for (th, n_trig, n_t, n_e, _) in &grid {
        if *n_trig > 0 {
            let trig_pct = (*n_trig as f64 / n_sims as f64) * 100.0;
            let t_pct = (*n_t as f64 / *n_trig as f64) * 100.0;
            let e_pct = (*n_e as f64 / *n_trig as f64) * 100.0;
            console.push_str(&format!("{:<10.1} {:>9.1}% {:>9.1}% {:>9.1}%\n", th, trig_pct, t_pct, e_pct));
        }
    }

    // Print to console
    print!("{}", console);

    // Generate report
    println!("\nGenerating report...");
    let stops: Vec<f64> = successes.iter().map(|t| t.stop_n.unwrap() as f64).collect();
    let min_wealths: Vec<f64> = trials.iter().map(|t| t.min_wealth).collect();

    let html = build_report(&console, method_name, n_pts, n_sims, threshold, fut_watch,
        &x, &y_lo, &y_med, &y_hi, &trajs, &stops, &min_wealths, &grid);

    let mut file = File::create("continuous_report.html").unwrap();
    file.write_all(html.as_bytes()).unwrap();
    println!("\n>> continuous_report.html");
}
