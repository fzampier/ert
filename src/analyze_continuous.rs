//! Analyze real continuous trial data from CSV

use std::error::Error;
use std::fs::File;
use std::io::Write;
use csv::ReaderBuilder;
use serde::Deserialize;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::ert_core::{
    get_input, get_input_usize, get_bool, get_string, get_choice, median, mad, chrono_lite,
    LinearERTProcess, MADProcess,
};

// === DATA STRUCTURES ===

#[derive(Debug, Deserialize)]
struct CsvRowRaw {
    #[serde(default)]
    #[allow(dead_code)]
    _index: Option<String>,
    treatment: String,
    outcome: String,
}

#[derive(Clone, Copy, PartialEq)]
enum Method { LinearERT, MAD }

#[derive(Clone)]
struct FutilityPoint {
    patient_num: usize,
    _wealth: f64,
    required_effect: f64,
    ratio_to_design: f64,
}

#[derive(Clone)]
struct DesignParams {
    control_mean: f64,
    _treatment_mean: f64,
    sd: f64,
    design_effect_linear: f64,
    design_effect_mad: f64,
}

struct AnalysisResult {
    method: Method,
    n_total: usize,
    n_trt: usize,
    n_ctrl: usize,
    crossed: bool,
    crossed_at: Option<usize>,
    effect_at_cross: Option<f64>,
    final_effect: f64,
    final_mean_trt: f64,
    final_mean_ctrl: f64,
    final_sd: f64,
    final_evalue: f64,
    type_m: Option<f64>,
    trajectory: Vec<f64>,
    // Futility
    futility_points: Vec<FutilityPoint>,
    futility_regions: Vec<(usize, usize)>,
    design: Option<DesignParams>,
}

// === CLI ===

pub fn run_cli(csv_path: &str, opts: &crate::AnalyzeOptions) -> Result<(), Box<dyn Error>> {
    println!("\n==========================================");
    println!("   ANALYZE CONTINUOUS TRIAL DATA");
    println!("==========================================\n");

    println!("Reading {}...", csv_path);
    let data = read_csv(csv_path)?;
    if data.is_empty() { println!("Error: No valid data."); return Ok(()); }

    // Summary stats
    let trt: Vec<f64> = data.iter().filter(|r| r.0 == 1).map(|r| r.1).collect();
    let ctrl: Vec<f64> = data.iter().filter(|r| r.0 == 0).map(|r| r.1).collect();
    let (n_trt, n_ctrl) = (trt.len(), ctrl.len());
    let mean_trt = if n_trt > 0 { trt.iter().sum::<f64>() / n_trt as f64 } else { 0.0 };
    let mean_ctrl = if n_ctrl > 0 { ctrl.iter().sum::<f64>() / n_ctrl as f64 } else { 0.0 };
    let min_obs = data.iter().map(|r| r.1).fold(f64::INFINITY, f64::min);
    let max_obs = data.iter().map(|r| r.1).fold(f64::NEG_INFINITY, f64::max);

    let ss_trt: f64 = trt.iter().map(|x| (x - mean_trt).powi(2)).sum();
    let ss_ctrl: f64 = ctrl.iter().map(|x| (x - mean_ctrl).powi(2)).sum();
    let pooled_sd = if n_trt > 1 && n_ctrl > 1 {
        ((ss_trt + ss_ctrl) / (n_trt + n_ctrl - 2) as f64).sqrt()
    } else { 1.0 };

    println!("\n--- Data Summary ---");
    println!("Total: {}  Trt: {} (mean {:.2})  Ctrl: {} (mean {:.2})", data.len(), n_trt, mean_trt, n_ctrl, mean_ctrl);
    println!("Range: [{:.2}, {:.2}]  SD: {:.2}  Diff: {:.2}  d: {:.2}",
             min_obs, max_obs, pooled_sd, mean_trt - mean_ctrl, (mean_trt - mean_ctrl) / pooled_sd);

    // Determine method from CLI options
    let method = match opts.method.as_deref() {
        Some("rto") | Some("RTo") | Some("linear") => Method::LinearERT,
        Some("rtc") | Some("RTc") | Some("mad") | Some("MAD") => Method::MAD,
        _ => Method::MAD, // default to MAD/unbounded
    };

    let burn_in = opts.burn_in.unwrap_or(50);
    let ramp = opts.ramp.unwrap_or(100);
    let threshold = opts.threshold.unwrap_or(20.0);
    let c_max = 0.6;

    let (analysis_min, analysis_max) = if method == Method::LinearERT {
        (opts.min_val.unwrap_or(min_obs), opts.max_val.unwrap_or(max_obs))
    } else {
        (min_obs, max_obs)
    };

    let method_name = match method { Method::LinearERT => "e-RTo", Method::MAD => "e-RTc" };
    println!("\n--- Parameters ---");
    println!("Method: {}  Burn-in: {}  Ramp: {}  Threshold: {}", method_name, burn_in, ramp, threshold);
    if method == Method::LinearERT {
        println!("Bounds: [{:.2}, {:.2}]", analysis_min, analysis_max);
    }

    println!("\n--- Running Analysis ---");
    let result = analyze(&data, method, burn_in, ramp, threshold, c_max,
                         analysis_min, analysis_max, pooled_sd, None, None);

    print_results(&result, threshold, None);

    if opts.generate_report {
        let html = build_html(&result, csv_path, burn_in, ramp, threshold, None, c_max, analysis_min, analysis_max);
        File::create("continuous_analysis_report.html")?.write_all(html.as_bytes())?;
        println!("\n>> Saved: continuous_analysis_report.html");
    }

    Ok(())
}

// === INTERACTIVE ===

pub fn run() -> Result<(), Box<dyn Error>> {
    println!("\n==========================================");
    println!("   ANALYZE CONTINUOUS TRIAL DATA");
    println!("==========================================\n");

    let csv_path = get_string("Path to CSV file: ");
    println!("\nReading {}...", csv_path);
    let data = read_csv(&csv_path)?;
    if data.is_empty() { println!("Error: No valid data."); return Ok(()); }

    // Summary stats
    let trt: Vec<f64> = data.iter().filter(|r| r.0 == 1).map(|r| r.1).collect();
    let ctrl: Vec<f64> = data.iter().filter(|r| r.0 == 0).map(|r| r.1).collect();
    let (n_trt, n_ctrl) = (trt.len(), ctrl.len());
    let mean_trt = if n_trt > 0 { trt.iter().sum::<f64>() / n_trt as f64 } else { 0.0 };
    let mean_ctrl = if n_ctrl > 0 { ctrl.iter().sum::<f64>() / n_ctrl as f64 } else { 0.0 };
    let min_val = data.iter().map(|r| r.1).fold(f64::INFINITY, f64::min);
    let max_val = data.iter().map(|r| r.1).fold(f64::NEG_INFINITY, f64::max);

    let ss_trt: f64 = trt.iter().map(|x| (x - mean_trt).powi(2)).sum();
    let ss_ctrl: f64 = ctrl.iter().map(|x| (x - mean_ctrl).powi(2)).sum();
    let pooled_sd = if n_trt > 1 && n_ctrl > 1 {
        ((ss_trt + ss_ctrl) / (n_trt + n_ctrl - 2) as f64).sqrt()
    } else { 1.0 };

    println!("\n--- Data Summary ---");
    println!("Total: {}  Trt: {} (mean {:.2})  Ctrl: {} (mean {:.2})", data.len(), n_trt, mean_trt, n_ctrl, mean_ctrl);
    println!("Range: [{:.2}, {:.2}]  SD: {:.2}  Diff: {:.2}  d: {:.2}",
             min_val, max_val, pooled_sd, mean_trt - mean_ctrl, (mean_trt - mean_ctrl) / pooled_sd);

    let method = if get_choice("\nSelect method:", &["e-RTo (ordinal/bounded)", "e-RTc (continuous/unbounded)"]) == 1 {
        Method::LinearERT
    } else { Method::MAD };

    let (analysis_min, analysis_max) = if method == Method::LinearERT {
        println!("\ne-RTo bounds (observed: [{:.2}, {:.2}]):", min_val, max_val);
        (get_input("Min: "), get_input("Max: "))
    } else { (min_val, max_val) };

    println!("\n--- Parameters ---");
    let burn_in = get_input_usize("Burn-in (default 50): ");
    let ramp = get_input_usize("Ramp (default 100): ");
    let threshold = get_input("Success threshold (default 20): ");
    let c_max = if method == Method::MAD { get_input("c_max (default 0.6): ") } else { 0.6 };

    let use_futility = get_bool("Enable futility monitoring?");
    let (fut_thresh, design) = if use_futility {
        let fut = get_input("Futility threshold (e.g., 0.5): ");
        println!("\nDesign assumptions:");
        let dc = get_input("Design control mean: ");
        let dt = get_input("Design treatment mean: ");
        let ds = get_input("Design SD: ");
        (Some(fut), Some(DesignParams {
            control_mean: dc, _treatment_mean: dt, sd: ds,
            design_effect_linear: (dt - dc).abs(),
            design_effect_mad: (dt - dc).abs() / ds,
        }))
    } else { (None, None) };

    println!("\n--- Running Analysis ---");
    let result = analyze(&data, method, burn_in, ramp, threshold, c_max,
                         analysis_min, analysis_max, pooled_sd, fut_thresh, design);

    print_results(&result, threshold, fut_thresh);

    if get_bool("\nGenerate HTML report?") {
        let html = build_html(&result, &csv_path, burn_in, ramp, threshold, fut_thresh, c_max, analysis_min, analysis_max);
        File::create("continuous_analysis_report.html")?.write_all(html.as_bytes())?;
        println!(">> Saved: continuous_analysis_report.html");
    }

    Ok(())
}

// === CSV PARSING ===

fn read_csv(path: &str) -> Result<Vec<(u8, f64)>, Box<dyn Error>> {
    let mut reader = ReaderBuilder::new().has_headers(true).flexible(true).from_path(path)?;
    let mut data = Vec::new();
    let mut skipped = 0;

    for result in reader.deserialize() {
        let row: CsvRowRaw = result?;
        let treatment = match row.treatment.parse::<u8>() {
            Ok(v) if v <= 1 => v,
            _ => { skipped += 1; continue; }
        };
        let outcome = match row.outcome.parse::<f64>() {
            Ok(v) if v.is_finite() => v,
            _ => { skipped += 1; continue; }
        };
        data.push((treatment, outcome));
    }

    if skipped > 0 { println!("  (Skipped {} invalid rows)", skipped); }
    Ok(data)
}

// === MONTE CARLO: REQUIRED EFFECT ===

fn required_effect_linear(wealth: f64, n_rem: usize, mu_ctrl: f64, sd: f64, min_val: f64, max_val: f64,
                          burn_in: usize, ramp: usize, threshold: f64, mc_sims: usize) -> f64 {
    if n_rem == 0 { return max_val - min_val; }
    let mut rng = StdRng::seed_from_u64(42);
    let (mut low, mut high) = (0.001, (max_val - min_val) / 2.0);

    for _ in 0..8 {
        let mid = (low + high) / 2.0;
        let mu_trt = mu_ctrl + mid;
        let mut successes = 0;

        for _ in 0..mc_sims {
            let mut w = wealth;
            let (mut sum_t, mut n_t, mut sum_c, mut n_c) = (0.0, 0.0, 0.0, 0.0);

            for j in 1..=n_rem {
                let is_trt = rng.gen_bool(0.5);
                let mu = if is_trt { mu_trt } else { mu_ctrl };
                let outcome = ((rng.gen::<f64>() * 2.0 - 1.0) * sd * 1.5 + mu).clamp(min_val, max_val);

                let mt = if n_t > 0.0 { sum_t / n_t } else { (min_val + max_val) / 2.0 };
                let mc = if n_c > 0.0 { sum_c / n_c } else { (min_val + max_val) / 2.0 };
                let delta = mt - mc;

                if is_trt { n_t += 1.0; sum_t += outcome; }
                else { n_c += 1.0; sum_c += outcome; }

                if j > burn_in {
                    let c_i = (((j - burn_in) as f64) / ramp as f64).clamp(0.0, 1.0);
                    let x = (outcome - min_val) / (max_val - min_val);
                    let scalar = 2.0 * x - 1.0;
                    let delta_norm = delta / (max_val - min_val);
                    let lambda = (0.5 + 0.5 * c_i * delta_norm * scalar).clamp(0.001, 0.999);
                    w *= if is_trt { lambda / 0.5 } else { (1.0 - lambda) / 0.5 };
                }
                if w >= threshold { successes += 1; break; }
            }
        }
        if (successes as f64 / mc_sims as f64) < 0.5 { low = mid; } else { high = mid; }
    }
    (low + high) / 2.0
}

fn required_effect_mad(wealth: f64, n_rem: usize, mu_ctrl: f64, sd: f64, burn_in: usize, ramp: usize,
                       c_max: f64, threshold: f64, mc_sims: usize) -> f64 {
    if n_rem == 0 { return 2.0; }
    let mut rng = StdRng::seed_from_u64(42);
    let (mut low, mut high) = (0.001, 2.0);

    for _ in 0..8 {
        let mid = (low + high) / 2.0;
        let mu_trt = mu_ctrl + mid * sd;
        let mut successes = 0;

        for _ in 0..mc_sims {
            let mut w = wealth;
            let (mut outcomes, mut treatments): (Vec<f64>, Vec<bool>) = (Vec::new(), Vec::new());

            for j in 1..=n_rem {
                let is_trt = rng.gen_bool(0.5);
                let outcome = rng.gen::<f64>() * sd * 2.0 - sd + if is_trt { mu_trt } else { mu_ctrl };

                let direction = if !outcomes.is_empty() {
                    let trt: Vec<f64> = outcomes.iter().zip(&treatments).filter(|(_, &t)| t).map(|(&o, _)| o).collect();
                    let ctrl: Vec<f64> = outcomes.iter().zip(&treatments).filter(|(_, &t)| !t).map(|(&o, _)| o).collect();
                    let mt = if !trt.is_empty() { trt.iter().sum::<f64>() / trt.len() as f64 } else { 0.0 };
                    let mc = if !ctrl.is_empty() { ctrl.iter().sum::<f64>() / ctrl.len() as f64 } else { 0.0 };
                    if mt > mc { 1.0 } else if mt < mc { -1.0 } else { 0.0 }
                } else { 0.0 };

                outcomes.push(outcome);
                treatments.push(is_trt);

                if j > burn_in && outcomes.len() > 1 {
                    let past: Vec<f64> = outcomes[..outcomes.len()-1].to_vec();
                    let med = median(&past);
                    let s = { let m = mad(&past); if m > 0.0 { m } else { 1.0 } };
                    let r = (outcome - med) / s;
                    let g = r / (1.0 + r.abs());
                    let c_i = (((j - burn_in) as f64) / ramp as f64).clamp(0.0, 1.0);
                    let lambda = (0.5 + c_i * c_max * g * direction).clamp(0.001, 0.999);
                    w *= if is_trt { lambda / 0.5 } else { (1.0 - lambda) / 0.5 };
                }
                if w >= threshold { successes += 1; break; }
            }
        }
        if (successes as f64 / mc_sims as f64) < 0.5 { low = mid; } else { high = mid; }
    }
    (low + high) / 2.0
}

// === ANALYSIS ===

fn analyze(data: &[(u8, f64)], method: Method, burn_in: usize, ramp: usize, threshold: f64, c_max: f64,
           min_val: f64, max_val: f64, pooled_sd: f64, fut_thresh: Option<f64>, design: Option<DesignParams>) -> AnalysisResult {
    let n_total = data.len();
    let checkpoint = (n_total as f64 * 0.02).ceil() as usize;

    let mut trajectory = vec![1.0];
    let mut crossed = false;
    let mut crossed_at = None;
    let mut effect_at_cross = None;
    let mut futility_points = Vec::new();
    let mut futility_regions = Vec::new();
    let mut in_futility = false;
    let mut fut_start = 0;

    let (final_effect, final_mean_trt, final_mean_ctrl, final_sd, n_trt, n_ctrl, final_evalue);

    match method {
        Method::LinearERT => {
            let mut proc = LinearERTProcess::new(burn_in, ramp, min_val, max_val);
            for (i, &(trt, outcome)) in data.iter().enumerate() {
                let pnum = i + 1;
                proc.update(pnum, outcome, trt == 1);
                trajectory.push(proc.wealth);

                if !crossed && proc.wealth >= threshold {
                    crossed = true;
                    crossed_at = Some(pnum);
                    effect_at_cross = Some(proc.current_effect());
                }

                if let (Some(ft), Some(ref d)) = (fut_thresh, &design) {
                    let below = proc.wealth < ft;
                    if below && !in_futility { in_futility = true; fut_start = pnum; }
                    else if !below && in_futility { in_futility = false; futility_regions.push((fut_start, pnum - 1)); }

                    if below && pnum % checkpoint == 0 && pnum > burn_in {
                        let req = required_effect_linear(proc.wealth, n_total - pnum, d.control_mean, d.sd,
                                                         min_val, max_val, burn_in, ramp, threshold, 100);
                        futility_points.push(FutilityPoint {
                            patient_num: pnum, _wealth: proc.wealth, required_effect: req,
                            ratio_to_design: req / d.design_effect_linear,
                        });
                    }
                }
            }
            if in_futility { futility_regions.push((fut_start, n_total)); }

            final_effect = proc.current_effect();
            let (mt, mc) = proc.get_means();
            final_mean_trt = mt; final_mean_ctrl = mc;
            final_sd = pooled_sd;
            let (nt, nc) = proc.get_ns();
            n_trt = nt; n_ctrl = nc;
            final_evalue = proc.wealth;
        }
        Method::MAD => {
            let mut proc = MADProcess::new(burn_in, ramp, c_max);
            for (i, &(trt, outcome)) in data.iter().enumerate() {
                let pnum = i + 1;
                proc.update(pnum, outcome, trt == 1);
                trajectory.push(proc.wealth);

                if !crossed && proc.wealth >= threshold {
                    crossed = true;
                    crossed_at = Some(pnum);
                    effect_at_cross = Some(proc.current_effect(pooled_sd));
                }

                if let (Some(ft), Some(ref d)) = (fut_thresh, &design) {
                    let below = proc.wealth < ft;
                    if below && !in_futility { in_futility = true; fut_start = pnum; }
                    else if !below && in_futility { in_futility = false; futility_regions.push((fut_start, pnum - 1)); }

                    if below && pnum % checkpoint == 0 && pnum > burn_in {
                        let req = required_effect_mad(proc.wealth, n_total - pnum, d.control_mean, d.sd,
                                                      burn_in, ramp, c_max, threshold, 100);
                        futility_points.push(FutilityPoint {
                            patient_num: pnum, _wealth: proc.wealth, required_effect: req,
                            ratio_to_design: req / d.design_effect_mad,
                        });
                    }
                }
            }
            if in_futility { futility_regions.push((fut_start, n_total)); }

            final_effect = proc.current_effect(proc.get_pooled_sd());
            let (mt, mc) = proc.get_means();
            final_mean_trt = mt; final_mean_ctrl = mc;
            final_sd = proc.get_pooled_sd();
            let (nt, nc) = proc.get_ns();
            n_trt = nt; n_ctrl = nc;
            final_evalue = proc.wealth;
        }
    }

    let type_m = if crossed {
        let ec = effect_at_cross.unwrap().abs();
        let ef = final_effect.abs();
        if ef > 0.0 { Some(ec / ef) } else { None }
    } else { None };

    AnalysisResult {
        method, n_total, n_trt, n_ctrl, crossed, crossed_at, effect_at_cross,
        final_effect, final_mean_trt, final_mean_ctrl, final_sd, final_evalue, type_m, trajectory,
        futility_points, futility_regions, design,
    }
}

// === CONSOLE OUTPUT ===

fn print_results(r: &AnalysisResult, threshold: f64, fut_thresh: Option<f64>) {
    let (method_name, effect_label) = match r.method {
        Method::LinearERT => ("e-RTo", "Mean Diff"),
        Method::MAD => ("e-RTc", "Cohen's d"),
    };

    println!("\n=== RESULTS ({}) ===", method_name);
    println!("e-value: {:.4}  Threshold: {:.0}", r.final_evalue, threshold);

    if r.crossed {
        println!("Status: CROSSED at patient {}", r.crossed_at.unwrap());
    } else if let Some(f) = fut_thresh {
        if r.final_evalue < f { println!("Status: Below futility ({:.2})", f); }
        else { println!("Status: Did not cross"); }
    } else { println!("Status: Did not cross"); }

    if r.crossed {
        println!("\nAt crossing: {} = {:.3}", effect_label, r.effect_at_cross.unwrap());
    }
    println!("Final: {} = {:.3}  Trt={:.2}  Ctrl={:.2}  SD={:.2}",
             effect_label, r.final_effect, r.final_mean_trt, r.final_mean_ctrl, r.final_sd);

    if let Some(tm) = r.type_m { println!("Type M: {:.2}x", tm); }

    if let Some(ref d) = r.design {
        let de = match r.method {
            Method::LinearERT => d.design_effect_linear,
            Method::MAD => d.design_effect_mad,
        };
        println!("\n--- Futility ---");
        println!("Design {}: {:.3}  Episodes: {}", effect_label, de, r.futility_regions.len());
        if let Some(worst) = r.futility_points.iter().max_by(|a, b| a.ratio_to_design.partial_cmp(&b.ratio_to_design).unwrap()) {
            println!("Worst: patient {} req {:.3} ({:.2}x design)", worst.patient_num, worst.required_effect, worst.ratio_to_design);
        }
    }
}

// === HTML REPORT ===

fn build_html(r: &AnalysisResult, csv_path: &str, burn_in: usize, ramp: usize, threshold: f64,
              fut_thresh: Option<f64>, c_max: f64, min_val: f64, max_val: f64) -> String {
    let (method_name, effect_label) = match r.method {
        Method::LinearERT => ("e-RTo", "Mean Difference"),
        Method::MAD => ("e-RTc", "Cohen's d"),
    };

    let _status = if r.crossed {
        format!("<span style='color:#27ae60;font-weight:bold'>CROSSED at patient {}</span>", r.crossed_at.unwrap())
    } else if fut_thresh.map_or(false, |f| r.final_evalue < f) {
        "<span style='color:#e67e22'>Below futility</span>".to_string()
    } else {
        "<span style='color:#7f8c8d'>Did not cross</span>".to_string()
    };

    let cross_html = if r.crossed {
        format!("<p><strong>At crossing (patient {}):</strong> {} = {:.3}</p>", r.crossed_at.unwrap(), effect_label, r.effect_at_cross.unwrap())
    } else { String::new() };

    let type_m_html = r.type_m.map_or(String::new(), |tm| format!("<p><strong>Type M:</strong> {:.2}x</p>", tm));

    let futility_html = if let Some(ref d) = r.design {
        let de = match r.method { Method::LinearERT => d.design_effect_linear, Method::MAD => d.design_effect_mad };
        let worst_html = r.futility_points.iter()
            .max_by(|a, b| a.ratio_to_design.partial_cmp(&b.ratio_to_design).unwrap())
            .map_or(String::new(), |w| format!("<p>Worst: patient {} required {:.3} ({:.2}x design)</p>", w.patient_num, w.required_effect, w.ratio_to_design));
        format!(r#"<h3>Futility Analysis</h3>
<p><em>Note: Futility is decision support, not anytime-valid inference.</em></p>
<p>Design {}: {:.3}  |  Episodes: {}</p>{}"#, effect_label, de, r.futility_regions.len(), worst_html)
    } else { String::new() };

    let method_params = match r.method {
        Method::LinearERT => format!("Bounds: [{:.1}, {:.1}]", min_val, max_val),
        Method::MAD => format!("c_max: {:.2}", c_max),
    };

    let mut shapes = String::new();
    for (s, e) in &r.futility_regions {
        shapes.push_str(&format!("{{type:'rect',x0:{},x1:{},y0:0,y1:1,yref:'paper',fillcolor:'rgba(230,126,34,0.15)',line:{{width:0}}}},", s, e));
    }
    if let Some(f) = fut_thresh {
        shapes.push_str(&format!("{{type:'line',x0:0,x1:1,xref:'paper',y0:{},y1:{},line:{{color:'#e67e22',width:1.5,dash:'dot'}}}},", f, f));
    }
    shapes.push_str(&format!("{{type:'line',x0:0,x1:1,xref:'paper',y0:{},y1:{},line:{{color:'#27ae60',width:2,dash:'dash'}}}}", threshold, threshold));

    format!(r#"<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Continuous e-RT Analysis</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
body{{font-family:system-ui,-apple-system,sans-serif;max-width:1400px;margin:0 auto;padding:20px;background:#fafafa}}
h1{{color:#1a1a2e}}h2,h3{{color:#16213e}}
pre{{background:#fff;padding:15px;border-radius:8px;border:1px solid #ddd;overflow-x:auto;font-size:13px}}
.plot-container{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin:20px 0}}
.plot{{background:#fff;border-radius:8px;padding:10px;box-shadow:0 1px 3px rgba(0,0,0,0.1)}}
</style></head><body>
<h1>Continuous e-RT Analysis Report</h1>
<h2>Console Output</h2>
<pre>{}
File: {}
Patients: {} (Trt: {}, Ctrl: {})
Method: {} | {} | Burn-in: {} | Ramp: {}

e-value: {:.4}  Threshold: {:.0}
Status: {}
{}</pre>

<h2>Effect Estimates</h2>
<pre>{}: {:.3}
Mean (Trt): {:.2}  Mean (Ctrl): {:.2}  SD: {:.2}
{}
{}</pre>

<h2>Visualizations</h2>
<div class="plot-container">
<div class="plot"><div id="p1" style="height:350px"></div></div>
<div class="plot"><div id="p2" style="height:350px"></div></div>
</div>

<script>
var x={:?};var y={:?};var threshold={};
Plotly.newPlot('p1',[{{type:'scatter',mode:'lines',x:x,y:y,line:{{color:'#3498db',width:2}},name:'e-value'}}],{{
  title:'e-Value Trajectory',yaxis:{{type:'log',title:'e-value',range:[-0.5,2]}},xaxis:{{title:'Patients'}},
  shapes:[{}]}});
var support=y.map(e=>Math.log(e));
Plotly.newPlot('p2',[{{type:'scatter',mode:'lines',x:x,y:support,line:{{color:'#3498db',width:2}},name:'Support'}}],{{
  title:'Support (ln e-value)',yaxis:{{title:'ln(e-value)'}},xaxis:{{title:'Patients'}},
  shapes:[{{type:'line',x0:0,x1:1,xref:'paper',y0:{:.4},y1:{:.4},line:{{color:'#27ae60',width:2,dash:'dash'}}}}]}});
</script>
</body></html>"#,
        chrono_lite(), csv_path, r.n_total, r.n_trt, r.n_ctrl,
        method_name, method_params, burn_in, ramp,
        r.final_evalue, threshold, if r.crossed { format!("CROSSED at {}", r.crossed_at.unwrap()) } else { "Did not cross".into() },
        cross_html,
        effect_label, r.final_effect, r.final_mean_trt, r.final_mean_ctrl, r.final_sd,
        type_m_html, futility_html,
        (0..=r.n_total).collect::<Vec<_>>(), r.trajectory, threshold, shapes, threshold.ln(), threshold.ln()
    )
}
