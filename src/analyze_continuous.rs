//! Analyze real continuous trial data from CSV (e-RTc)

use std::error::Error;
use std::fs::File;
use std::io::Write;
use csv::ReaderBuilder;
use serde::Deserialize;

use crate::ert_core::{
    get_input, get_input_usize, get_bool, get_string, chrono_lite, MADProcess,
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

struct AnalysisResult {
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
}

// === CLI ===

pub fn run_cli(csv_path: &str, opts: &crate::AnalyzeOptions) -> Result<(), Box<dyn Error>> {
    println!("\n==========================================");
    println!("   ANALYZE CONTINUOUS TRIAL DATA (e-RTc)");
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

    let burn_in = opts.burn_in.unwrap_or(50);
    let ramp = opts.ramp.unwrap_or(100);
    let threshold = opts.threshold.unwrap_or(20.0);
    let c_max = 0.6;

    println!("\n--- Parameters ---");
    println!("Burn-in: {}  Ramp: {}  Threshold: {}  c_max: {:.2}", burn_in, ramp, threshold, c_max);

    println!("\n--- Running Analysis ---");
    let result = analyze(&data, burn_in, ramp, threshold, c_max, pooled_sd);

    print_results(&result, threshold);

    if opts.generate_report {
        let html = build_html(&result, csv_path, burn_in, ramp, threshold, c_max);
        File::create("continuous_analysis_report.html")?.write_all(html.as_bytes())?;
        println!("\n>> Saved: continuous_analysis_report.html");
    }

    Ok(())
}

// === INTERACTIVE ===

pub fn run() -> Result<(), Box<dyn Error>> {
    println!("\n==========================================");
    println!("   ANALYZE CONTINUOUS TRIAL DATA (e-RTc)");
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

    println!("\n--- Parameters ---");
    let burn_in = get_input_usize("Burn-in (default 50): ");
    let ramp = get_input_usize("Ramp (default 100): ");
    let threshold = get_input("Success threshold (default 20): ");
    let c_max = get_input("c_max (default 0.6): ");

    println!("\n--- Running Analysis ---");
    let result = analyze(&data, burn_in, ramp, threshold, c_max, pooled_sd);

    print_results(&result, threshold);

    if get_bool("\nGenerate HTML report?") {
        let html = build_html(&result, &csv_path, burn_in, ramp, threshold, c_max);
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

// === ANALYSIS ===

fn analyze(data: &[(u8, f64)], burn_in: usize, ramp: usize, threshold: f64, c_max: f64, pooled_sd: f64) -> AnalysisResult {
    let n_total = data.len();
    let mut trajectory = vec![1.0];
    let mut crossed = false;
    let mut crossed_at = None;
    let mut effect_at_cross = None;

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
    }

    let final_effect = proc.current_effect(proc.get_pooled_sd());
    let (final_mean_trt, final_mean_ctrl) = proc.get_means();
    let final_sd = proc.get_pooled_sd();
    let (n_trt, n_ctrl) = proc.get_ns();
    let final_evalue = proc.wealth;

    let type_m = if crossed {
        let ec = effect_at_cross.unwrap().abs();
        let ef = final_effect.abs();
        if ef > 0.0 { Some(ec / ef) } else { None }
    } else { None };

    AnalysisResult {
        n_total, n_trt, n_ctrl, crossed, crossed_at, effect_at_cross,
        final_effect, final_mean_trt, final_mean_ctrl, final_sd, final_evalue, type_m, trajectory,
    }
}

// === CONSOLE OUTPUT ===

fn print_results(r: &AnalysisResult, threshold: f64) {
    println!("\n=== RESULTS (e-RTc) ===");
    println!("e-value: {:.4}  Threshold: {:.0}", r.final_evalue, threshold);

    if r.crossed {
        println!("Status: CROSSED at patient {}", r.crossed_at.unwrap());
    } else {
        println!("Status: Did not cross");
    }

    if r.crossed {
        println!("\nAt crossing: Cohen's d = {:.3}", r.effect_at_cross.unwrap());
    }
    println!("Final: Cohen's d = {:.3}  Trt={:.2}  Ctrl={:.2}  SD={:.2}",
             r.final_effect, r.final_mean_trt, r.final_mean_ctrl, r.final_sd);

    if let Some(tm) = r.type_m { println!("Type M: {:.2}x", tm); }
}

// === HTML REPORT ===

fn build_html(r: &AnalysisResult, csv_path: &str, burn_in: usize, ramp: usize, threshold: f64, c_max: f64) -> String {
    let cross_html = if r.crossed {
        format!("<p><strong>At crossing (patient {}):</strong> Cohen's d = {:.3}</p>", r.crossed_at.unwrap(), r.effect_at_cross.unwrap())
    } else { String::new() };

    let type_m_html = r.type_m.map_or(String::new(), |tm| format!("<p><strong>Type M:</strong> {:.2}x</p>", tm));

    format!(r#"<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>e-RTc Analysis Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
body{{font-family:system-ui,-apple-system,sans-serif;max-width:1200px;margin:0 auto;padding:20px;background:#fafafa}}
h1{{color:#1a1a2e}}h2,h3{{color:#16213e}}
pre{{background:#fff;padding:15px;border-radius:8px;border:1px solid #ddd;overflow-x:auto;font-size:13px}}
.plot{{background:#fff;border-radius:8px;padding:10px;box-shadow:0 1px 3px rgba(0,0,0,0.1);margin:20px 0}}
</style></head><body>
<h1>e-RTc Analysis Report</h1>
<h2>Console Output</h2>
<pre>{}
File: {}
Patients: {} (Trt: {}, Ctrl: {})
Burn-in: {} | Ramp: {} | c_max: {:.2}

e-value: {:.4}  Threshold: {:.0}
Status: {}
{}</pre>

<h2>Effect Estimates</h2>
<pre>Cohen's d: {:.3}
Mean (Trt): {:.2}  Mean (Ctrl): {:.2}  SD: {:.2}
{}</pre>

<h2>e-Value Trajectory</h2>
<div class="plot"><div id="p1" style="height:400px"></div></div>

<h2>Support (ln e-value)</h2>
<div class="plot"><div id="p2" style="height:400px"></div></div>

<script>
var x={:?};var y={:?};var threshold={};
Plotly.newPlot('p1',[{{type:'scatter',mode:'lines',x:x,y:y,line:{{color:'#3498db',width:2}},name:'e-value'}}],{{
  yaxis:{{type:'log',title:'e-value',range:[-0.5,2]}},xaxis:{{title:'Patients'}},
  shapes:[{{type:'line',x0:0,x1:1,xref:'paper',y0:{},y1:{},line:{{color:'#27ae60',width:2,dash:'dash'}}}}]}});
var support=y.map(e=>Math.log(e));
Plotly.newPlot('p2',[{{type:'scatter',mode:'lines',x:x,y:support,line:{{color:'#3498db',width:2}},name:'Support'}}],{{
  yaxis:{{title:'ln(e-value)'}},xaxis:{{title:'Patients'}},
  shapes:[{{type:'line',x0:0,x1:1,xref:'paper',y0:{:.4},y1:{:.4},line:{{color:'#27ae60',width:2,dash:'dash'}}}}]}});
</script>
</body></html>"#,
        chrono_lite(), csv_path, r.n_total, r.n_trt, r.n_ctrl,
        burn_in, ramp, c_max,
        r.final_evalue, threshold, if r.crossed { format!("CROSSED at {}", r.crossed_at.unwrap()) } else { "Did not cross".into() },
        cross_html,
        r.final_effect, r.final_mean_trt, r.final_mean_ctrl, r.final_sd,
        type_m_html,
        (0..=r.n_total).collect::<Vec<_>>(), r.trajectory, threshold, threshold, threshold, threshold.ln(), threshold.ln()
    )
}
