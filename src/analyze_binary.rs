//! Analyze real binary trial data from CSV

use std::error::Error;
use std::fs::File;
use std::io::Write;
use csv::ReaderBuilder;
use serde::Deserialize;

use crate::ert_core::{
    get_input, get_input_usize, get_bool, get_string, chrono_lite, BinaryERTProcess,
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
    risk_diff_at_cross: Option<f64>,
    rd_ci_at_cross: Option<(f64, f64)>,       // Anytime-valid CI
    or_at_cross: Option<(f64, f64, f64)>,
    or_ci_at_cross: Option<(f64, f64)>,       // Anytime-valid CI
    final_risk_diff: f64,
    final_rd_ci: (f64, f64),                  // Anytime-valid CI
    final_or: (f64, f64, f64),
    final_or_ci: (f64, f64),                  // Anytime-valid CI
    rate_trt: f64,
    rate_ctrl: f64,
    type_m: Option<f64>,
    final_evalue: f64,
    trajectory: Vec<f64>,
}

// === CLI ===

pub fn run_cli(csv_path: &str, opts: &crate::AnalyzeOptions) -> Result<(), Box<dyn Error>> {
    let burn_in = opts.burn_in.unwrap_or(50);
    let ramp = opts.ramp.unwrap_or(100);
    let threshold = opts.threshold.unwrap_or(20.0);

    // CSV output mode
    if opts.csv_output {
        let data = read_csv(csv_path)?;
        if data.is_empty() {
            eprintln!("Error: No valid rows in CSV.");
            return Ok(());
        }
        let result = analyze(&data, burn_in, ramp, threshold);
        print_csv(&result, csv_path);
        return Ok(());
    }

    // Normal interactive output
    println!("\n==========================================");
    println!("   ANALYZE BINARY TRIAL DATA");
    println!("==========================================\n");

    println!("Reading {}...", csv_path);
    let data = read_csv(csv_path)?;
    if data.is_empty() {
        println!("Error: No valid rows in CSV.");
        return Ok(());
    }

    let n_total = data.len();
    let n_trt: usize = data.iter().filter(|&(t, _)| *t == 1).count();
    let n_ctrl = n_total - n_trt;
    let events_trt: usize = data.iter().filter(|&(t, o)| *t == 1 && *o == 1).count();
    let events_ctrl: usize = data.iter().filter(|&(t, o)| *t == 0 && *o == 1).count();

    println!("\n--- Data Summary ---");
    println!("Total:      {}", n_total);
    println!("Treatment:  {} ({} events, {:.1}%)", n_trt, events_trt,
             if n_trt > 0 { events_trt as f64 / n_trt as f64 * 100.0 } else { 0.0 });
    println!("Control:    {} ({} events, {:.1}%)", n_ctrl, events_ctrl,
             if n_ctrl > 0 { events_ctrl as f64 / n_ctrl as f64 * 100.0 } else { 0.0 });

    println!("\n--- Parameters ---");
    println!("Burn-in: {}  Ramp: {}  Threshold: {}", burn_in, ramp, threshold);

    println!("\n--- Running Analysis ---");
    let result = analyze(&data, burn_in, ramp, threshold);

    print_results(&result, threshold);

    if opts.generate_report {
        let html = build_report(&result, csv_path, burn_in, ramp, threshold);
        File::create("binary_analysis_report.html")?.write_all(html.as_bytes())?;
        println!("\n>> Saved: binary_analysis_report.html");
    }

    Ok(())
}

// === INTERACTIVE ===

pub fn run() -> Result<(), Box<dyn Error>> {
    println!("\n==========================================");
    println!("   ANALYZE BINARY TRIAL DATA");
    println!("==========================================\n");

    let csv_path = get_string("Path to CSV file: ");
    println!("\nReading {}...", csv_path);

    let data = read_csv(&csv_path)?;
    if data.is_empty() {
        println!("Error: No valid rows in CSV.");
        return Ok(());
    }

    let n_total = data.len();
    let n_trt: usize = data.iter().filter(|&(t, _)| *t == 1).count();
    let n_ctrl = n_total - n_trt;
    let events_trt: usize = data.iter().filter(|&(t, o)| *t == 1 && *o == 1).count();
    let events_ctrl: usize = data.iter().filter(|&(t, o)| *t == 0 && *o == 1).count();

    println!("\n--- Data Summary ---");
    println!("Total:      {}", n_total);
    println!("Treatment:  {} ({} events, {:.1}%)", n_trt, events_trt,
             if n_trt > 0 { events_trt as f64 / n_trt as f64 * 100.0 } else { 0.0 });
    println!("Control:    {} ({} events, {:.1}%)", n_ctrl, events_ctrl,
             if n_ctrl > 0 { events_ctrl as f64 / n_ctrl as f64 * 100.0 } else { 0.0 });

    println!("\n--- Parameters ---");
    let burn_in = get_input_usize("Burn-in (default 50): ");
    let ramp = get_input_usize("Ramp (default 100): ");
    let threshold: f64 = get_input("Success threshold (default 20): ");

    println!("\n--- Running Analysis ---");
    let result = analyze(&data, burn_in, ramp, threshold);

    // Print results
    print_results(&result, threshold);

    if get_bool("\nGenerate HTML report?") {
        let html = build_report(&result, &csv_path, burn_in, ramp, threshold);
        File::create("binary_analysis_report.html")?.write_all(html.as_bytes())?;
        println!("\n>> binary_analysis_report.html");
    }

    Ok(())
}

// === CSV PARSING ===

fn read_csv(path: &str) -> Result<Vec<(u8, u8)>, Box<dyn Error>> {
    let mut reader = ReaderBuilder::new().has_headers(true).flexible(true).from_path(path)?;
    let mut data = Vec::new();
    let mut skipped = 0;

    for result in reader.deserialize() {
        let row: CsvRowRaw = result?;
        let treatment = match row.treatment.parse::<u8>() {
            Ok(v) if v <= 1 => v,
            _ => { skipped += 1; continue; }
        };
        let outcome = match row.outcome.parse::<u8>() {
            Ok(v) if v <= 1 => v,
            _ => { skipped += 1; continue; }
        };
        data.push((treatment, outcome));
    }

    if skipped > 0 { println!("  (Skipped {} invalid rows)", skipped); }
    Ok(data)
}

// === ANALYSIS ===

fn analyze(
    data: &[(u8, u8)], burn_in: usize, ramp: usize, threshold: f64,
) -> AnalysisResult {
    let n_total = data.len();
    let mut proc = BinaryERTProcess::new(burn_in, ramp);
    let mut trajectory = vec![1.0];

    let mut crossed = false;
    let mut crossed_at = None;
    let mut risk_diff_at_cross = None;
    let mut rd_ci_at_cross = None;
    let mut or_at_cross = None;
    let mut or_ci_at_cross = None;

    for (i, &(treatment, outcome)) in data.iter().enumerate() {
        let patient = i + 1;
        proc.update(patient, outcome as f64, treatment == 1);
        trajectory.push(proc.wealth);

        if !crossed && proc.wealth >= threshold {
            crossed = true;
            crossed_at = Some(patient);
            risk_diff_at_cross = Some(proc.current_risk_diff());
            rd_ci_at_cross = Some(proc.confidence_sequence_rd(0.05));
            or_at_cross = Some(proc.current_odds_ratio());
            or_ci_at_cross = Some(proc.confidence_sequence_or(0.05));
        }
    }

    let final_rd = proc.current_risk_diff();
    let final_rd_ci = proc.confidence_sequence_rd(0.05);
    let final_or = proc.current_odds_ratio();
    let final_or_ci = proc.confidence_sequence_or(0.05);
    let (rate_trt, rate_ctrl) = proc.get_rates();
    let (n_trt, n_ctrl) = proc.get_ns();

    let type_m = if crossed {
        let rd_c = risk_diff_at_cross.unwrap().abs();
        let rd_f = final_rd.abs();
        if rd_f > 0.0 { Some(rd_c / rd_f) } else { None }
    } else { None };

    AnalysisResult {
        n_total, n_trt, n_ctrl, crossed, crossed_at,
        risk_diff_at_cross, rd_ci_at_cross, or_at_cross, or_ci_at_cross,
        final_risk_diff: final_rd, final_rd_ci, final_or, final_or_ci,
        rate_trt, rate_ctrl, type_m,
        final_evalue: proc.wealth, trajectory,
    }
}

// === CONSOLE OUTPUT ===

fn print_results(r: &AnalysisResult, threshold: f64) {
    println!("\n==========================================");
    println!("   RESULTS");
    println!("==========================================");

    println!("\n--- e-Value ---");
    println!("Final:     {:.4}", r.final_evalue);
    println!("Threshold: {:.1}", threshold);
    if r.crossed {
        println!("Status:    CROSSED at patient {}", r.crossed_at.unwrap());
    } else {
        println!("Status:    Did not cross");
    }

    println!("\n--- Effect Sizes ---");
    if r.crossed {
        let (or, _, _) = r.or_at_cross.unwrap();
        let (rd_lo, rd_hi) = r.rd_ci_at_cross.unwrap();
        let (or_lo, or_hi) = r.or_ci_at_cross.unwrap();
        println!("At crossing ({}):", r.crossed_at.unwrap());
        println!("  RD:  {:.1}% (95% CI: {:.1}% to {:.1}%)",
                 r.risk_diff_at_cross.unwrap() * 100.0, rd_lo * 100.0, rd_hi * 100.0);
        println!("  OR:  {:.3} (95% CI: {:.3} to {:.3})", or, or_lo, or_hi);
    }
    let (or, _, _) = r.final_or;
    let (rd_lo, rd_hi) = r.final_rd_ci;
    let (or_lo, or_hi) = r.final_or_ci;
    println!("Final ({}):", r.n_total);
    println!("  RD:  {:.1}% (95% CI: {:.1}% to {:.1}%)",
             r.final_risk_diff * 100.0, rd_lo * 100.0, rd_hi * 100.0);
    println!("  OR:  {:.3} (95% CI: {:.3} to {:.3})", or, or_lo, or_hi);
    println!("\n  (CIs are anytime-valid confidence sequences)");

    // Bidirectional testing warning
    if r.crossed && r.risk_diff_at_cross.unwrap() < 0.0 {
        println!("\n⚠️  WARNING: Effect favors CONTROL (RD < 0).");
        println!("    The e-value crossed threshold but treatment appears HARMFUL.");
        println!("    This is evidence AGAINST treatment benefit.");
    }

    if let Some(tm) = r.type_m { println!("\nType M: {:.2}x", tm); }

    println!("\n--- Rates ---");
    println!("Treatment: {:.1}% (n={})", r.rate_trt * 100.0, r.n_trt);
    println!("Control:   {:.1}% (n={})", r.rate_ctrl * 100.0, r.n_ctrl);
}

// === CSV OUTPUT ===

fn print_csv(r: &AnalysisResult, file: &str) {
    let (or, _, _) = r.final_or;
    let (rd_lo, rd_hi) = r.final_rd_ci;
    let (or_lo, or_hi) = r.final_or_ci;
    // Header
    println!("file,n_total,n_trt,n_ctrl,crossed,crossed_at,evalue,rd,rd_ci_lo,rd_ci_hi,or,or_ci_lo,or_ci_hi,type_m");
    // Data row
    println!("{},{},{},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{}",
        file,
        r.n_total,
        r.n_trt,
        r.n_ctrl,
        r.crossed,
        r.crossed_at.map_or("".to_string(), |v| v.to_string()),
        r.final_evalue,
        r.final_risk_diff,
        rd_lo,
        rd_hi,
        or,
        or_lo,
        or_hi,
        r.type_m.map_or("".to_string(), |v| format!("{:.2}", v)),
    );
}

// === HTML REPORT ===

fn build_report(r: &AnalysisResult, csv_path: &str, burn_in: usize, ramp: usize, threshold: f64) -> String {
    let x: Vec<usize> = (0..=r.n_total).collect();
    let (or, _, _) = r.final_or;
    let (rd_lo, rd_hi) = r.final_rd_ci;
    let (or_lo, or_hi) = r.final_or_ci;

    let status = if r.crossed {
        format!("<span style='color:green'>CROSSED at {}</span>", r.crossed_at.unwrap())
    } else { "<span style='color:gray'>Did not cross</span>".into() };

    let crossing = if r.crossed {
        let (c_or, _, _) = r.or_at_cross.unwrap();
        let (c_rd_lo, c_rd_hi) = r.rd_ci_at_cross.unwrap();
        let (c_or_lo, c_or_hi) = r.or_ci_at_cross.unwrap();
        format!(r#"<tr><td>At crossing:</td><td>RD {:.1}% (95% CI: {:.1}% to {:.1}%)<br>OR {:.3} (95% CI: {:.3} to {:.3})</td></tr>"#,
                r.risk_diff_at_cross.unwrap() * 100.0, c_rd_lo * 100.0, c_rd_hi * 100.0,
                c_or, c_or_lo, c_or_hi)
    } else { String::new() };

    let type_m = r.type_m.map_or(String::new(), |t| format!("<tr><td>Type M:</td><td>{:.2}x</td></tr>", t));

    format!(r#"<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Binary Analysis Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
body{{font-family:system-ui,-apple-system,sans-serif;max-width:1400px;margin:0 auto;padding:20px;background:#fafafa}}
h1{{color:#1a1a2e}}h2{{color:#16213e;border-bottom:1px solid #ddd;padding-bottom:5px}}
table{{margin:10px 0}}td{{padding:4px 12px}}
.plot{{background:#fff;border-radius:8px;padding:10px;margin:20px 0;box-shadow:0 1px 3px rgba(0,0,0,0.1)}}
.note{{font-size:0.9em;color:#666;margin-top:10px}}
</style></head><body>
<h1>Binary e-RT Analysis Report</h1>
<p>{}</p>

<h2>Data</h2>
<table>
<tr><td>File:</td><td>{}</td></tr>
<tr><td>Total:</td><td>{}</td></tr>
<tr><td>Treatment:</td><td>{} ({:.1}%)</td></tr>
<tr><td>Control:</td><td>{} ({:.1}%)</td></tr>
</table>

<h2>Parameters</h2>
<table>
<tr><td>Burn-in:</td><td>{}</td></tr>
<tr><td>Ramp:</td><td>{}</td></tr>
<tr><td>Threshold:</td><td>{}</td></tr>
</table>

<h2>Results</h2>
<table>
<tr><td>Final e-value:</td><td><strong>{:.4}</strong></td></tr>
<tr><td>Status:</td><td>{}</td></tr>
{}
<tr><td>Final:</td><td>RD {:.1}% (95% CI: {:.1}% to {:.1}%)<br>OR {:.3} (95% CI: {:.3} to {:.3})</td></tr>
{}
</table>
<p class="note">CIs are anytime-valid confidence sequences.</p>

<h2>e-Value Trajectory</h2>
<div class="plot"><div id="p1" style="height:400px"></div></div>

<script>
Plotly.newPlot('p1',[{{type:'scatter',mode:'lines',x:{:?},y:{:?},line:{{color:'steelblue',width:2}}}}],{{
yaxis:{{type:'log',title:'e-value'}},xaxis:{{title:'Patients'}},
shapes:[{{type:'line',x0:0,x1:1,xref:'paper',y0:{},y1:{},line:{{color:'green',dash:'dash',width:2}}}}]}});
</script>
</body></html>"#,
        chrono_lite(), csv_path, r.n_total, r.n_trt, r.rate_trt * 100.0, r.n_ctrl, r.rate_ctrl * 100.0,
        burn_in, ramp, threshold,
        r.final_evalue, status, crossing,
        r.final_risk_diff * 100.0, rd_lo * 100.0, rd_hi * 100.0, or, or_lo, or_hi, type_m,
        x, r.trajectory, threshold, threshold)
}
