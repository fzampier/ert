//! Analyze real binary trial data from CSV

use std::error::Error;
use std::fs::File;
use std::io::Write;
use csv::ReaderBuilder;
use serde::Deserialize;
// rand no longer needed - FutilityMonitor uses internal RNG

use crate::ert_core::{
    get_input, get_input_usize, get_bool, get_string, chrono_lite, BinaryERTProcess,
    FutilityMonitor, FutilityConfig,
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

#[derive(Clone)]
struct DesignParams {
    control_rate: f64,
    _treatment_rate: f64,
    design_arr: f64,
}

struct AnalysisResult {
    n_total: usize,
    n_trt: usize,
    n_ctrl: usize,
    crossed: bool,
    crossed_at: Option<usize>,
    risk_diff_at_cross: Option<f64>,
    or_at_cross: Option<(f64, f64, f64)>,
    final_risk_diff: f64,
    final_or: (f64, f64, f64),
    rate_trt: f64,
    rate_ctrl: f64,
    type_m: Option<f64>,
    final_evalue: f64,
    trajectory: Vec<f64>,
    futility_summary: Option<String>,
    futility_recommended_stop: bool,
    futility_worst_ratio: Option<f64>,
    design: Option<DesignParams>,
}

// === CLI ===

pub fn run_cli(csv_path: &str, opts: &crate::AnalyzeOptions) -> Result<(), Box<dyn Error>> {
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

    let burn_in = opts.burn_in.unwrap_or(50);
    let ramp = opts.ramp.unwrap_or(100);
    let threshold = opts.threshold.unwrap_or(20.0);

    println!("\n--- Parameters ---");
    println!("Burn-in: {}  Ramp: {}  Threshold: {}", burn_in, ramp, threshold);

    println!("\n--- Running Analysis ---");
    let result = analyze(&data, burn_in, ramp, threshold, None, None);

    print_results(&result, threshold, None);

    if opts.generate_report {
        let html = build_report(&result, csv_path, burn_in, ramp, threshold, None);
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

    let (fut_thresh, design) = if get_bool("Enable futility monitoring?") {
        let fut: f64 = get_input("Futility threshold (e.g., 0.5): ");
        println!("\n--- Design Assumptions ---");
        let p_ctrl: f64 = get_input("Design control rate (e.g., 0.30): ");
        let p_trt: f64 = get_input("Design treatment rate (e.g., 0.20): ");
        let arr = (p_ctrl - p_trt).abs();
        println!("Design ARR: {:.1}%", arr * 100.0);
        (Some(fut), Some(DesignParams { control_rate: p_ctrl, _treatment_rate: p_trt, design_arr: arr }))
    } else {
        (None, None)
    };

    println!("\n--- Running Analysis ---");
    let result = analyze(&data, burn_in, ramp, threshold, fut_thresh, design);

    // Print results
    print_results(&result, threshold, fut_thresh);

    if get_bool("\nGenerate HTML report?") {
        let html = build_report(&result, &csv_path, burn_in, ramp, threshold, fut_thresh);
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
    _fut_thresh: Option<f64>, design: Option<DesignParams>,
) -> AnalysisResult {
    let n_total = data.len();
    let mut proc = BinaryERTProcess::new(burn_in, ramp);
    let mut trajectory = vec![1.0];

    let mut crossed = false;
    let mut crossed_at = None;
    let mut risk_diff_at_cross = None;
    let mut or_at_cross = None;

    // Create FutilityMonitor if design params available
    let mut fut_monitor = design.as_ref().map(|d| {
        FutilityMonitor::new(
            FutilityConfig::default(),
            d.design_arr,
            d.control_rate,
            n_total,
            threshold,
            burn_in,
            ramp,
        )
    });

    for (i, &(treatment, outcome)) in data.iter().enumerate() {
        let patient = i + 1;
        proc.update(patient, outcome as f64, treatment == 1);
        trajectory.push(proc.wealth);

        if !crossed && proc.wealth >= threshold {
            crossed = true;
            crossed_at = Some(patient);
            risk_diff_at_cross = Some(proc.current_risk_diff());
            or_at_cross = Some(proc.current_odds_ratio());
        }

        // Futility monitoring - pass current process state for accurate forward simulation
        if let Some(ref mut monitor) = fut_monitor {
            if monitor.should_check(patient) {
                monitor.check(patient, proc.wealth, proc.n_trt, proc.events_trt, proc.n_ctrl, proc.events_ctrl);
            }
        }
    }

    let final_rd = proc.current_risk_diff();
    let final_or = proc.current_odds_ratio();
    let (rate_trt, rate_ctrl) = proc.get_rates();
    let (n_trt, n_ctrl) = proc.get_ns();

    let type_m = if crossed {
        let rd_c = risk_diff_at_cross.unwrap().abs();
        let rd_f = final_rd.abs();
        if rd_f > 0.0 { Some(rd_c / rd_f) } else { None }
    } else { None };

    // Extract futility results
    let (fut_summary, fut_stop, fut_ratio) = if let Some(monitor) = fut_monitor {
        (Some(monitor.summary()), monitor.ever_recommended_stop(), monitor.worst_ratio())
    } else {
        (None, false, None)
    };

    AnalysisResult {
        n_total, n_trt, n_ctrl, crossed, crossed_at, risk_diff_at_cross, or_at_cross,
        final_risk_diff: final_rd, final_or, rate_trt, rate_ctrl, type_m,
        final_evalue: proc.wealth, trajectory,
        futility_summary: fut_summary,
        futility_recommended_stop: fut_stop,
        futility_worst_ratio: fut_ratio,
        design,
    }
}

// === CONSOLE OUTPUT ===

fn print_results(r: &AnalysisResult, threshold: f64, fut_thresh: Option<f64>) {
    println!("\n==========================================");
    println!("   RESULTS");
    println!("==========================================");

    println!("\n--- e-Value ---");
    println!("Final:     {:.4}", r.final_evalue);
    println!("Threshold: {:.1}", threshold);
    if r.crossed {
        println!("Status:    CROSSED at patient {}", r.crossed_at.unwrap());
    } else if let Some(f) = fut_thresh {
        println!("Status:    {}", if r.final_evalue < f { "Below futility" } else { "Did not cross" });
    } else {
        println!("Status:    Did not cross");
    }

    println!("\n--- Effect Sizes ---");
    if r.crossed {
        let (or, lo, hi) = r.or_at_cross.unwrap();
        println!("At crossing ({}):", r.crossed_at.unwrap());
        println!("  RD:  {:.1}%", r.risk_diff_at_cross.unwrap() * 100.0);
        println!("  OR:  {:.3} ({:.3}-{:.3})", or, lo, hi);
    }
    let (or, lo, hi) = r.final_or;
    println!("Final ({}):", r.n_total);
    println!("  RD:  {:.1}%", r.final_risk_diff * 100.0);
    println!("  OR:  {:.3} ({:.3}-{:.3})", or, lo, hi);

    if let Some(tm) = r.type_m { println!("\nType M: {:.2}x", tm); }

    println!("\n--- Rates ---");
    println!("Treatment: {:.1}% (n={})", r.rate_trt * 100.0, r.n_trt);
    println!("Control:   {:.1}% (n={})", r.rate_ctrl * 100.0, r.n_ctrl);

    if let Some(ref d) = r.design {
        println!("\n--- Futility Monitor ---");
        println!("(Simulation-based, NOT anytime-valid)");
        println!("Design ARR:       {:.1}%", d.design_arr * 100.0);
        println!("Stop recommended: {}", if r.futility_recommended_stop { "YES" } else { "No" });
        if let Some(ratio) = r.futility_worst_ratio {
            println!("Worst ratio:      {:.2}x design", ratio);
        }
        if let Some(ref summary) = r.futility_summary {
            println!("\n{}", summary);
        }
    }
}

// === HTML REPORT ===

fn build_report(r: &AnalysisResult, csv_path: &str, burn_in: usize, ramp: usize, threshold: f64, fut_thresh: Option<f64>) -> String {
    let x: Vec<usize> = (0..=r.n_total).collect();
    let (or, lo, hi) = r.final_or;

    let status = if r.crossed {
        format!("<span style='color:green'>CROSSED at {}</span>", r.crossed_at.unwrap())
    } else { "<span style='color:gray'>Did not cross</span>".into() };

    let crossing = if r.crossed {
        let (c_or, c_lo, c_hi) = r.or_at_cross.unwrap();
        format!("<tr><td>At crossing:</td><td>RD {:.1}%, OR {:.3} ({:.3}-{:.3})</td></tr>",
                r.risk_diff_at_cross.unwrap() * 100.0, c_or, c_lo, c_hi)
    } else { String::new() };

    let type_m = r.type_m.map_or(String::new(), |t| format!("<tr><td>Type M:</td><td>{:.2}x</td></tr>", t));

    let _fut_line = fut_thresh.map_or(String::new(), |f|
        format!("{{type:'line',x0:0,x1:1,xref:'paper',y0:{},y1:{},line:{{color:'orange',dash:'dot'}}}},", f, f));

    let fut_section = if let Some(ref d) = r.design {
        format!(r#"<h2>Futility Monitor</h2>
<p><em>Simulation-based decision support, NOT anytime-valid inference.</em></p>
<table>
<tr><td>Design ARR:</td><td>{:.1}%</td></tr>
<tr><td>Stop recommended:</td><td>{}</td></tr>
<tr><td>Worst ratio:</td><td>{}</td></tr>
</table>
<pre>{}</pre>"#,
            d.design_arr * 100.0,
            if r.futility_recommended_stop { "<strong style='color:red'>YES</strong>" } else { "No" },
            r.futility_worst_ratio.map_or("N/A".into(), |w| format!("{:.2}x design", w)),
            r.futility_summary.as_deref().unwrap_or(""))
    } else { String::new() };

    format!(r#"<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Binary Analysis Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
body{{font-family:system-ui,-apple-system,sans-serif;max-width:1400px;margin:0 auto;padding:20px;background:#fafafa}}
h1{{color:#1a1a2e}}h2{{color:#16213e;border-bottom:1px solid #ddd;padding-bottom:5px}}
table{{margin:10px 0}}td{{padding:4px 12px}}
.plot{{background:#fff;border-radius:8px;padding:10px;margin:20px 0;box-shadow:0 1px 3px rgba(0,0,0,0.1)}}
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
<tr><td>Final:</td><td>RD {:.1}%, OR {:.3} ({:.3}-{:.3})</td></tr>
{}
</table>

{}

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
        r.final_evalue, status, crossing, r.final_risk_diff * 100.0, or, lo, hi, type_m, fut_section,
        x, r.trajectory, threshold, threshold)
}
