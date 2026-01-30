//! Analyze deaths-only trial data from CSV (e-RTd)
//!
//! CSV format: Each row is one death event.
//! Columns: arm (1=treatment, 0=control), time (optional, for plotting)

use std::error::Error;
use std::fs::File;
use std::io::Write;
use csv::ReaderBuilder;
use serde::Deserialize;

use crate::ert_core::{get_input, get_input_usize, get_bool, get_string, chrono_lite, report_path};
use crate::deaths_only::{DeathsOnlyERT, Arm};

// === DATA STRUCTURES ===

#[derive(Debug, Deserialize)]
struct CsvRowRaw {
    arm: String,
    #[serde(default)]
    time: Option<f64>,
}

struct DeathEvent {
    arm: Arm,
    time: Option<f64>,
}

struct AnalysisResult {
    n_deaths: usize,
    d_trt: u64,
    d_ctrl: u64,
    crossed: bool,
    crossed_at: Option<usize>,
    p_at_cross: Option<f64>,
    rr_at_cross: Option<f64>,
    p_ci_at_cross: Option<(f64, f64)>,
    rr_ci_at_cross: Option<(f64, f64)>,
    final_p: f64,
    final_rr: f64,
    final_p_ci: (f64, f64),
    final_rr_ci: (f64, f64),
    final_evalue: f64,
    trajectory: Vec<f64>,
}

// === CLI ===

pub fn run_cli(csv_path: &str, opts: &crate::AnalyzeOptions) -> Result<(), Box<dyn Error>> {
    let burn_in = opts.burn_in.unwrap_or(30);
    let ramp = opts.ramp.unwrap_or(50);
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
    println!("   ANALYZE DEATHS-ONLY DATA (e-RTd)");
    println!("==========================================\n");

    println!("Reading {}...", csv_path);
    let data = read_csv(csv_path)?;
    if data.is_empty() {
        println!("Error: No valid rows in CSV.");
        return Ok(());
    }

    let n_deaths = data.len();
    let d_trt = data.iter().filter(|d| d.arm == Arm::Treatment).count();
    let d_ctrl = n_deaths - d_trt;
    let has_times = data.iter().any(|d| d.time.is_some());

    println!("\n--- Data Summary ---");
    println!("Total deaths: {}", n_deaths);
    println!("Treatment:    {} ({:.1}%)", d_trt, d_trt as f64 / n_deaths as f64 * 100.0);
    println!("Control:      {} ({:.1}%)", d_ctrl, d_ctrl as f64 / n_deaths as f64 * 100.0);
    if has_times {
        println!("Time data:    Available");
    }

    println!("\n--- Parameters ---");
    println!("Burn-in: {}  Ramp: {}  Threshold: {}", burn_in, ramp, threshold);

    println!("\n--- Running Analysis ---");
    let result = analyze(&data, burn_in, ramp, threshold);

    print_results(&result, threshold);

    if opts.generate_report {
        let html = build_report(&result, csv_path, burn_in, ramp, threshold);
        let out_path = report_path(csv_path, "deaths_analysis_report.html");
        File::create(&out_path)?.write_all(html.as_bytes())?;
        println!("\n>> Saved: {}", out_path);
    }

    Ok(())
}

// === INTERACTIVE ===

pub fn run() -> Result<(), Box<dyn Error>> {
    println!("\n==========================================");
    println!("   ANALYZE DEATHS-ONLY DATA (e-RTd)");
    println!("==========================================\n");

    println!("CSV format: arm (1=treatment, 0=control), time (optional)");
    println!("Each row represents one death event.\n");

    let csv_path = get_string("Path to CSV file: ");
    println!("\nReading {}...", csv_path);

    let data = read_csv(&csv_path)?;
    if data.is_empty() {
        println!("Error: No valid rows in CSV.");
        return Ok(());
    }

    let n_deaths = data.len();
    let d_trt = data.iter().filter(|d| d.arm == Arm::Treatment).count();
    let d_ctrl = n_deaths - d_trt;

    println!("\n--- Data Summary ---");
    println!("Total deaths: {}", n_deaths);
    println!("Treatment:    {} ({:.1}%)", d_trt, d_trt as f64 / n_deaths as f64 * 100.0);
    println!("Control:      {} ({:.1}%)", d_ctrl, d_ctrl as f64 / n_deaths as f64 * 100.0);

    println!("\n--- Parameters ---");
    let burn_in = get_input_usize("Burn-in (default 30): ");
    let ramp = get_input_usize("Ramp (default 50): ");
    let threshold: f64 = get_input("Success threshold (default 20): ");

    println!("\n--- Running Analysis ---");
    let result = analyze(&data, burn_in, ramp, threshold);

    print_results(&result, threshold);

    if get_bool("\nGenerate HTML report?") {
        let html = build_report(&result, &csv_path, burn_in, ramp, threshold);
        let out_path = report_path(&csv_path, "deaths_analysis_report.html");
        File::create(&out_path)?.write_all(html.as_bytes())?;
        println!("\n>> Saved: {}", out_path);
    }

    Ok(())
}

// === CSV PARSING ===

fn read_csv(path: &str) -> Result<Vec<DeathEvent>, Box<dyn Error>> {
    let mut reader = ReaderBuilder::new().has_headers(true).flexible(true).from_path(path)?;
    let mut data = Vec::new();
    let mut skipped = 0;

    for result in reader.deserialize() {
        let row: CsvRowRaw = result?;
        let arm = match row.arm.trim().parse::<u8>() {
            Ok(1) => Arm::Treatment,
            Ok(0) => Arm::Control,
            _ => { skipped += 1; continue; }
        };
        data.push(DeathEvent { arm, time: row.time });
    }

    if skipped > 0 { println!("  (Skipped {} invalid rows)", skipped); }
    Ok(data)
}

// === ANALYSIS ===

fn analyze(data: &[DeathEvent], burn_in: usize, ramp: usize, threshold: f64) -> AnalysisResult {
    let n_deaths = data.len();
    let mut ert = DeathsOnlyERT::new(burn_in, ramp, threshold);
    let mut trajectory = vec![1.0];

    let mut crossed = false;
    let mut crossed_at = None;
    let mut p_at_cross = None;
    let mut rr_at_cross = None;
    let mut p_ci_at_cross = None;
    let mut rr_ci_at_cross = None;

    for (i, death) in data.iter().enumerate() {
        ert.observe(death.arm);
        trajectory.push(ert.wealth());

        if !crossed && ert.wealth() >= threshold {
            crossed = true;
            crossed_at = Some(i + 1);
            p_at_cross = Some(ert.death_proportion_trt());
            rr_at_cross = Some(ert.mortality_rate_ratio());
            p_ci_at_cross = Some(ert.confidence_sequence_p(0.05));
            rr_ci_at_cross = Some(ert.confidence_sequence_rr(0.05));
        }
    }

    let (d_trt, d_ctrl) = ert.get_counts();

    AnalysisResult {
        n_deaths,
        d_trt,
        d_ctrl,
        crossed,
        crossed_at,
        p_at_cross,
        rr_at_cross,
        p_ci_at_cross,
        rr_ci_at_cross,
        final_p: ert.death_proportion_trt(),
        final_rr: ert.mortality_rate_ratio(),
        final_p_ci: ert.confidence_sequence_p(0.05),
        final_rr_ci: ert.confidence_sequence_rr(0.05),
        final_evalue: ert.wealth(),
        trajectory,
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
        println!("Status:    CROSSED at death {}", r.crossed_at.unwrap());
    } else {
        println!("Status:    Did not cross");
    }

    println!("\n--- Death Counts ---");
    println!("Treatment: {} ({:.1}%)", r.d_trt, r.d_trt as f64 / r.n_deaths as f64 * 100.0);
    println!("Control:   {} ({:.1}%)", r.d_ctrl, r.d_ctrl as f64 / r.n_deaths as f64 * 100.0);

    println!("\n--- Effect Estimates ---");
    println!("p = P(death from treatment | death occurred)");
    println!("RR = Mortality Rate Ratio = p / (1 - p)");
    println!("Under null: p = 0.50, RR = 1.00");
    println!();

    if r.crossed {
        let (p_lo, p_hi) = r.p_ci_at_cross.unwrap();
        let (rr_lo, rr_hi) = r.rr_ci_at_cross.unwrap();
        println!("At crossing ({}):", r.crossed_at.unwrap());
        println!("  p:  {:.3} (95% CI: {:.3} to {:.3})", r.p_at_cross.unwrap(), p_lo, p_hi);
        println!("  RR: {:.3} (95% CI: {:.3} to {:.3})", r.rr_at_cross.unwrap(), rr_lo, rr_hi);
    }

    let (p_lo, p_hi) = r.final_p_ci;
    let (rr_lo, rr_hi) = r.final_rr_ci;
    println!("Final ({} deaths):", r.n_deaths);
    println!("  p:  {:.3} (95% CI: {:.3} to {:.3})", r.final_p, p_lo, p_hi);
    println!("  RR: {:.3} (95% CI: {:.3} to {:.3})", r.final_rr, rr_lo, rr_hi);

    println!("\n  (CIs are anytime-valid confidence sequences)");

    // Interpretation
    println!("\n--- Interpretation ---");
    if r.final_p < 0.5 {
        let reduction = (1.0 - r.final_rr) * 100.0;
        println!("Fewer deaths in treatment arm (p < 0.5).");
        println!("Estimated {:.0}% relative reduction in mortality rate.", reduction);
        if r.crossed {
            println!("Evidence threshold CROSSED - reject null hypothesis.");
        }
    } else if r.final_p > 0.5 {
        let increase = (r.final_rr - 1.0) * 100.0;
        println!("MORE deaths in treatment arm (p > 0.5).");
        println!("Estimated {:.0}% relative INCREASE in mortality rate.", increase);
        if r.crossed {
            println!("WARNING: Evidence of HARM - treatment increases mortality.");
        }
    } else {
        println!("Deaths split evenly between arms (p ≈ 0.5).");
    }
}

// === CSV OUTPUT ===

fn print_csv(r: &AnalysisResult, file: &str) {
    let (p_lo, p_hi) = r.final_p_ci;
    let (rr_lo, rr_hi) = r.final_rr_ci;
    // Header
    println!("file,n_deaths,d_trt,d_ctrl,crossed,crossed_at,evalue,p,p_ci_lo,p_ci_hi,rr,rr_ci_lo,rr_ci_hi");
    // Data row
    println!("{},{},{},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
        file,
        r.n_deaths,
        r.d_trt,
        r.d_ctrl,
        r.crossed,
        r.crossed_at.map_or("".to_string(), |v| v.to_string()),
        r.final_evalue,
        r.final_p,
        p_lo,
        p_hi,
        r.final_rr,
        rr_lo,
        rr_hi,
    );
}

// === HTML REPORT ===

fn build_report(r: &AnalysisResult, csv_path: &str, burn_in: usize, ramp: usize, threshold: f64) -> String {
    let x: Vec<usize> = (0..=r.n_deaths).collect();
    let (p_lo, p_hi) = r.final_p_ci;
    let (rr_lo, rr_hi) = r.final_rr_ci;

    let status = if r.crossed {
        format!("<span style='color:green'>CROSSED at death {}</span>", r.crossed_at.unwrap())
    } else {
        "<span style='color:gray'>Did not cross</span>".into()
    };

    let crossing = if r.crossed {
        let (c_p_lo, c_p_hi) = r.p_ci_at_cross.unwrap();
        let (c_rr_lo, c_rr_hi) = r.rr_ci_at_cross.unwrap();
        format!(r#"<tr><td>At crossing:</td><td>p = {:.3} (95% CI: {:.3} to {:.3})<br>RR = {:.3} (95% CI: {:.3} to {:.3})</td></tr>"#,
                r.p_at_cross.unwrap(), c_p_lo, c_p_hi,
                r.rr_at_cross.unwrap(), c_rr_lo, c_rr_hi)
    } else { String::new() };

    let interpretation = if r.final_p < 0.5 {
        let reduction = (1.0 - r.final_rr) * 100.0;
        format!("Fewer deaths in treatment arm. Estimated {:.0}% relative reduction.", reduction)
    } else if r.final_p > 0.5 {
        let increase = (r.final_rr - 1.0) * 100.0;
        format!("<span style='color:red'>MORE deaths in treatment arm. Estimated {:.0}% relative INCREASE.</span>", increase)
    } else {
        "Deaths split evenly between arms.".into()
    };

    format!(r#"<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Deaths-Only Analysis Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
body{{font-family:system-ui,-apple-system,sans-serif;max-width:1400px;margin:0 auto;padding:20px;background:#fafafa}}
h1{{color:#1a1a2e}}h2{{color:#16213e;border-bottom:1px solid #ddd;padding-bottom:5px}}
table{{margin:10px 0}}td{{padding:4px 12px}}
.plot{{background:#fff;border-radius:8px;padding:10px;margin:20px 0;box-shadow:0 1px 3px rgba(0,0,0,0.1)}}
.note{{font-size:0.9em;color:#666;margin-top:10px}}
</style></head><body>
<h1>e-RTd Deaths-Only Analysis Report</h1>
<p>{}</p>

<h2>Data</h2>
<table>
<tr><td>File:</td><td>{}</td></tr>
<tr><td>Total deaths:</td><td>{}</td></tr>
<tr><td>Treatment:</td><td>{} ({:.1}%)</td></tr>
<tr><td>Control:</td><td>{} ({:.1}%)</td></tr>
</table>

<h2>Parameters</h2>
<table>
<tr><td>Burn-in:</td><td>{} deaths</td></tr>
<tr><td>Ramp:</td><td>{} deaths</td></tr>
<tr><td>Threshold:</td><td>{}</td></tr>
</table>

<h2>Results</h2>
<table>
<tr><td>Final e-value:</td><td><strong>{:.4}</strong></td></tr>
<tr><td>Status:</td><td>{}</td></tr>
{}
<tr><td>Final:</td><td>p = {:.3} (95% CI: {:.3} to {:.3})<br>RR = {:.3} (95% CI: {:.3} to {:.3})</td></tr>
</table>
<p><strong>Interpretation:</strong> {}</p>
<p class="note">p = P(death from treatment | death). Under null, p = 0.5.<br>
RR = Mortality Rate Ratio = p / (1-p). Under null, RR = 1.0.<br>
CIs are anytime-valid confidence sequences.</p>

<h2>e-Value Trajectory</h2>
<div class="plot"><div id="p1" style="height:400px"></div></div>

<script>
Plotly.newPlot('p1',[{{type:'scatter',mode:'lines',x:{:?},y:{:?},line:{{color:'steelblue',width:2}}}}],{{
yaxis:{{type:'log',title:'e-value'}},xaxis:{{title:'Deaths'}},
shapes:[{{type:'line',x0:0,x1:1,xref:'paper',y0:{},y1:{},line:{{color:'green',dash:'dash',width:2}}}}]}});
</script>
</body></html>"#,
        chrono_lite(),
        csv_path,
        r.n_deaths,
        r.d_trt, r.d_trt as f64 / r.n_deaths as f64 * 100.0,
        r.d_ctrl, r.d_ctrl as f64 / r.n_deaths as f64 * 100.0,
        burn_in, ramp, threshold,
        r.final_evalue, status, crossing,
        r.final_p, p_lo, p_hi,
        r.final_rr, rr_lo, rr_hi,
        interpretation,
        x, r.trajectory, threshold, threshold)
}
