//! Analyze real binary trial data from CSV

use std::error::Error;
use std::fs::File;
use std::io::Write;
use csv::ReaderBuilder;
use serde::Deserialize;

use crate::ert_core::{
    get_input, get_input_usize, get_bool, get_string,
    chrono_lite, BinaryERTProcess,
};

/// Raw CSV row (may have NA values)
#[derive(Debug, Deserialize)]
struct CsvRowRaw {
    #[serde(default)]
    #[allow(dead_code)]
    _index: Option<String>,  // Handle R-style row index column
    treatment: String,
    outcome: String,
}

/// Validated CSV row
#[derive(Debug)]
struct CsvRow {
    treatment: u8,
    outcome: u8,
}

/// Result of analyzing real trial data
struct AnalysisResult {
    n_total: usize,
    n_trt: usize,
    n_ctrl: usize,

    // Did we cross threshold?
    crossed: bool,
    crossed_at: Option<usize>,

    // Effect estimates at crossing (if crossed)
    risk_diff_at_cross: Option<f64>,
    or_at_cross: Option<(f64, f64, f64)>, // (OR, lower, upper)

    // Final effect estimates
    final_risk_diff: f64,
    final_or: (f64, f64, f64),

    // Event rates
    rate_trt: f64,
    rate_ctrl: f64,

    // Type M error (if crossed)
    type_m: Option<f64>,

    // Final e-value
    final_evalue: f64,

    // Trajectory for plotting
    trajectory: Vec<f64>,
}

pub fn run() -> Result<(), Box<dyn Error>> {
    println!("\n==========================================");
    println!("   ANALYZE BINARY TRIAL DATA");
    println!("==========================================\n");

    // Get CSV file path
    let csv_path = get_string("Path to CSV file: ");

    // Read and parse CSV
    println!("\nReading {}...", csv_path);
    let data = read_csv(&csv_path)?;
    let n_total = data.len();

    if n_total == 0 {
        println!("Error: CSV file is empty or has no valid rows.");
        return Ok(());
    }

    // Validate data
    let n_trt: usize = data.iter().filter(|r| r.treatment == 1).count();
    let n_ctrl = n_total - n_trt;
    let n_events_trt: usize = data.iter().filter(|r| r.treatment == 1 && r.outcome == 1).count();
    let n_events_ctrl: usize = data.iter().filter(|r| r.treatment == 0 && r.outcome == 1).count();

    println!("\n--- Data Summary ---");
    println!("Total patients:     {}", n_total);
    println!("Treatment arm:      {} ({} events, {:.1}%)",
             n_trt, n_events_trt,
             if n_trt > 0 { n_events_trt as f64 / n_trt as f64 * 100.0 } else { 0.0 });
    println!("Control arm:        {} ({} events, {:.1}%)",
             n_ctrl, n_events_ctrl,
             if n_ctrl > 0 { n_events_ctrl as f64 / n_ctrl as f64 * 100.0 } else { 0.0 });

    // Get analysis parameters
    println!("\n--- Analysis Parameters ---");

    println!("Burn-in period (default = 50):");
    let burn_in = get_input_usize("Burn-in: ");

    println!("Ramp period (default = 100):");
    let ramp = get_input_usize("Ramp: ");

    println!("Success threshold (1/alpha, default = 20 for alpha=0.05):");
    let success_threshold = get_input("Success threshold: ");

    let use_futility = get_bool("Set futility watch threshold?");
    let futility_threshold = if use_futility {
        Some(get_input("Futility threshold (e.g., 0.5): "))
    } else {
        None
    };

    // Run analysis
    println!("\n--- Running e-RT Analysis ---");
    let result = analyze_data(&data, burn_in, ramp, success_threshold, futility_threshold);

    // Print console results
    print_results(&result, success_threshold, futility_threshold);

    // Optional HTML report
    if get_bool("\nGenerate HTML report?") {
        let html = build_html_report(&result, &csv_path, burn_in, ramp,
                                      success_threshold, futility_threshold);
        let report_path = "binary_analysis_report.html";
        let mut file = File::create(report_path)?;
        file.write_all(html.as_bytes())?;
        println!("\n>> Report saved: {}", report_path);
    }

    println!("\n==========================================");
    Ok(())
}

fn read_csv(path: &str) -> Result<Vec<CsvRow>, Box<dyn Error>> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)  // Allow variable number of columns
        .from_path(path)?;

    let mut data = Vec::new();
    let mut skipped = 0;

    for result in reader.deserialize() {
        let row: CsvRowRaw = result?;

        // Try to parse treatment and outcome, skip NA rows
        let treatment = match row.treatment.parse::<u8>() {
            Ok(v) if v <= 1 => v,
            _ => { skipped += 1; continue; }
        };
        let outcome = match row.outcome.parse::<u8>() {
            Ok(v) if v <= 1 => v,
            _ => { skipped += 1; continue; }
        };

        data.push(CsvRow { treatment, outcome });
    }

    if skipped > 0 {
        println!("  (Skipped {} rows with NA or invalid values)", skipped);
    }

    Ok(data)
}

fn analyze_data(
    data: &[CsvRow],
    burn_in: usize,
    ramp: usize,
    success_threshold: f64,
    _futility_threshold: Option<f64>,
) -> AnalysisResult {
    let n_total = data.len();
    let mut proc = BinaryERTProcess::new(burn_in, ramp);

    let mut trajectory = Vec::with_capacity(n_total + 1);
    trajectory.push(1.0); // Initial wealth

    let mut crossed = false;
    let mut crossed_at: Option<usize> = None;
    let mut risk_diff_at_cross: Option<f64> = None;
    let mut or_at_cross: Option<(f64, f64, f64)> = None;

    // Process each patient in enrollment order
    for (i, row) in data.iter().enumerate() {
        let patient_num = i + 1;
        let is_trt = row.treatment == 1;
        let outcome = row.outcome as f64;

        proc.update(patient_num, outcome, is_trt);
        trajectory.push(proc.wealth);

        // Check for crossing
        if !crossed && proc.wealth >= success_threshold {
            crossed = true;
            crossed_at = Some(patient_num);
            risk_diff_at_cross = Some(proc.current_risk_diff());
            or_at_cross = Some(proc.current_odds_ratio());
        }
    }

    // Final statistics
    let final_risk_diff = proc.current_risk_diff();
    let final_or = proc.current_odds_ratio();
    let (rate_trt, rate_ctrl) = proc.get_rates();
    let (n_trt, n_ctrl) = proc.get_ns();

    // Type M error (if crossed)
    let type_m = if crossed {
        let rd_cross = risk_diff_at_cross.unwrap().abs();
        let rd_final = final_risk_diff.abs();
        if rd_final > 0.0 {
            Some(rd_cross / rd_final)
        } else {
            None
        }
    } else {
        None
    };

    AnalysisResult {
        n_total,
        n_trt,
        n_ctrl,
        crossed,
        crossed_at,
        risk_diff_at_cross,
        or_at_cross,
        final_risk_diff,
        final_or,
        rate_trt,
        rate_ctrl,
        type_m,
        final_evalue: proc.wealth,
        trajectory,
    }
}

fn print_results(result: &AnalysisResult, success_threshold: f64, futility_threshold: Option<f64>) {
    println!("\n==========================================");
    println!("   RESULTS");
    println!("==========================================");

    // e-value result
    println!("\n--- e-Value ---");
    println!("Final e-value:      {:.4}", result.final_evalue);
    println!("Threshold:          {:.1}", success_threshold);

    if result.crossed {
        println!("Status:             CROSSED at patient {}", result.crossed_at.unwrap());
    } else if let Some(fut) = futility_threshold {
        if result.final_evalue < fut {
            println!("Status:             Below futility threshold ({:.2})", fut);
        } else {
            println!("Status:             Did not cross (ongoing)");
        }
    } else {
        println!("Status:             Did not cross (ongoing)");
    }

    // Effect sizes
    println!("\n--- Effect Sizes ---");

    if result.crossed {
        println!("\nAt Crossing (patient {}):", result.crossed_at.unwrap());
        let (or, or_lo, or_hi) = result.or_at_cross.unwrap();
        println!("  Risk Difference:  {:.1}%", result.risk_diff_at_cross.unwrap() * 100.0);
        println!("  Odds Ratio:       {:.3} (95% CI: {:.3} - {:.3})", or, or_lo, or_hi);
    }

    println!("\nFinal (patient {}):", result.n_total);
    let (or, or_lo, or_hi) = result.final_or;
    println!("  Risk Difference:  {:.1}%", result.final_risk_diff * 100.0);
    println!("  Odds Ratio:       {:.3} (95% CI: {:.3} - {:.3})", or, or_lo, or_hi);

    // Type M error
    if let Some(type_m) = result.type_m {
        println!("\n--- Type M Error (Magnification) ---");
        println!("  |RD at cross| / |RD final|: {:.2}x", type_m);
    }

    // Event rates by arm
    println!("\n--- Event Rates ---");
    println!("  Treatment:        {:.1}% ({} patients)", result.rate_trt * 100.0, result.n_trt);
    println!("  Control:          {:.1}% ({} patients)", result.rate_ctrl * 100.0, result.n_ctrl);
}

fn build_html_report(
    result: &AnalysisResult,
    csv_path: &str,
    burn_in: usize,
    ramp: usize,
    success_threshold: f64,
    futility_threshold: Option<f64>,
) -> String {
    let timestamp = chrono_lite();

    // Prepare trajectory data for plotting
    let x_axis: Vec<usize> = (0..=result.n_total).collect();
    let x_json = format!("{:?}", x_axis);
    let y_json = format!("{:?}", result.trajectory);

    let status_text = if result.crossed {
        format!("<span style='color:green;font-weight:bold'>CROSSED at patient {}</span>",
                result.crossed_at.unwrap())
    } else if let Some(fut) = futility_threshold {
        if result.final_evalue < fut {
            format!("<span style='color:orange'>Below futility threshold ({:.2})</span>", fut)
        } else {
            "<span style='color:gray'>Did not cross (ongoing)</span>".to_string()
        }
    } else {
        "<span style='color:gray'>Did not cross (ongoing)</span>".to_string()
    };

    // Effect at crossing section
    let crossing_section = if result.crossed {
        let (or, or_lo, or_hi) = result.or_at_cross.unwrap();
        format!(r#"
        <h3>At Crossing (Patient {})</h3>
        <table>
            <tr><td>Risk Difference:</td><td><strong>{:.1}%</strong></td></tr>
            <tr><td>Odds Ratio:</td><td><strong>{:.3}</strong> (95% CI: {:.3} - {:.3})</td></tr>
        </table>
        "#,
        result.crossed_at.unwrap(),
        result.risk_diff_at_cross.unwrap() * 100.0,
        or, or_lo, or_hi)
    } else {
        String::new()
    };

    // Type M section
    let type_m_section = if let Some(type_m) = result.type_m {
        format!(r#"
        <h3>Type M Error (Magnification)</h3>
        <table>
            <tr><td>|RD at cross| / |RD final|:</td><td><strong>{:.2}x</strong></td></tr>
        </table>
        "#, type_m)
    } else {
        String::new()
    };

    let futility_line = if let Some(fut) = futility_threshold {
        format!("{{type:'line',x0:0,x1:1,xref:'paper',y0:{},y1:{},line:{{color:'orange',width:1.5,dash:'dot'}}}}", fut, fut)
    } else {
        String::new()
    };

    let (final_or, final_or_lo, final_or_hi) = result.final_or;

    format!(r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Binary e-RT Analysis Report</title>
    <script src="https://cdn.plot.ly/plotly-2.12.1.min.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        .container {{ background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        h3 {{ color: #7f8c8d; }}
        table {{ border-collapse: collapse; margin: 15px 0; }}
        td {{ padding: 8px 16px; border-bottom: 1px solid #eee; }}
        td:first-child {{ color: #7f8c8d; }}
        .highlight {{ background: #e8f4f8; font-weight: bold; }}
        .timestamp {{ color: #95a5a6; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Binary e-RT Analysis Report</h1>
        <p class="timestamp">Generated: {}</p>

        <h2>Data Source</h2>
        <table>
            <tr><td>File:</td><td>{}</td></tr>
            <tr><td>Total Patients:</td><td>{}</td></tr>
            <tr><td>Treatment Arm:</td><td>{} ({:.1}% event rate)</td></tr>
            <tr><td>Control Arm:</td><td>{} ({:.1}% event rate)</td></tr>
        </table>

        <h2>Parameters</h2>
        <table>
            <tr><td>Burn-in:</td><td>{}</td></tr>
            <tr><td>Ramp:</td><td>{}</td></tr>
            <tr><td>Success Threshold:</td><td>{}</td></tr>
            <tr><td>Futility Threshold:</td><td>{}</td></tr>
        </table>

        <h2>Results</h2>
        <table>
            <tr class="highlight"><td>Final e-value:</td><td>{:.4}</td></tr>
            <tr class="highlight"><td>Status:</td><td>{}</td></tr>
        </table>

        {}

        <h3>Final Effect Estimates (Patient {})</h3>
        <table>
            <tr><td>Risk Difference:</td><td><strong>{:.1}%</strong></td></tr>
            <tr><td>Odds Ratio:</td><td><strong>{:.3}</strong> (95% CI: {:.3} - {:.3})</td></tr>
        </table>

        {}

        <h2>e-Value Trajectory</h2>
        <div id="plot1" style="width:100%;height:500px;"></div>
    </div>

    <script>
        Plotly.newPlot('plot1', [
            {{
                type: 'scatter',
                mode: 'lines',
                x: {},
                y: {},
                line: {{color: 'blue', width: 2}},
                name: 'e-value'
            }}
        ], {{
            yaxis: {{type: 'log', title: 'e-value'}},
            xaxis: {{title: 'Patients Enrolled'}},
            shapes: [
                {{type:'line',x0:0,x1:1,xref:'paper',y0:{},y1:{},line:{{color:'green',width:2,dash:'dash'}}}},
                {}
            ],
            annotations: [
                {{x:1,xref:'paper',y:{},text:'Success',showarrow:false,font:{{color:'green'}}}}
            ]
        }});
    </script>
</body>
</html>"#,
        // Header
        timestamp,
        // Data source
        csv_path, result.n_total,
        result.n_trt, result.rate_trt * 100.0,
        result.n_ctrl, result.rate_ctrl * 100.0,
        // Parameters
        burn_in, ramp, success_threshold,
        futility_threshold.map_or("None".to_string(), |f| format!("{:.2}", f)),
        // Results
        result.final_evalue,
        status_text,
        // Crossing section
        crossing_section,
        // Final estimates
        result.n_total,
        result.final_risk_diff * 100.0,
        final_or, final_or_lo, final_or_hi,
        // Type M section
        type_m_section,
        // Plot data
        x_json, y_json,
        success_threshold, success_threshold,
        futility_line,
        success_threshold
    )
}
