//! Analyze real binary trial data from CSV

use std::error::Error;
use std::fs::File;
use std::io::Write;
use csv::ReaderBuilder;
use serde::Deserialize;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

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

/// Design assumptions for futility analysis
#[derive(Clone)]
struct DesignParams {
    control_rate: f64,
    treatment_rate: f64,
    design_arr: f64,
}

/// A single futility checkpoint measurement
#[derive(Clone)]
struct FutilityPoint {
    patient_num: usize,
    wealth: f64,
    required_arr: f64,
    ratio_to_design: f64,
}

/// Futility analysis summary
struct FutilityAnalysis {
    design: DesignParams,
    points: Vec<FutilityPoint>,          // All checkpoint measurements
    regions: Vec<(usize, usize)>,        // (start, end) of futility regions
    worst_point: Option<FutilityPoint>,  // Highest ratio point
    ever_triggered: bool,
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

    // Futility analysis (if enabled)
    futility_analysis: Option<FutilityAnalysis>,
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

    // Futility configuration
    let use_futility = get_bool("Enable futility monitoring?");
    let (futility_threshold, design_params) = if use_futility {
        let fut = get_input("Futility threshold (e.g., 0.5): ");

        println!("\n--- Design Assumptions (for futility analysis) ---");
        let p_ctrl = get_input("Design control event rate (e.g., 0.30): ");
        let p_trt = get_input("Design treatment event rate (e.g., 0.20): ");
        let design_arr = (p_ctrl - p_trt).abs();
        println!("Design ARR: {:.1}%", design_arr * 100.0);

        (Some(fut), Some(DesignParams {
            control_rate: p_ctrl,
            treatment_rate: p_trt,
            design_arr,
        }))
    } else {
        (None, None)
    };

    // Run analysis
    println!("\n--- Running e-RT Analysis ---");
    let result = analyze_data(&data, burn_in, ramp, success_threshold,
                               futility_threshold, design_params);

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
        .flexible(true)
        .from_path(path)?;

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

        data.push(CsvRow { treatment, outcome });
    }

    if skipped > 0 {
        println!("  (Skipped {} rows with NA or invalid values)", skipped);
    }

    Ok(data)
}

/// Monte Carlo: find required ARR for 50% chance of recovery
fn required_arr_for_recovery(
    current_wealth: f64,
    n_remaining: usize,
    p_ctrl: f64,
    burn_in: usize,
    ramp: usize,
    success_threshold: f64,
    mc_sims: usize,
) -> f64 {
    if n_remaining == 0 {
        return 1.0;
    }

    let mut rng = StdRng::seed_from_u64(42); // Fixed seed for reproducibility
    let mut low = 0.001;
    let mut high = 0.50;

    for _ in 0..8 {  // Binary search iterations
        let mid = (low + high) / 2.0;
        let p_trt = (p_ctrl - mid).max(0.001);

        let mut successes = 0;
        for _ in 0..mc_sims {
            let mut wealth = current_wealth;
            let mut n_trt = 0.0;
            let mut events_trt = 0.0;
            let mut n_ctrl = 0.0;
            let mut events_ctrl = 0.0;

            for j in 1..=n_remaining {
                let is_trt = rng.gen_bool(0.5);
                let prob = if is_trt { p_trt } else { p_ctrl };
                let outcome = if rng.gen_bool(prob) { 1.0 } else { 0.0 };

                let rate_trt = if n_trt > 0.0 { events_trt / n_trt } else { 0.5 };
                let rate_ctrl = if n_ctrl > 0.0 { events_ctrl / n_ctrl } else { 0.5 };
                let delta_hat = rate_trt - rate_ctrl;

                if is_trt {
                    n_trt += 1.0;
                    if outcome == 1.0 { events_trt += 1.0; }
                } else {
                    n_ctrl += 1.0;
                    if outcome == 1.0 { events_ctrl += 1.0; }
                }

                if j > burn_in {
                    let num = ((j - burn_in) as f64).max(0.0);
                    let c_i = (num / ramp as f64).clamp(0.0, 1.0);

                    let lambda = if outcome == 1.0 {
                        0.5 + 0.5 * c_i * delta_hat
                    } else {
                        0.5 - 0.5 * c_i * delta_hat
                    };
                    let lambda = lambda.clamp(0.001, 0.999);
                    let multiplier = if is_trt { lambda / 0.5 } else { (1.0 - lambda) / 0.5 };
                    wealth *= multiplier;
                }

                if wealth >= success_threshold {
                    successes += 1;
                    break;
                }
            }
        }

        let success_rate = successes as f64 / mc_sims as f64;
        if success_rate < 0.5 {
            low = mid;
        } else {
            high = mid;
        }
    }

    (low + high) / 2.0
}

fn analyze_data(
    data: &[CsvRow],
    burn_in: usize,
    ramp: usize,
    success_threshold: f64,
    futility_threshold: Option<f64>,
    design_params: Option<DesignParams>,
) -> AnalysisResult {
    let n_total = data.len();
    let mut proc = BinaryERTProcess::new(burn_in, ramp);

    let mut trajectory = Vec::with_capacity(n_total + 1);
    trajectory.push(1.0);

    let mut crossed = false;
    let mut crossed_at: Option<usize> = None;
    let mut risk_diff_at_cross: Option<f64> = None;
    let mut or_at_cross: Option<(f64, f64, f64)> = None;

    // Futility tracking
    let checkpoint_interval = (n_total as f64 * 0.02).ceil() as usize; // 2% intervals
    let mut futility_points: Vec<FutilityPoint> = Vec::new();
    let mut futility_regions: Vec<(usize, usize)> = Vec::new();
    let mut in_futility = false;
    let mut futility_start: usize = 0;

    // Process each patient
    for (i, row) in data.iter().enumerate() {
        let patient_num = i + 1;
        let is_trt = row.treatment == 1;
        let outcome = row.outcome as f64;

        proc.update(patient_num, outcome, is_trt);
        trajectory.push(proc.wealth);

        // Check for success crossing
        if !crossed && proc.wealth >= success_threshold {
            crossed = true;
            crossed_at = Some(patient_num);
            risk_diff_at_cross = Some(proc.current_risk_diff());
            or_at_cross = Some(proc.current_odds_ratio());
        }

        // Futility tracking (if enabled)
        if let (Some(fut_thresh), Some(ref design)) = (futility_threshold, &design_params) {
            let below_futility = proc.wealth < fut_thresh;

            // Track futility regions
            if below_futility && !in_futility {
                in_futility = true;
                futility_start = patient_num;
            } else if !below_futility && in_futility {
                in_futility = false;
                futility_regions.push((futility_start, patient_num - 1));
            }

            // Calculate required effect at checkpoints when below futility
            if below_futility && patient_num % checkpoint_interval == 0 && patient_num > burn_in {
                let n_remaining = n_total - patient_num;
                let required_arr = required_arr_for_recovery(
                    proc.wealth,
                    n_remaining,
                    design.control_rate,
                    burn_in,
                    ramp,
                    success_threshold,
                    100,  // MC simulations
                );

                futility_points.push(FutilityPoint {
                    patient_num,
                    wealth: proc.wealth,
                    required_arr,
                    ratio_to_design: required_arr / design.design_arr,
                });
            }
        }
    }

    // Close final futility region if still in one
    if in_futility {
        futility_regions.push((futility_start, n_total));
    }

    // Final statistics
    let final_risk_diff = proc.current_risk_diff();
    let final_or = proc.current_odds_ratio();
    let (rate_trt, rate_ctrl) = proc.get_rates();
    let (n_trt, n_ctrl) = proc.get_ns();

    let type_m = if crossed {
        let rd_cross = risk_diff_at_cross.unwrap().abs();
        let rd_final = final_risk_diff.abs();
        if rd_final > 0.0 { Some(rd_cross / rd_final) } else { None }
    } else {
        None
    };

    // Build futility analysis
    let futility_analysis = if let Some(design) = design_params {
        let worst_point = futility_points.iter()
            .max_by(|a, b| a.ratio_to_design.partial_cmp(&b.ratio_to_design).unwrap())
            .cloned();

        let ever_triggered = !futility_regions.is_empty();

        Some(FutilityAnalysis {
            design,
            points: futility_points,
            regions: futility_regions,
            worst_point,
            ever_triggered,
        })
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
        futility_analysis,
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

    // Event rates
    println!("\n--- Event Rates ---");
    println!("  Treatment:        {:.1}% ({} patients)", result.rate_trt * 100.0, result.n_trt);
    println!("  Control:          {:.1}% ({} patients)", result.rate_ctrl * 100.0, result.n_ctrl);

    // Futility analysis
    if let Some(ref fut_analysis) = result.futility_analysis {
        println!("\n--- Futility Analysis ---");
        println!("  Design ARR:       {:.1}%", fut_analysis.design.design_arr * 100.0);

        if fut_analysis.ever_triggered {
            println!("  Episodes:         {} time(s) below threshold", fut_analysis.regions.len());
            for (i, (start, end)) in fut_analysis.regions.iter().enumerate() {
                println!("    Episode {}: patients {} - {}", i + 1, start, end);
            }

            if let Some(ref worst) = fut_analysis.worst_point {
                println!("  Worst point:");
                println!("    Patient:        {}", worst.patient_num);
                println!("    Wealth:         {:.4}", worst.wealth);
                println!("    Required ARR:   {:.1}%", worst.required_arr * 100.0);
                println!("    Ratio to design: {:.2}x", worst.ratio_to_design);
            }
        } else {
            println!("  Status:           Never dropped below futility threshold");
        }
    }
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

    // Futility lines for plots
    let futility_line = if let Some(fut) = futility_threshold {
        format!("{{type:'line',x0:0,x1:1,xref:'paper',y0:{},y1:{},line:{{color:'orange',width:1.5,dash:'dot'}}}}", fut, fut)
    } else {
        String::new()
    };

    let futility_line_support = if let Some(fut) = futility_threshold {
        format!("{{type:'line',x0:0,x1:1,xref:'paper',y0:{:.4},y1:{:.4},line:{{color:'orange',width:1.5,dash:'dot'}}}}", fut.ln(), fut.ln())
    } else {
        String::new()
    };

    // Futility shading regions for plots
    let mut futility_shapes = String::new();
    let mut futility_shapes_support = String::new();

    if let Some(ref fut_analysis) = result.futility_analysis {
        for (start, end) in &fut_analysis.regions {
            // For e-value plot (log scale)
            futility_shapes.push_str(&format!(
                "{{type:'rect',x0:{},x1:{},y0:0,y1:1,yref:'paper',fillcolor:'rgba(255,165,0,0.15)',line:{{width:0}}}},",
                start, end
            ));
            // For support plot
            futility_shapes_support.push_str(&format!(
                "{{type:'rect',x0:{},x1:{},y0:0,y1:1,yref:'paper',fillcolor:'rgba(255,165,0,0.15)',line:{{width:0}}}},",
                start, end
            ));
        }
    }

    // Futility analysis section for HTML
    let futility_section = if let Some(ref fut_analysis) = result.futility_analysis {
        let mut html = format!(r#"
        <h2>Futility Analysis</h2>
        <p><em>Note: Futility monitoring is decision support, not anytime-valid inference.</em></p>
        <table>
            <tr><td>Design Control Rate:</td><td>{:.1}%</td></tr>
            <tr><td>Design Treatment Rate:</td><td>{:.1}%</td></tr>
            <tr><td>Design ARR:</td><td><strong>{:.1}%</strong></td></tr>
            <tr><td>Ever Below Threshold:</td><td>{}</td></tr>
        </table>
        "#,
        fut_analysis.design.control_rate * 100.0,
        fut_analysis.design.treatment_rate * 100.0,
        fut_analysis.design.design_arr * 100.0,
        if fut_analysis.ever_triggered { "Yes" } else { "No" }
        );

        if fut_analysis.ever_triggered {
            html.push_str("<h3>Futility Episodes</h3><table>");
            for (i, (start, end)) in fut_analysis.regions.iter().enumerate() {
                html.push_str(&format!(
                    "<tr><td>Episode {}:</td><td>Patients {} - {}</td></tr>",
                    i + 1, start, end
                ));
            }
            html.push_str("</table>");

            if let Some(ref worst) = fut_analysis.worst_point {
                html.push_str(&format!(r#"
                <h3>Worst Point</h3>
                <table>
                    <tr><td>Patient:</td><td>{}</td></tr>
                    <tr><td>Wealth:</td><td>{:.4}</td></tr>
                    <tr><td>Required ARR for recovery:</td><td><strong>{:.1}%</strong></td></tr>
                    <tr><td>Ratio to design ARR:</td><td><strong>{:.2}x</strong></td></tr>
                </table>
                "#,
                worst.patient_num, worst.wealth, worst.required_arr * 100.0, worst.ratio_to_design
                ));
            }
        }

        // Add ratio plot if we have data points
        if !fut_analysis.points.is_empty() {
            let ratio_x: Vec<usize> = fut_analysis.points.iter().map(|p| p.patient_num).collect();
            let ratio_y: Vec<f64> = fut_analysis.points.iter().map(|p| p.ratio_to_design).collect();
            let ratio_x_json = format!("{:?}", ratio_x);
            let ratio_y_json = format!("{:?}", ratio_y);

            html.push_str(&format!(r#"
            <h3>Required Effect Ratio Over Time</h3>
            <p>Shows ratio of required ARR to design ARR when below futility threshold. Higher = harder to recover.</p>
            <div id="plot3" style="width:100%;height:350px;"></div>
            <script>
                Plotly.newPlot('plot3', [
                    {{
                        type: 'scatter',
                        mode: 'lines+markers',
                        x: {},
                        y: {},
                        line: {{color: 'darkorange', width: 2}},
                        marker: {{size: 6}},
                        name: 'Required/Design Ratio'
                    }}
                ], {{
                    yaxis: {{title: 'Required ARR / Design ARR', rangemode: 'tozero'}},
                    xaxis: {{title: 'Patient Number'}},
                    shapes: [
                        {{type:'line',x0:0,x1:1,xref:'paper',y0:1,y1:1,line:{{color:'gray',width:1.5,dash:'dash'}}}}
                    ],
                    annotations: [
                        {{x:1,xref:'paper',y:1,text:'Design (1.0x)',showarrow:false,font:{{color:'gray'}}}}
                    ]
                }});
            </script>
            "#, ratio_x_json, ratio_y_json));
        }

        html
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
        em {{ color: #95a5a6; }}
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

        {}

        <h2>e-Value Trajectory</h2>
        <div id="plot1" style="width:100%;height:400px;"></div>

        <h2>Support (ln e-value)</h2>
        <div id="plot2" style="width:100%;height:400px;"></div>
    </div>

    <script>
        // e-value trajectory (log scale)
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
                {}
                {{type:'line',x0:0,x1:1,xref:'paper',y0:{},y1:{},line:{{color:'green',width:2,dash:'dash'}}}},
                {}
            ],
            annotations: [
                {{x:1,xref:'paper',y:{},text:'Success (e={:.0})',showarrow:false,font:{{color:'green'}}}}
            ]
        }});

        // Support trajectory (linear scale, ln of e-value)
        var support = {}.map(function(e) {{ return Math.log(e); }});
        Plotly.newPlot('plot2', [
            {{
                type: 'scatter',
                mode: 'lines',
                x: {},
                y: support,
                line: {{color: 'blue', width: 2}},
                name: 'Support'
            }}
        ], {{
            yaxis: {{title: 'Support (ln e-value)'}},
            xaxis: {{title: 'Patients Enrolled'}},
            shapes: [
                {}
                {{type:'line',x0:0,x1:1,xref:'paper',y0:{:.4},y1:{:.4},line:{{color:'green',width:2,dash:'dash'}}}},
                {}
            ],
            annotations: [
                {{x:1,xref:'paper',y:{:.4},text:'Success (ln {:.0} = {:.2})',showarrow:false,font:{{color:'green'}}}}
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
        // Futility section
        futility_section,
        // Plot 1: e-value (log scale)
        x_json, y_json,
        futility_shapes,
        success_threshold, success_threshold,
        futility_line,
        success_threshold, success_threshold,
        // Plot 2: Support (ln scale)
        y_json,
        x_json,
        futility_shapes_support,
        success_threshold.ln(), success_threshold.ln(),
        futility_line_support,
        success_threshold.ln(), success_threshold, success_threshold.ln()
    )
}
