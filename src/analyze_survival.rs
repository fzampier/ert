//! Analyze real survival/time-to-event trial data from CSV

use std::error::Error;
use std::fs::File;
use std::io::Write;
use csv::ReaderBuilder;
use serde::Deserialize;

use crate::ert_core::{get_input, get_input_usize, get_bool, get_string, chrono_lite};

// === DATA STRUCTURES ===

#[derive(Debug, Deserialize)]
struct CsvRowRaw {
    treatment: String,
    time: String,
    status: String,
}

struct SurvivalRecord {
    treatment: u8,  // 0=control, 1=treatment
    time: f64,
    status: u8,     // 0=censored, 1=event
}

struct AnalysisResult {
    n_total: usize,
    n_trt: usize,
    n_ctrl: usize,
    n_events: usize,
    events_trt: usize,
    events_ctrl: usize,
    crossed: bool,
    crossed_at: Option<usize>,  // event number
    hr_at_cross: Option<f64>,
    hr_final: f64,
    hr_ci: (f64, f64),
    type_m: Option<f64>,
    final_evalue: f64,
    min_evalue: f64,
    trajectory: Vec<f64>,  // wealth at each event
}

// === CLI ===

pub fn run_cli(csv_path: &str, opts: &crate::AnalyzeOptions) -> Result<(), Box<dyn Error>> {
    println!("\n==========================================");
    println!("   ANALYZE SURVIVAL TRIAL DATA (e-RTs)");
    println!("==========================================\n");

    println!("Reading {}...", csv_path);
    let data = read_csv(csv_path)?;
    if data.is_empty() {
        println!("Error: No valid rows in CSV.");
        return Ok(());
    }

    print_data_summary(&data);

    let burn_in = opts.burn_in.unwrap_or(30);
    let ramp = opts.ramp.unwrap_or(50);
    let threshold = opts.threshold.unwrap_or(20.0);

    println!("\n--- Parameters ---");
    println!("Burn-in: {}  Ramp: {}  Threshold: {}", burn_in, ramp, threshold);

    println!("\n--- Running Analysis ---");
    let result = analyze(&data, burn_in, ramp, threshold);

    print_results(&result, threshold);

    if opts.generate_report {
        let html = build_report(&result, csv_path, burn_in, ramp, threshold);
        File::create("survival_analysis_report.html")?.write_all(html.as_bytes())?;
        println!("\n>> Saved: survival_analysis_report.html");
    }

    Ok(())
}

// === INTERACTIVE ===

pub fn run() -> Result<(), Box<dyn Error>> {
    println!("\n==========================================");
    println!("   ANALYZE SURVIVAL TRIAL DATA (e-RTs)");
    println!("==========================================\n");

    let csv_path = get_string("Path to CSV file: ");
    println!("\nReading {}...", csv_path);

    let data = read_csv(&csv_path)?;
    if data.is_empty() {
        println!("Error: No valid rows in CSV.");
        return Ok(());
    }

    print_data_summary(&data);

    println!("\n--- Parameters ---");
    let burn_in = get_input_usize("Burn-in events (default 30): ");
    let ramp = get_input_usize("Ramp events (default 50): ");
    let threshold: f64 = get_input("Success threshold (default 20): ");

    println!("\n--- Running Analysis ---");
    let result = analyze(&data, burn_in, ramp, threshold);

    print_results(&result, threshold);

    if get_bool("\nGenerate HTML report?") {
        let html = build_report(&result, &csv_path, burn_in, ramp, threshold);
        File::create("survival_analysis_report.html")?.write_all(html.as_bytes())?;
        println!("\n>> survival_analysis_report.html");
    }

    Ok(())
}

// === CSV PARSING ===

fn read_csv(path: &str) -> Result<Vec<SurvivalRecord>, Box<dyn Error>> {
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
        let time = match row.time.parse::<f64>() {
            Ok(v) if v >= 0.0 => v,
            _ => { skipped += 1; continue; }
        };
        let status = match row.status.parse::<u8>() {
            Ok(v) if v <= 1 => v,
            _ => { skipped += 1; continue; }
        };

        data.push(SurvivalRecord { treatment, time, status });
    }

    if skipped > 0 {
        println!("  (Skipped {} invalid rows)", skipped);
    }
    Ok(data)
}

fn print_data_summary(data: &[SurvivalRecord]) {
    let n_total = data.len();
    let n_trt = data.iter().filter(|r| r.treatment == 1).count();
    let n_ctrl = n_total - n_trt;
    let events_trt = data.iter().filter(|r| r.treatment == 1 && r.status == 1).count();
    let events_ctrl = data.iter().filter(|r| r.treatment == 0 && r.status == 1).count();
    let n_events = events_trt + events_ctrl;

    let median_time: f64 = {
        let mut times: Vec<f64> = data.iter().map(|r| r.time).collect();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        times[times.len() / 2]
    };

    println!("\n--- Data Summary ---");
    println!("Total:      {} patients", n_total);
    println!("Events:     {} ({:.1}%)", n_events, n_events as f64 / n_total as f64 * 100.0);
    println!("Treatment:  {} ({} events, {:.1}%)", n_trt, events_trt,
             if n_trt > 0 { events_trt as f64 / n_trt as f64 * 100.0 } else { 0.0 });
    println!("Control:    {} ({} events, {:.1}%)", n_ctrl, events_ctrl,
             if n_ctrl > 0 { events_ctrl as f64 / n_ctrl as f64 * 100.0 } else { 0.0 });
    println!("Median time: {:.2}", median_time);
}

// === ANALYSIS (e-RTs) ===

fn analyze(data: &[SurvivalRecord], burn_in: usize, ramp: usize, threshold: f64) -> AnalysisResult {
    let lambda_max: f64 = 0.25;

    // Sort by time
    let mut sorted: Vec<&SurvivalRecord> = data.iter().collect();
    sorted.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());

    // Initialize risk sets
    let mut risk_trt: i32 = data.iter().filter(|r| r.treatment == 1).count() as i32;
    let mut risk_ctrl: i32 = data.iter().filter(|r| r.treatment == 0).count() as i32;

    let mut wealth: f64 = 1.0;
    let mut cumulative_z: f64 = 0.0;
    let mut event_count: usize = 0;

    let mut trajectory: Vec<f64> = vec![1.0];
    let mut min_wealth: f64 = 1.0;

    let mut crossed = false;
    let mut crossed_at: Option<usize> = None;
    let mut hr_at_cross: Option<f64> = None;

    // Track for HR calculation
    let mut events_trt: f64 = 0.0;
    let mut events_ctrl: f64 = 0.0;
    let mut time_trt: f64 = 0.0;
    let mut time_ctrl: f64 = 0.0;

    for rec in &sorted {
        let is_event = rec.status == 1;
        let is_trt = rec.treatment == 1;

        // Accumulate person-time (simplified: use event/censor time)
        if is_trt {
            time_trt += rec.time;
        } else {
            time_ctrl += rec.time;
        }

        if is_event {
            event_count += 1;

            // Betting
            let b_i = if event_count > burn_in {
                let c_i = ((event_count - burn_in) as f64 / ramp as f64).clamp(0.0, 1.0);
                let bet_direction = if cumulative_z > 0.0 { 1.0 }
                    else if cumulative_z < 0.0 { -1.0 }
                    else { 0.0 };
                c_i * lambda_max * bet_direction
            } else {
                0.0
            };

            let total_risk = risk_trt + risk_ctrl;
            let p_null = if total_risk > 0 {
                risk_trt as f64 / total_risk as f64
            } else {
                0.5
            };

            let obs = if is_trt { 1.0 } else { 0.0 };
            let u_i = obs - p_null;
            let multiplier = 1.0 + b_i * u_i;

            cumulative_z += u_i;
            wealth *= multiplier;
            min_wealth = min_wealth.min(wealth);

            // Track events for HR
            if is_trt {
                events_trt += 1.0;
            } else {
                events_ctrl += 1.0;
            }

            trajectory.push(wealth);

            // Check crossing
            if !crossed && wealth >= threshold {
                crossed = true;
                crossed_at = Some(event_count);
                // Use event ratio as simple HR proxy at crossing
                // (crude rate HR can be misleading for early interim analyses)
                hr_at_cross = if events_ctrl > 0.0 {
                    Some(events_trt / events_ctrl)
                } else {
                    Some(1.0)
                };
            }
        }

        // Update risk sets
        if is_trt {
            risk_trt = (risk_trt - 1).max(0);
        } else {
            risk_ctrl = (risk_ctrl - 1).max(0);
        }
    }

    // Final HR
    let hr_final = calculate_hr(events_trt, events_ctrl, time_trt, time_ctrl);
    let hr_ci = calculate_hr_ci(events_trt as usize, events_ctrl as usize, hr_final);

    // Type M
    let type_m = if crossed {
        let hr_c = hr_at_cross.unwrap();
        let log_hr_c = hr_c.ln().abs();
        let log_hr_f = hr_final.ln().abs();
        if log_hr_f > 0.001 { Some(log_hr_c / log_hr_f) } else { None }
    } else {
        None
    };

    let n_trt = data.iter().filter(|r| r.treatment == 1).count();
    let n_ctrl = data.len() - n_trt;

    AnalysisResult {
        n_total: data.len(),
        n_trt,
        n_ctrl,
        n_events: event_count,
        events_trt: events_trt as usize,
        events_ctrl: events_ctrl as usize,
        crossed,
        crossed_at,
        hr_at_cross,
        hr_final,
        hr_ci,
        type_m,
        final_evalue: wealth,
        min_evalue: min_wealth,
        trajectory,
    }
}

fn calculate_hr(events_trt: f64, events_ctrl: f64, time_trt: f64, time_ctrl: f64) -> f64 {
    let rate_trt = if time_trt > 0.0 { events_trt / time_trt } else { 0.0 };
    let rate_ctrl = if time_ctrl > 0.0 { events_ctrl / time_ctrl } else { 0.0 };
    if rate_ctrl > 0.0 { rate_trt / rate_ctrl } else { 1.0 }
}

fn calculate_hr_ci(events_trt: usize, events_ctrl: usize, hr: f64) -> (f64, f64) {
    // Approximate CI using 1/sqrt(events)
    let total_events = events_trt + events_ctrl;
    if total_events < 2 {
        return (0.0, f64::INFINITY);
    }
    let se_log_hr = ((1.0 / events_trt as f64) + (1.0 / events_ctrl as f64)).sqrt();
    let log_hr = hr.ln();
    let lower = (log_hr - 1.96 * se_log_hr).exp();
    let upper = (log_hr + 1.96 * se_log_hr).exp();
    (lower, upper)
}

// === CONSOLE OUTPUT ===

fn print_results(r: &AnalysisResult, threshold: f64) {
    println!("\n==========================================");
    println!("   RESULTS");
    println!("==========================================");

    println!("\n--- e-Value ---");
    println!("Final:     {:.4}", r.final_evalue);
    println!("Minimum:   {:.4}", r.min_evalue);
    println!("Threshold: {:.1}", threshold);

    if r.crossed {
        println!("Status:    CROSSED at event {}", r.crossed_at.unwrap());
    } else {
        println!("Status:    Did not cross");
    }

    println!("\n--- Hazard Ratio ---");
    if r.crossed {
        println!("At crossing ({} events):", r.crossed_at.unwrap());
        println!("  HR: {:.3}", r.hr_at_cross.unwrap());
    }
    println!("Final ({} events):", r.n_events);
    println!("  HR: {:.3} ({:.3}-{:.3})", r.hr_final, r.hr_ci.0, r.hr_ci.1);

    if let Some(tm) = r.type_m {
        println!("\nType M: {:.2}x", tm);
    }

    println!("\n--- Events ---");
    println!("Treatment: {}/{} ({:.1}%)", r.events_trt, r.n_trt,
             if r.n_trt > 0 { r.events_trt as f64 / r.n_trt as f64 * 100.0 } else { 0.0 });
    println!("Control:   {}/{} ({:.1}%)", r.events_ctrl, r.n_ctrl,
             if r.n_ctrl > 0 { r.events_ctrl as f64 / r.n_ctrl as f64 * 100.0 } else { 0.0 });
}

// === HTML REPORT ===

fn build_report(r: &AnalysisResult, csv_path: &str, burn_in: usize, ramp: usize, threshold: f64) -> String {
    let status = if r.crossed {
        format!("<span style='color:green'>CROSSED at event {}</span>", r.crossed_at.unwrap())
    } else {
        "<span style='color:gray'>Did not cross</span>".into()
    };

    let crossing_row = if r.crossed {
        format!("<tr><td>At crossing:</td><td>HR {:.3}</td></tr>", r.hr_at_cross.unwrap())
    } else {
        String::new()
    };

    let type_m_row = r.type_m.map_or(String::new(), |t|
        format!("<tr><td>Type M:</td><td>{:.2}x</td></tr>", t));

    // Event indices for x-axis
    let x_events: Vec<usize> = (0..r.trajectory.len()).collect();

    format!(r#"<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Survival Analysis Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
body{{font-family:system-ui,-apple-system,sans-serif;max-width:1400px;margin:0 auto;padding:20px;background:#fafafa}}
h1{{color:#1a1a2e}}h2{{color:#16213e;border-bottom:1px solid #ddd;padding-bottom:5px}}
table{{margin:10px 0}}td{{padding:4px 12px}}
.plot{{background:#fff;border-radius:8px;padding:10px;margin:20px 0;box-shadow:0 1px 3px rgba(0,0,0,0.1)}}
</style></head><body>
<h1>e-RTs Survival Analysis Report</h1>
<p>{}</p>

<h2>Data</h2>
<table>
<tr><td>File:</td><td>{}</td></tr>
<tr><td>Total patients:</td><td>{}</td></tr>
<tr><td>Total events:</td><td>{} ({:.1}%)</td></tr>
<tr><td>Treatment:</td><td>{} patients, {} events ({:.1}%)</td></tr>
<tr><td>Control:</td><td>{} patients, {} events ({:.1}%)</td></tr>
</table>

<h2>Parameters</h2>
<table>
<tr><td>Burn-in:</td><td>{} events</td></tr>
<tr><td>Ramp:</td><td>{} events</td></tr>
<tr><td>Threshold:</td><td>{} (alpha={:.3})</td></tr>
<tr><td>Lambda max:</td><td>0.25 (fixed wager)</td></tr>
</table>

<h2>Results</h2>
<table>
<tr><td>Final e-value:</td><td><strong>{:.4}</strong></td></tr>
<tr><td>Minimum e-value:</td><td>{:.4}</td></tr>
<tr><td>Status:</td><td>{}</td></tr>
{}
<tr><td>Final HR:</td><td>{:.3} ({:.3}-{:.3})</td></tr>
{}
</table>

<h2>e-Value Trajectory</h2>
<div class="plot"><div id="p1" style="height:400px"></div></div>

<h2>Support (ln e-value)</h2>
<div class="plot"><div id="p2" style="height:300px"></div></div>

<script>
var x = {:?};
var y = {:?};
var y_log = y.map(v => Math.log(v));

Plotly.newPlot('p1',[{{
    type:'scatter',mode:'lines',x:x,y:y,
    line:{{color:'steelblue',width:2}},
    name:'e-value'
}}],{{
    yaxis:{{type:'log',title:'e-value'}},
    xaxis:{{title:'Events'}},
    shapes:[{{
        type:'line',x0:0,x1:{},y0:{},y1:{},
        line:{{color:'green',dash:'dash',width:2}}
    }}]
}});

Plotly.newPlot('p2',[{{
    type:'scatter',mode:'lines',x:x,y:y_log,
    line:{{color:'coral',width:2}},
    name:'ln(e-value)'
}}],{{
    yaxis:{{title:'ln(e-value)'}},
    xaxis:{{title:'Events'}},
    shapes:[{{
        type:'line',x0:0,x1:{},y0:{:.4},y1:{:.4},
        line:{{color:'green',dash:'dash',width:2}}
    }}]
}});
</script>
</body></html>"#,
        chrono_lite(),
        csv_path,
        r.n_total,
        r.n_events, r.n_events as f64 / r.n_total as f64 * 100.0,
        r.n_trt, r.events_trt, if r.n_trt > 0 { r.events_trt as f64 / r.n_trt as f64 * 100.0 } else { 0.0 },
        r.n_ctrl, r.events_ctrl, if r.n_ctrl > 0 { r.events_ctrl as f64 / r.n_ctrl as f64 * 100.0 } else { 0.0 },
        burn_in,
        ramp,
        threshold, 1.0 / threshold,
        r.final_evalue,
        r.min_evalue,
        status,
        crossing_row,
        r.hr_final, r.hr_ci.0, r.hr_ci.1,
        type_m_row,
        x_events,
        r.trajectory,
        r.n_events, threshold, threshold,
        r.n_events, threshold.ln(), threshold.ln()
    )
}
