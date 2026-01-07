// analyze_multistate.rs - Analyze multi-state trial data from CSV
//
// CSV format: patient_id,time,state,treatment
// States are indices 0 to N-1 (ordered worst to best)
// Transitions between timepoints are classified as good (→higher), bad (→lower), or neutral (same)

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};

use crate::agnostic::{AgnosticERT, Signal};
use crate::ert_core::chrono_lite;

// === DATA STRUCTURES ===

#[derive(Debug, Clone)]
struct PatientRecord {
    patient_id: String,
    time: i32,
    state: usize,
    treatment: u8,  // 0 = control, 1 = treatment
}

#[derive(Debug, Clone)]
struct Transition {
    patient_id: String,
    from_state: usize,
    to_state: usize,
    from_time: i32,
    to_time: i32,
    treatment: u8,
}

impl Transition {
    fn is_good(&self) -> bool {
        self.to_state > self.from_state
    }

    fn is_bad(&self) -> bool {
        self.to_state < self.from_state
    }
}

struct AnalysisResult {
    // Data summary
    n_patients: usize,
    n_control: usize,
    n_treatment: usize,
    n_transitions: usize,
    n_good: usize,
    n_bad: usize,

    // State info
    state_names: Vec<String>,
    n_states: usize,

    // e-process results
    e_values: Vec<f64>,
    crossed: bool,
    crossing_index: Option<usize>,
    final_e: f64,

    // State distributions at final time
    ctrl_final_dist: Vec<f64>,
    trt_final_dist: Vec<f64>,

    // Effect estimates
    prop_or: f64,
    mann_whitney: f64,
}

// === CSV PARSING ===

fn parse_csv(path: &str) -> Result<Vec<PatientRecord>, String> {
    let file = File::open(path).map_err(|e| format!("Cannot open file: {}", e))?;
    let reader = BufReader::new(file);
    let mut records = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| format!("Read error at line {}: {}", i + 1, e))?;

        // Skip header
        if i == 0 && line.to_lowercase().contains("patient") {
            continue;
        }

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 4 {
            return Err(format!("Line {} has {} columns, expected 4", i + 1, parts.len()));
        }

        let patient_id = parts[0].trim().to_string();
        let time: i32 = parts[1].trim().parse()
            .map_err(|_| format!("Invalid time at line {}", i + 1))?;
        let state: usize = parts[2].trim().parse()
            .map_err(|_| format!("Invalid state at line {}", i + 1))?;
        let treatment: u8 = parts[3].trim().parse()
            .map_err(|_| format!("Invalid treatment at line {}", i + 1))?;

        records.push(PatientRecord { patient_id, time, state, treatment });
    }

    Ok(records)
}

fn extract_transitions(records: &[PatientRecord]) -> Vec<Transition> {
    // Group by patient
    let mut by_patient: HashMap<String, Vec<&PatientRecord>> = HashMap::new();
    for r in records {
        by_patient.entry(r.patient_id.clone()).or_default().push(r);
    }

    let mut transitions = Vec::new();

    for (pid, mut patient_records) in by_patient {
        // Sort by time
        patient_records.sort_by_key(|r| r.time);

        // Extract transitions
        for i in 1..patient_records.len() {
            let prev = patient_records[i - 1];
            let curr = patient_records[i];

            // Only record actual state changes
            if prev.state != curr.state {
                transitions.push(Transition {
                    patient_id: pid.clone(),
                    from_state: prev.state,
                    to_state: curr.state,
                    from_time: prev.time,
                    to_time: curr.time,
                    treatment: curr.treatment,
                });
            }
        }
    }

    // Sort transitions by time for sequential analysis
    transitions.sort_by_key(|t| t.to_time);

    transitions
}

// === E-PROCESS ANALYSIS ===

fn run_e_process(transitions: &[Transition], threshold: f64, burn_in: usize, ramp: usize) -> (Vec<f64>, bool, Option<usize>) {
    let mut ert = AgnosticERT::new(burn_in, ramp, threshold);
    let mut e_values = Vec::new();
    let mut crossed = false;
    let mut crossing_index = None;

    for (i, t) in transitions.iter().enumerate() {
        // Good transition = movement to better state
        // Bad transition = movement to worse state
        // Signal is (arm, is_good)
        if t.is_good() {
            let signal = if t.treatment == 1 {
                Signal::treatment(true)
            } else {
                Signal::control(true)
            };
            ert.observe(signal);
        } else if t.is_bad() {
            let signal = if t.treatment == 1 {
                Signal::treatment(false)
            } else {
                Signal::control(false)
            };
            ert.observe(signal);
        }
        // Neutral (same state) = no observation

        e_values.push(ert.wealth());

        if !crossed && ert.wealth() >= threshold {
            crossed = true;
            crossing_index = Some(i);
        }
    }

    (e_values, crossed, crossing_index)
}

// === EFFECT ESTIMATION ===

fn calculate_final_distributions(records: &[PatientRecord], n_states: usize) -> (Vec<f64>, Vec<f64>) {
    // Group by patient
    let mut by_patient: HashMap<String, Vec<&PatientRecord>> = HashMap::new();
    for r in records {
        by_patient.entry(r.patient_id.clone()).or_default().push(r);
    }

    let mut ctrl_counts = vec![0usize; n_states];
    let mut trt_counts = vec![0usize; n_states];
    let mut n_ctrl = 0;
    let mut n_trt = 0;

    for (_, patient_records) in by_patient {
        // Get last observation
        if let Some(last) = patient_records.iter().max_by_key(|r| r.time) {
            if last.state < n_states {
                if last.treatment == 1 {
                    trt_counts[last.state] += 1;
                    n_trt += 1;
                } else {
                    ctrl_counts[last.state] += 1;
                    n_ctrl += 1;
                }
            }
        }
    }

    let ctrl_dist: Vec<f64> = ctrl_counts.iter().map(|&c| c as f64 / n_ctrl.max(1) as f64).collect();
    let trt_dist: Vec<f64> = trt_counts.iter().map(|&c| c as f64 / n_trt.max(1) as f64).collect();

    (ctrl_dist, trt_dist)
}

fn calculate_proportional_or(ctrl_dist: &[f64], trt_dist: &[f64]) -> f64 {
    let n = ctrl_dist.len();
    if n == 0 { return 1.0; }

    let mut numer = 0.0;
    let mut denom = 0.0;

    for k in 0..n {
        // P(T >= k)
        let p_trt_ge: f64 = trt_dist[k..].iter().sum();
        let p_ctrl_ge: f64 = ctrl_dist[k..].iter().sum();

        // P(T < k)
        let p_trt_lt: f64 = trt_dist[..k].iter().sum();
        let p_ctrl_lt: f64 = ctrl_dist[..k].iter().sum();

        // Weight by marginal probability at this cut
        let weight = (p_ctrl_ge + p_trt_ge) * (p_ctrl_lt + p_trt_lt);

        if weight > 0.0 && p_ctrl_ge > 0.0 && p_trt_lt > 0.0 {
            numer += weight * p_trt_ge * p_ctrl_lt;
            denom += weight * p_ctrl_ge * p_trt_lt;
        }
    }

    if denom > 0.0 { numer / denom } else { 1.0 }
}

fn calculate_mann_whitney(ctrl_dist: &[f64], trt_dist: &[f64]) -> f64 {
    let n = ctrl_dist.len();
    let mut prob = 0.0;

    for i in 0..n {
        for j in 0..n {
            if j > i {
                // Treatment better than control
                prob += trt_dist[j] * ctrl_dist[i];
            } else if j == i {
                // Tie: split 50-50
                prob += 0.5 * trt_dist[j] * ctrl_dist[i];
            }
        }
    }

    prob
}

// === HTML REPORT ===

fn generate_html_report(result: &AnalysisResult, path: &str) -> std::io::Result<()> {
    let mut file = File::create(path)?;

    let timestamp = chrono_lite();

    // E-value trajectory for chart
    let e_data: Vec<String> = result.e_values.iter().map(|e| format!("{:.4}", e)).collect();
    let e_json = format!("[{}]", e_data.join(","));

    // State distribution bars
    let mut ctrl_bars = String::new();
    let mut trt_bars = String::new();
    for i in 0..result.n_states {
        let name = &result.state_names[i];
        let ctrl_pct = result.ctrl_final_dist[i] * 100.0;
        let trt_pct = result.trt_final_dist[i] * 100.0;
        ctrl_bars.push_str(&format!(
            "<div style='display:flex;align-items:center;margin:2px 0'>\
             <span style='width:80px'>{}</span>\
             <div style='background:#e74c3c;height:20px;width:{}%'></div>\
             <span style='margin-left:5px'>{:.1}%</span></div>",
            name, ctrl_pct * 2.0, ctrl_pct
        ));
        trt_bars.push_str(&format!(
            "<div style='display:flex;align-items:center;margin:2px 0'>\
             <span style='width:80px'>{}</span>\
             <div style='background:#27ae60;height:20px;width:{}%'></div>\
             <span style='margin-left:5px'>{:.1}%</span></div>",
            name, trt_pct * 2.0, trt_pct
        ));
    }

    let crossing_text = if result.crossed {
        format!("Crossed at transition {} (e={:.2})",
                result.crossing_index.unwrap_or(0) + 1,
                result.e_values[result.crossing_index.unwrap_or(0)])
    } else {
        format!("Did not cross (final e={:.4})", result.final_e)
    };

    let html = format!(r#"<!DOCTYPE html>
<html>
<head>
    <title>Multi-State Trial Analysis</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               max-width: 1000px; margin: 40px auto; padding: 20px; background: #f5f5f5; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; margin: 20px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 0; }}
        .stat {{ display: inline-block; margin: 10px 20px 10px 0; }}
        .stat-value {{ font-size: 28px; font-weight: bold; color: #2980b9; }}
        .stat-label {{ color: #7f8c8d; font-size: 14px; }}
        .crossed {{ color: #27ae60; font-weight: bold; }}
        .not-crossed {{ color: #e74c3c; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #ecf0f1; }}
        th {{ background: #f8f9fa; }}
    </style>
</head>
<body>
    <h1>Multi-State Trial Analysis (e-RTms)</h1>
    <p style="color:#7f8c8d">{timestamp}</p>

    <div class="card">
        <h2>Data Summary</h2>
        <div class="stat">
            <div class="stat-value">{n_patients}</div>
            <div class="stat-label">Patients</div>
        </div>
        <div class="stat">
            <div class="stat-value">{n_ctrl} / {n_trt}</div>
            <div class="stat-label">Control / Treatment</div>
        </div>
        <div class="stat">
            <div class="stat-value">{n_transitions}</div>
            <div class="stat-label">Transitions</div>
        </div>
        <div class="stat">
            <div class="stat-value">{n_good} / {n_bad}</div>
            <div class="stat-label">Good / Bad</div>
        </div>
        <p><strong>States:</strong> {states}</p>
    </div>

    <div class="card">
        <h2>e-Process Result</h2>
        <p class="{crossed_class}">{crossing_text}</p>
        <canvas id="eChart" height="100"></canvas>
    </div>

    <div class="card">
        <h2>Final State Distribution</h2>
        <div class="grid">
            <div>
                <h3 style="color:#e74c3c">Control (n={n_ctrl})</h3>
                {ctrl_bars}
            </div>
            <div>
                <h3 style="color:#27ae60">Treatment (n={n_trt})</h3>
                {trt_bars}
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Effect Estimates</h2>
        <table>
            <tr><th>Metric</th><th>Value</th><th>Interpretation</th></tr>
            <tr>
                <td>Proportional Odds Ratio</td>
                <td><strong>{prop_or:.2}</strong></td>
                <td>{or_interp}</td>
            </tr>
            <tr>
                <td>Mann-Whitney P(T&gt;C)</td>
                <td><strong>{mw:.1}%</strong></td>
                <td>{mw_interp}</td>
            </tr>
        </table>
    </div>

    <script>
        new Chart(document.getElementById('eChart'), {{
            type: 'line',
            data: {{
                labels: {e_json}.map((_, i) => i + 1),
                datasets: [{{
                    label: 'e-value',
                    data: {e_json},
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    fill: true,
                    tension: 0.1
                }}, {{
                    label: 'Threshold',
                    data: Array({n_transitions}).fill(20),
                    borderColor: '#e74c3c',
                    borderDash: [5, 5],
                    fill: false,
                    pointRadius: 0
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{
                    y: {{
                        type: 'logarithmic',
                        min: 0.1,
                        title: {{ display: true, text: 'e-value' }}
                    }},
                    x: {{ title: {{ display: true, text: 'Transition' }} }}
                }}
            }}
        }});
    </script>
</body>
</html>"#,
        timestamp = timestamp,
        n_patients = result.n_patients,
        n_ctrl = result.n_control,
        n_trt = result.n_treatment,
        n_transitions = result.n_transitions,
        n_good = result.n_good,
        n_bad = result.n_bad,
        states = result.state_names.join(" < "),
        crossed_class = if result.crossed { "crossed" } else { "not-crossed" },
        crossing_text = crossing_text,
        ctrl_bars = ctrl_bars,
        trt_bars = trt_bars,
        prop_or = result.prop_or,
        or_interp = if result.prop_or > 1.0 { "Treatment favored" } else if result.prop_or < 1.0 { "Control favored" } else { "No difference" },
        mw = result.mann_whitney * 100.0,
        mw_interp = if result.mann_whitney > 0.5 { "Treatment tends to have better outcomes" } else { "Control tends to have better outcomes" },
        e_json = e_json,
    );

    file.write_all(html.as_bytes())?;
    Ok(())
}

// === MAIN ENTRY ===

pub fn run(path: &str, state_names: Option<Vec<String>>, threshold: f64, burn_in: usize, ramp: usize, no_report: bool) {
    println!("\n==========================================");
    println!("   ANALYZE MULTI-STATE TRIAL DATA");
    println!("==========================================\n");

    println!("Reading {}...", path);

    let records = match parse_csv(path) {
        Ok(r) => r,
        Err(e) => {
            println!("Error: {}", e);
            return;
        }
    };

    if records.is_empty() {
        println!("Error: No data found in CSV");
        return;
    }

    // Determine number of states
    let max_state = records.iter().map(|r| r.state).max().unwrap_or(0);
    let n_states = max_state + 1;

    // State names
    let state_names = state_names.unwrap_or_else(|| {
        (0..n_states).map(|i| format!("State{}", i)).collect()
    });

    if state_names.len() != n_states {
        println!("Warning: {} state names provided but {} states found in data",
                 state_names.len(), n_states);
    }

    // Count patients
    let mut patients: HashMap<String, u8> = HashMap::new();
    for r in &records {
        patients.insert(r.patient_id.clone(), r.treatment);
    }
    let n_patients = patients.len();
    let n_control = patients.values().filter(|&&t| t == 0).count();
    let n_treatment = patients.values().filter(|&&t| t == 1).count();

    println!("Found {} patients ({} control, {} treatment)", n_patients, n_control, n_treatment);
    println!("States: {} (ordered worst to best)", state_names.join(" < "));

    // Extract transitions
    let transitions = extract_transitions(&records);
    let n_transitions = transitions.len();
    let n_good = transitions.iter().filter(|t| t.is_good()).count();
    let n_bad = transitions.iter().filter(|t| t.is_bad()).count();

    println!("Transitions: {} ({} good, {} bad, {} neutral)",
             n_transitions, n_good, n_bad, n_transitions - n_good - n_bad);

    if transitions.is_empty() {
        println!("Error: No transitions found");
        return;
    }

    // Run e-process
    println!("\nRunning e-process (threshold={}, burn-in={}, ramp={})...", threshold, burn_in, ramp);
    let (e_values, crossed, crossing_index) = run_e_process(&transitions, threshold, burn_in, ramp);
    let final_e = *e_values.last().unwrap_or(&1.0);

    // Final state distributions
    let (ctrl_dist, trt_dist) = calculate_final_distributions(&records, n_states);

    // Effect estimates
    let prop_or = calculate_proportional_or(&ctrl_dist, &trt_dist);
    let mann_whitney = calculate_mann_whitney(&ctrl_dist, &trt_dist);

    // Print results
    println!("\n--- Results ---");
    if crossed {
        println!("CROSSED at transition {} (e={:.2})",
                 crossing_index.unwrap_or(0) + 1,
                 e_values[crossing_index.unwrap_or(0)]);
    } else {
        println!("Did not cross threshold (final e={:.4})", final_e);
    }

    println!("\n--- Final State Distribution ---");
    print!("Control:   ");
    for (i, &p) in ctrl_dist.iter().enumerate() {
        print!("{}={:.1}% ", state_names[i], p * 100.0);
    }
    println!();
    print!("Treatment: ");
    for (i, &p) in trt_dist.iter().enumerate() {
        print!("{}={:.1}% ", state_names[i], p * 100.0);
    }
    println!();

    println!("\n--- Effect Estimates ---");
    println!("Proportional OR:     {:.2}", prop_or);
    println!("Mann-Whitney P(T>C): {:.1}%", mann_whitney * 100.0);

    // Generate report
    if !no_report {
        let result = AnalysisResult {
            n_patients,
            n_control,
            n_treatment,
            n_transitions,
            n_good,
            n_bad,
            state_names,
            n_states,
            e_values,
            crossed,
            crossing_index,
            final_e,
            ctrl_final_dist: ctrl_dist,
            trt_final_dist: trt_dist,
            prop_or,
            mann_whitney,
        };

        if let Err(e) = generate_html_report(&result, "multistate_analysis_report.html") {
            println!("\nWarning: Could not generate report: {}", e);
        } else {
            println!("\n>> Saved: multistate_analysis_report.html");
        }
    }
}
