//! Compare e-RTo vs e-RTc methods under identical conditions

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::fs::File;
use std::io::Write;

use crate::ert_core::{
    get_input, get_input_usize, get_bool, get_optional_input,
    chrono_lite, LinearERTProcess, MADProcess,
};

// ============================================================================
// COMPARISON RESULTS
// ============================================================================

#[derive(Clone)]
#[allow(dead_code)]
struct MethodStats {
    name: String,
    type1_error: f64,
    power: f64,
    avg_stop_n: f64,
    avg_effect_at_stop: f64,
    avg_effect_final: f64,
    type_m: f64,
    trajectories: Vec<Vec<f64>>,
    stop_times: Vec<usize>,
}

#[allow(dead_code)]
struct ComparisonResult {
    // Shared parameters
    n_patients: usize,
    n_sims: usize,
    mu_ctrl: f64,
    mu_trt: f64,
    sd: f64,
    min_val: f64,
    max_val: f64,
    cohen_d: f64,
    success_threshold: f64,
    burn_in: usize,
    ramp: usize,
    c_max: f64,

    // Results
    linear: MethodStats,
    mad: MethodStats,

    // Head-to-head (same trials)
    linear_wins: usize,  // Linear stopped first
    mad_wins: usize,     // MAD stopped first
    both_stopped: usize,
    neither_stopped: usize,
}

// ============================================================================
// MAIN COMPARISON FUNCTION
// ============================================================================

pub fn run() {
    println!("\n==========================================");
    println!("   COMPARE: e-RTo vs e-RTc");
    println!("==========================================\n");

    println!("Runs BOTH methods on IDENTICAL simulated data");
    println!("for a fair head-to-head comparison.\n");

    // Get parameters
    let mu_ctrl = get_input("Control Mean (μ_ctrl, e.g., 14): ");
    let mu_trt = get_input("Treatment Mean (μ_trt, e.g., 18): ");
    let sd = get_input("Standard Deviation (σ, e.g., 10): ");
    let min_val = get_input("Min bound (e.g., 0): ");
    let max_val = get_input("Max bound (e.g., 28): ");

    let cohen_d = ((mu_trt - mu_ctrl) / sd).abs();
    let mean_diff = (mu_trt - mu_ctrl).abs();

    println!("\nDesign effect:");
    println!("  Mean Difference: {:.2}", mean_diff);
    println!("  Cohen's d:       {:.2}", cohen_d);

    let n_patients = get_input_usize("\nNumber of patients per trial: ");
    let n_sims = get_input_usize("Number of simulations: ");

    println!("\nSuccess threshold (1/alpha, default = 20):");
    let success_threshold = get_input("Success threshold: ");

    let seed = get_optional_input("Seed (press Enter for random): ");

    // Fixed tuning parameters
    let burn_in: usize = 20;
    let ramp: usize = 50;
    let c_max: f64 = 0.6;

    println!("\n--- Fixed Parameters ---");
    println!("Burn-in: {} (default)", burn_in);
    println!("Ramp:    {} (default)", ramp);
    println!("c_max:   {} (default)", c_max);

    // Run comparison
    println!("\n==========================================");
    println!("   RUNNING COMPARISON");
    println!("==========================================\n");

    let result = run_comparison(
        n_patients, n_sims,
        mu_ctrl, mu_trt, sd, min_val, max_val,
        success_threshold, burn_in, ramp, c_max, seed,
    );

    // Print results
    print_comparison(&result);

    // Generate HTML report
    if get_bool("\nGenerate HTML report?") {
        let html = build_comparison_html(&result);
        let path = "comparison_report.html";
        let mut file = File::create(path).unwrap();
        file.write_all(html.as_bytes()).unwrap();
        println!("\n>> Report saved: {}", path);
    }

    println!("\n==========================================");
}

fn run_comparison(
    n_patients: usize,
    n_sims: usize,
    mu_ctrl: f64,
    mu_trt: f64,
    sd: f64,
    min_val: f64,
    max_val: f64,
    success_threshold: f64,
    burn_in: usize,
    ramp: usize,
    c_max: f64,
    seed: Option<u64>,
) -> ComparisonResult {
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    let cohen_d = ((mu_trt - mu_ctrl) / sd).abs();

    // =========== Phase 1: Type I Error (null hypothesis) ===========
    print!("Phase 1: Type I Error (H0: no effect)... ");
    std::io::stdout().flush().unwrap();

    let mut linear_null_reject = 0;
    let mut mad_null_reject = 0;

    for _ in 0..n_sims {
        let mut linear = LinearERTProcess::new(burn_in, ramp, min_val, max_val);
        let mut mad_proc = MADProcess::new(burn_in, ramp, c_max);

        let mut linear_crossed = false;
        let mut mad_crossed = false;

        for i in 1..=n_patients {
            let is_trt = rng.gen_bool(0.5);
            // Under null: both arms have same mean
            let outcome = generate_outcome(&mut rng, mu_ctrl, sd, min_val, max_val);

            linear.update(i, outcome, is_trt);
            mad_proc.update(i, outcome, is_trt);

            if !linear_crossed && linear.wealth >= success_threshold {
                linear_crossed = true;
            }
            if !mad_crossed && mad_proc.wealth >= success_threshold {
                mad_crossed = true;
            }
        }

        if linear_crossed { linear_null_reject += 1; }
        if mad_crossed { mad_null_reject += 1; }
    }

    println!("Done.");
    println!("  e-RTo: {:.2}%", (linear_null_reject as f64 / n_sims as f64) * 100.0);
    println!("  e-RTc: {:.2}%", (mad_null_reject as f64 / n_sims as f64) * 100.0);

    // =========== Phase 2: Power (alternative hypothesis) ===========
    print!("\nPhase 2: Power (H1: treatment works)");
    std::io::stdout().flush().unwrap();

    let mut linear_successes = 0;
    let mut mad_successes = 0;
    let mut linear_wins = 0;
    let mut mad_wins = 0;
    let mut both_stopped = 0;
    let mut neither_stopped = 0;

    let mut linear_stop_times: Vec<usize> = Vec::new();
    let mut mad_stop_times: Vec<usize> = Vec::new();
    let mut linear_effects_stop: Vec<f64> = Vec::new();
    let mut mad_effects_stop: Vec<f64> = Vec::new();
    let mut linear_effects_final: Vec<f64> = Vec::new();
    let mut mad_effects_final: Vec<f64> = Vec::new();

    let mut linear_trajectories: Vec<Vec<f64>> = Vec::new();
    let mut mad_trajectories: Vec<Vec<f64>> = Vec::new();

    let pb_interval = (n_sims / 20).max(1);

    for sim in 0..n_sims {
        if sim % pb_interval == 0 { print!("."); std::io::stdout().flush().unwrap(); }

        let mut linear = LinearERTProcess::new(burn_in, ramp, min_val, max_val);
        let mut mad_proc = MADProcess::new(burn_in, ramp, c_max);

        let mut linear_traj = vec![1.0];
        let mut mad_traj = vec![1.0];

        let mut linear_crossed = false;
        let mut mad_crossed = false;
        let mut linear_stop: Option<usize> = None;
        let mut mad_stop: Option<usize> = None;
        let mut linear_eff_stop: Option<f64> = None;
        let mut mad_eff_stop: Option<f64> = None;

        for i in 1..=n_patients {
            let is_trt = rng.gen_bool(0.5);
            let mu = if is_trt { mu_trt } else { mu_ctrl };
            let outcome = generate_outcome(&mut rng, mu, sd, min_val, max_val);

            // SAME outcome to BOTH methods
            linear.update(i, outcome, is_trt);
            mad_proc.update(i, outcome, is_trt);

            linear_traj.push(linear.wealth);
            mad_traj.push(mad_proc.wealth);

            if !linear_crossed && linear.wealth >= success_threshold {
                linear_crossed = true;
                linear_stop = Some(i);
                linear_eff_stop = Some(linear.current_effect());
            }
            if !mad_crossed && mad_proc.wealth >= success_threshold {
                mad_crossed = true;
                mad_stop = Some(i);
                mad_eff_stop = Some(mad_proc.current_effect(sd));
            }
        }

        // Record results
        if linear_crossed {
            linear_successes += 1;
            linear_stop_times.push(linear_stop.unwrap());
            linear_effects_stop.push(linear_eff_stop.unwrap().abs());
        }
        if mad_crossed {
            mad_successes += 1;
            mad_stop_times.push(mad_stop.unwrap());
            mad_effects_stop.push(mad_eff_stop.unwrap().abs());
        }

        linear_effects_final.push(linear.current_effect().abs());
        mad_effects_final.push(mad_proc.current_effect(sd).abs());

        // Head-to-head comparison
        match (linear_crossed, mad_crossed) {
            (true, true) => {
                both_stopped += 1;
                if linear_stop.unwrap() < mad_stop.unwrap() {
                    linear_wins += 1;
                } else if mad_stop.unwrap() < linear_stop.unwrap() {
                    mad_wins += 1;
                }
                // ties don't count as wins
            }
            (true, false) => { linear_wins += 1; }
            (false, true) => { mad_wins += 1; }
            (false, false) => { neither_stopped += 1; }
        }

        // Store first 50 trajectories for plotting
        if sim < 50 {
            linear_trajectories.push(linear_traj);
            mad_trajectories.push(mad_traj);
        }
    }

    println!(" Done.");

    // Compute statistics
    let linear_power = (linear_successes as f64 / n_sims as f64) * 100.0;
    let mad_power = (mad_successes as f64 / n_sims as f64) * 100.0;

    let linear_avg_stop = if !linear_stop_times.is_empty() {
        linear_stop_times.iter().sum::<usize>() as f64 / linear_stop_times.len() as f64
    } else { 0.0 };

    let mad_avg_stop = if !mad_stop_times.is_empty() {
        mad_stop_times.iter().sum::<usize>() as f64 / mad_stop_times.len() as f64
    } else { 0.0 };

    let linear_avg_eff_stop = if !linear_effects_stop.is_empty() {
        linear_effects_stop.iter().sum::<f64>() / linear_effects_stop.len() as f64
    } else { 0.0 };

    let mad_avg_eff_stop = if !mad_effects_stop.is_empty() {
        mad_effects_stop.iter().sum::<f64>() / mad_effects_stop.len() as f64
    } else { 0.0 };

    let linear_avg_eff_final = linear_effects_final.iter().sum::<f64>() / linear_effects_final.len() as f64;
    let mad_avg_eff_final = mad_effects_final.iter().sum::<f64>() / mad_effects_final.len() as f64;

    let linear_type_m = if linear_avg_eff_final > 0.0 { linear_avg_eff_stop / linear_avg_eff_final } else { 0.0 };
    let mad_type_m = if mad_avg_eff_final > 0.0 { mad_avg_eff_stop / mad_avg_eff_final } else { 0.0 };

    ComparisonResult {
        n_patients,
        n_sims,
        mu_ctrl,
        mu_trt,
        sd,
        min_val,
        max_val,
        cohen_d,
        success_threshold,
        burn_in,
        ramp,
        c_max,
        linear: MethodStats {
            name: "e-RTo".to_string(),
            type1_error: (linear_null_reject as f64 / n_sims as f64) * 100.0,
            power: linear_power,
            avg_stop_n: linear_avg_stop,
            avg_effect_at_stop: linear_avg_eff_stop,
            avg_effect_final: linear_avg_eff_final,
            type_m: linear_type_m,
            trajectories: linear_trajectories,
            stop_times: linear_stop_times,
        },
        mad: MethodStats {
            name: "e-RTc".to_string(),
            type1_error: (mad_null_reject as f64 / n_sims as f64) * 100.0,
            power: mad_power,
            avg_stop_n: mad_avg_stop,
            avg_effect_at_stop: mad_avg_eff_stop,
            avg_effect_final: mad_avg_eff_final,
            type_m: mad_type_m,
            trajectories: mad_trajectories,
            stop_times: mad_stop_times,
        },
        linear_wins,
        mad_wins,
        both_stopped,
        neither_stopped,
    }
}

fn generate_outcome<R: Rng>(rng: &mut R, mu: f64, sd: f64, min_val: f64, max_val: f64) -> f64 {
    // Approximate normal with uniform (good enough for comparison)
    // Using sum of 12 uniforms for ~N(0,1)
    let u: f64 = (0..12).map(|_| rng.gen::<f64>()).sum::<f64>() - 6.0;
    let outcome = mu + sd * u;
    outcome.clamp(min_val, max_val)
}

fn print_comparison(r: &ComparisonResult) {
    println!("\n==========================================");
    println!("   COMPARISON RESULTS");
    println!("==========================================");

    println!("\n--- Design ---");
    println!("Patients:       {}", r.n_patients);
    println!("Simulations:    {}", r.n_sims);
    println!("Control Mean:   {:.2}", r.mu_ctrl);
    println!("Treatment Mean: {:.2}", r.mu_trt);
    println!("SD:             {:.2}", r.sd);
    println!("Bounds:         [{:.1}, {:.1}]", r.min_val, r.max_val);
    println!("Cohen's d:      {:.2}", r.cohen_d);
    println!("Threshold:      {:.0}", r.success_threshold);

    println!("\n--- Type I Error (should be ~{:.1}%) ---", 100.0 / r.success_threshold);
    println!("  e-RTo:  {:.2}%", r.linear.type1_error);
    println!("  e-RTc:  {:.2}%", r.mad.type1_error);

    println!("\n--- Power ---");
    println!("  e-RTo:  {:.1}%", r.linear.power);
    println!("  e-RTc:  {:.1}%", r.mad.power);
    let power_diff = r.linear.power - r.mad.power;
    if power_diff.abs() > 0.5 {
        println!("  Δ Power:    {:.1}% ({})", power_diff.abs(),
            if power_diff > 0.0 { "e-RTo wins" } else { "e-RTc wins" });
    } else {
        println!("  Δ Power:    ~0% (tied)");
    }

    println!("\n--- Average Stopping Time (when successful) ---");
    println!("  e-RTo:  {:.0} patients ({:.0}% of N)", r.linear.avg_stop_n, (r.linear.avg_stop_n / r.n_patients as f64) * 100.0);
    println!("  e-RTc:  {:.0} patients ({:.0}% of N)", r.mad.avg_stop_n, (r.mad.avg_stop_n / r.n_patients as f64) * 100.0);

    println!("\n--- Type M Error (effect magnification) ---");
    println!("  e-RTo:  {:.2}x", r.linear.type_m);
    println!("  e-RTc:  {:.2}x", r.mad.type_m);

    println!("\n--- Head-to-Head (same trials) ---");
    println!("  e-RTo stopped first: {} ({:.1}%)", r.linear_wins, (r.linear_wins as f64 / r.n_sims as f64) * 100.0);
    println!("  e-RTc stopped first: {} ({:.1}%)", r.mad_wins, (r.mad_wins as f64 / r.n_sims as f64) * 100.0);
    println!("  Both stopped (total):    {}", r.both_stopped);
    println!("  Neither stopped:         {} ({:.1}%)", r.neither_stopped, (r.neither_stopped as f64 / r.n_sims as f64) * 100.0);
}

fn build_comparison_html(r: &ComparisonResult) -> String {
    let timestamp = chrono_lite();

    // Prepare trajectory data (downsample for plotting)
    let step = (r.n_patients / 100).max(1);
    let x_axis: Vec<usize> = (0..=r.n_patients).step_by(step).collect();
    let x_json = format!("{:?}", x_axis);

    // Build trajectory traces
    let mut linear_traces = String::new();
    let mut mad_traces = String::new();

    for (idx, traj) in r.linear.trajectories.iter().take(30).enumerate() {
        let y: Vec<f64> = x_axis.iter().map(|&i| traj.get(i).copied().unwrap_or(1.0)).collect();
        let opacity = if idx < 5 { 0.7 } else { 0.3 };
        linear_traces.push_str(&format!(
            "{{type:'scatter',mode:'lines',x:{},y:{:?},line:{{color:'rgba(31,119,180,{})',width:1}},showlegend:false}},",
            x_json, y, opacity
        ));
    }

    for (idx, traj) in r.mad.trajectories.iter().take(30).enumerate() {
        let y: Vec<f64> = x_axis.iter().map(|&i| traj.get(i).copied().unwrap_or(1.0)).collect();
        let opacity = if idx < 5 { 0.7 } else { 0.3 };
        mad_traces.push_str(&format!(
            "{{type:'scatter',mode:'lines',x:{},y:{:?},line:{{color:'rgba(255,127,14,{})',width:1}},showlegend:false}},",
            x_json, y, opacity
        ));
    }

    // Stop times histograms
    let linear_stops_json = format!("{:?}", r.linear.stop_times);
    let mad_stops_json = format!("{:?}", r.mad.stop_times);

    // Winner determination
    let power_winner = if (r.linear.power - r.mad.power).abs() < 1.0 {
        "Tied"
    } else if r.linear.power > r.mad.power {
        "e-RTo"
    } else {
        "e-RTc"
    };

    let speed_winner = if r.linear.avg_stop_n < r.mad.avg_stop_n * 0.95 {
        "e-RTo"
    } else if r.mad.avg_stop_n < r.linear.avg_stop_n * 0.95 {
        "e-RTc"
    } else {
        "Tied"
    };

    format!(r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Method Comparison: e-RTo vs e-RTc</title>
    <script src="https://cdn.plot.ly/plotly-2.12.1.min.js"></script>
    <style>
        body {{ font-family: monospace; max-width: 1200px; margin: 0 auto; padding: 20px; background: #fff; }}
        h2 {{ border-bottom: 1px solid #ccc; padding-bottom: 5px; margin-top: 20px; }}
        table {{ border-collapse: collapse; margin: 10px 0; }}
        th, td {{ padding: 6px 12px; text-align: left; }}
        th {{ background: #f8f8f8; }}
        .linear {{ color: #1f77b4; font-weight: bold; }}
        .mad {{ color: #ff7f0e; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Method Comparison: e-RTo vs e-RTc</h1>
        <p class="timestamp">Generated: {}</p>

        <h2>Simulation Parameters</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Patients per trial</td><td>{}</td></tr>
            <tr><td>Number of simulations</td><td>{}</td></tr>
            <tr><td>Control Mean (μ_ctrl)</td><td>{:.2}</td></tr>
            <tr><td>Treatment Mean (μ_trt)</td><td>{:.2}</td></tr>
            <tr><td>Standard Deviation (σ)</td><td>{:.2}</td></tr>
            <tr><td>Bounds [min, max]</td><td>[{:.1}, {:.1}]</td></tr>
            <tr><td>Cohen's d</td><td><strong>{:.2}</strong></td></tr>
            <tr><td>Success Threshold</td><td>{:.0}</td></tr>
        </table>

        <h2>Key Results</h2>
        <div class="comparison-grid">
            <div class="metric-card">
                <div class="metric-value linear">{:.1}%</div>
                <div class="metric-label">e-RTo Power</div>
            </div>
            <div class="metric-card">
                <div class="metric-value mad">{:.1}%</div>
                <div class="metric-label">e-RTc Power</div>
            </div>
            <div class="metric-card">
                <div class="metric-value linear">{:.0}</div>
                <div class="metric-label">e-RTo Avg Stop</div>
            </div>
            <div class="metric-card">
                <div class="metric-value mad">{:.0}</div>
                <div class="metric-label">e-RTc Avg Stop</div>
            </div>
        </div>

        <h2>Detailed Comparison</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th class="linear">e-RTo</th>
                <th class="mad">e-RTc</th>
                <th>Winner</th>
            </tr>
            <tr>
                <td>Type I Error</td>
                <td>{:.2}%</td>
                <td>{:.2}%</td>
                <td>{}</td>
            </tr>
            <tr class="{}">
                <td>Power</td>
                <td>{:.1}%</td>
                <td>{:.1}%</td>
                <td><strong>{}</strong></td>
            </tr>
            <tr class="{}">
                <td>Avg Stopping Time</td>
                <td>{:.0} ({:.0}%)</td>
                <td>{:.0} ({:.0}%)</td>
                <td><strong>{}</strong></td>
            </tr>
            <tr>
                <td>Type M Error</td>
                <td>{:.2}x</td>
                <td>{:.2}x</td>
                <td>{}</td>
            </tr>
        </table>

        <h2>Head-to-Head (Same Trials)</h2>
        <table>
            <tr><td>e-RTo stopped first</td><td><strong>{}</strong> ({:.1}%)</td></tr>
            <tr><td>e-RTc stopped first</td><td><strong>{}</strong> ({:.1}%)</td></tr>
            <tr><td>Both stopped</td><td>{}</td></tr>
            <tr><td>Neither stopped</td><td>{}</td></tr>
        </table>

        <h2>e-Value Trajectories</h2>

        <h3 class="linear">e-RTo Trajectories (30 sample trials)</h3>
        <div id="plot1" style="width:100%;height:400px;"></div>

        <h3 class="mad">e-RTc Trajectories (30 sample trials)</h3>
        <div id="plot2" style="width:100%;height:400px;"></div>

        <h2>Stopping Time Distributions</h2>
        <div id="plot3" style="width:100%;height:400px;"></div>
    </div>

    <script>
        // e-RTo trajectories
        Plotly.newPlot('plot1', [
            {}
            {{type:'scatter',mode:'lines',x:[0,{}],y:[{},{}],line:{{color:'green',width:2,dash:'dash'}},name:'Threshold'}}
        ], {{
            yaxis: {{type:'log',title:'e-value',range:[-1,2.5]}},
            xaxis: {{title:'Patients Enrolled'}},
            title: 'e-RTo Sample Trajectories'
        }});

        // MAD trajectories
        Plotly.newPlot('plot2', [
            {}
            {{type:'scatter',mode:'lines',x:[0,{}],y:[{},{}],line:{{color:'green',width:2,dash:'dash'}},name:'Threshold'}}
        ], {{
            yaxis: {{type:'log',title:'e-value',range:[-1,2.5]}},
            xaxis: {{title:'Patients Enrolled'}},
            title: 'e-RTc Sample Trajectories'
        }});

        // Stopping times comparison
        Plotly.newPlot('plot3', [
            {{
                type: 'histogram',
                x: {},
                name: 'e-RTo',
                opacity: 0.7,
                marker: {{color: '#1f77b4'}}
            }},
            {{
                type: 'histogram',
                x: {},
                name: 'e-RTc',
                opacity: 0.7,
                marker: {{color: '#ff7f0e'}}
            }}
        ], {{
            barmode: 'overlay',
            xaxis: {{title: 'Stopping Time (Patient Number)'}},
            yaxis: {{title: 'Count'}},
            title: 'Stopping Time Distributions'
        }});
    </script>
</body>
</html>"#,
        timestamp,
        r.n_patients, r.n_sims,
        r.mu_ctrl, r.mu_trt, r.sd, r.min_val, r.max_val, r.cohen_d,
        r.success_threshold,
        // Key metrics
        r.linear.power, r.mad.power,
        r.linear.avg_stop_n, r.mad.avg_stop_n,
        // Type I
        r.linear.type1_error, r.mad.type1_error,
        if r.linear.type1_error < r.mad.type1_error { "e-RTo" } else { "e-RTc" },
        // Power
        if power_winner == "e-RTo" { "winner" } else { "" },
        r.linear.power, r.mad.power, power_winner,
        // Speed
        if speed_winner == "e-RTo" { "winner" } else { "" },
        r.linear.avg_stop_n, (r.linear.avg_stop_n / r.n_patients as f64) * 100.0,
        r.mad.avg_stop_n, (r.mad.avg_stop_n / r.n_patients as f64) * 100.0,
        speed_winner,
        // Type M
        r.linear.type_m, r.mad.type_m,
        if r.linear.type_m < r.mad.type_m { "e-RTo" } else { "e-RTc" },
        // Head-to-head
        r.linear_wins, (r.linear_wins as f64 / r.n_sims as f64) * 100.0,
        r.mad_wins, (r.mad_wins as f64 / r.n_sims as f64) * 100.0,
        r.both_stopped, r.neither_stopped,
        // Plots
        linear_traces, r.n_patients, r.success_threshold, r.success_threshold,
        mad_traces, r.n_patients, r.success_threshold, r.success_threshold,
        linear_stops_json, mad_stops_json
    )
}
