use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::io::{self, Write};
use std::fs::File;

// --- Helper: Input Number ---
fn get_input(prompt: &str) -> f64 {
    loop {
        print!("{}", prompt);
        io::stdout().flush().unwrap();
        let mut buffer = String::new();
        match io::stdin().read_line(&mut buffer) {
            Ok(_) => match buffer.trim().parse::<f64>() {
                Ok(num) => return num,
                Err(_) => println!("Invalid number."),
            },
            Err(_) => println!("Error."),
        }
    }
}

// --- Helper: Input Integer ---
fn get_input_usize(prompt: &str) -> usize {
    loop {
        print!("{}", prompt);
        io::stdout().flush().unwrap();
        let mut buffer = String::new();
        match io::stdin().read_line(&mut buffer) {
            Ok(_) => match buffer.trim().parse::<usize>() {
                Ok(num) => return num,
                Err(_) => println!("Invalid number."),
            },
            Err(_) => println!("Error."),
        }
    }
}

// --- Helper: Input Yes/No ---
fn get_bool(prompt: &str) -> bool {
    loop {
        print!("{} (y/n): ", prompt);
        io::stdout().flush().unwrap();
        let mut buffer = String::new();
        io::stdin().read_line(&mut buffer).unwrap();
        match buffer.trim().to_lowercase().as_str() {
            "y" | "yes" => return true,
            "n" | "no" => return false,
            _ => println!("Please type 'y' or 'n'."),
        }
    }
}

// --- Helper: Optional Input ---
fn get_optional_input(prompt: &str) -> Option<u64> {
    print!("{}", prompt);
    io::stdout().flush().unwrap();
    let mut buffer = String::new();
    io::stdin().read_line(&mut buffer).unwrap();
    let trimmed = buffer.trim();
    if trimmed.is_empty() {
        None
    } else {
        trimmed.parse::<u64>().ok()
    }
}

// --- Sample Size Calculator (Binary) ---
fn calculate_n_binary(p_ctrl: f64, p_trt: f64, power: f64) -> usize {
    let z_alpha: f64 = 1.96;
    let z_beta: f64 = if power > 0.85 { 1.28 } else { 0.84 }; 
    
    let p_bar = (p_ctrl + p_trt) / 2.0;
    let delta = (p_ctrl - p_trt).abs();

    let term1 = 4.0 * (z_alpha + z_beta).powi(2);
    let term2 = p_bar * (1.0 - p_bar);
    let term3 = delta.powi(2);

    ((term1 * term2) / term3).ceil() as usize
}

// --- Monte Carlo: Required Effect Size for Recovery ---
fn required_effect_for_success<R: Rng + ?Sized>(
    rng: &mut R,
    current_wealth: f64,
    n_remaining: usize,
    p_ctrl: f64,
    burn_in: usize,
    ramp: usize,
    mc_sims: usize,
) -> f64 {
    if n_remaining == 0 {
        return 1.0;
    }
    
    let mut low = 0.001;
    let mut high = 0.50;
    
    for _ in 0..6 {
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
                
                if wealth >= 20.0 { 
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

struct FutilityInfo {
    patient_number: usize,
    wealth_at_trigger: f64,
    required_arr: f64,
    ratio_to_design: f64,
}

struct TrialResult {
    stopped_at: Option<usize>,
    success: bool,
    risk_diff_at_stop: Option<f64>,
    final_risk_diff: f64,
    futility_info: Option<FutilityInfo>,
}

struct ERTProcess {
    wealth: f64,
    burn_in: usize,
    ramp: usize,
    n_trt: f64, 
    events_trt: f64,
    n_ctrl: f64, 
    events_ctrl: f64,
}

impl ERTProcess {
    fn new(burn_in: usize, ramp: usize) -> Self {
        ERTProcess { 
            wealth: 1.0, 
            burn_in, 
            ramp, 
            n_trt: 0.0, 
            events_trt: 0.0, 
            n_ctrl: 0.0, 
            events_ctrl: 0.0 
        }
    }

    fn update(&mut self, i: usize, outcome: f64, is_trt: bool) {
        let rate_trt = if self.n_trt > 0.0 { self.events_trt / self.n_trt } else { 0.5 };
        let rate_ctrl = if self.n_ctrl > 0.0 { self.events_ctrl / self.n_ctrl } else { 0.5 };
        let delta_hat = rate_trt - rate_ctrl;

        if is_trt { 
            self.n_trt += 1.0; 
            if outcome == 1.0 { self.events_trt += 1.0; } 
        } else { 
            self.n_ctrl += 1.0; 
            if outcome == 1.0 { self.events_ctrl += 1.0; } 
        }

        if i > self.burn_in {
            let num = ((i - self.burn_in) as f64).max(0.0);
            let c_i = (num / self.ramp as f64).clamp(0.0, 1.0);

            let lambda = if outcome == 1.0 { 
                0.5 + 0.5 * c_i * delta_hat 
            } else { 
                0.5 - 0.5 * c_i * delta_hat 
            };
            
            let lambda = lambda.clamp(0.001, 0.999);
            let multiplier = if is_trt { lambda / 0.5 } else { (1.0 - lambda) / 0.5 };
            self.wealth *= multiplier;
        }
    }

    fn current_risk_diff(&self) -> f64 {
        let r_t = if self.n_trt > 0.0 { self.events_trt / self.n_trt } else { 0.0 };
        let r_c = if self.n_ctrl > 0.0 { self.events_ctrl / self.n_ctrl } else { 0.0 };
        r_t - r_c
    }
}

pub fn run() {
    println!("\n==========================================");
    println!("   BINARY e-RT SIMULATION");
    println!("==========================================\n");
    
    // --- User Inputs ---
    let p_ctrl = get_input("Control Event Rate (e.g. 0.40): ");
    let p_trt  = get_input("Treatment Event Rate (e.g. 0.30): ");
    let design_arr = (p_ctrl - p_trt).abs();

    // Sample size
    let n_patients = if get_bool("Calculate Sample Size automatically?") {
        let mut power = get_input("Target Power (e.g. 0.80): ");
        if power >= 1.0 {
            println!("Power must be < 1.0. Capping at 0.99.");
            power = 0.99;
        }
        let freq_n = calculate_n_binary(p_ctrl, p_trt, power);
        println!("\nFrequentist N (Power {:.0}%): {}", power * 100.0, freq_n);
        
        if get_bool("Add buffer? (10-20% increase may improve e-process power)") {
            let buffer_pct = get_input("Buffer percentage (e.g. 15): ");
            let buffered = (freq_n as f64 * (1.0 + buffer_pct / 100.0)).ceil() as usize;
            println!("Buffered N: {}", buffered);
            buffered
        } else {
            freq_n
        }
    } else {
        get_input_usize("Enter Number of Patients: ")
    };
    
    // Simulation parameters
    let n_sims = get_input_usize("Number of simulations (e.g. 2000): ");
    
    // Success threshold
    println!("\nSuccess threshold (1/alpha). Default = 20 (alpha=0.05)");
    let success_threshold = get_input("Success threshold (e.g. 20): ");
    
    // Futility watch
    println!("\nFutility watch threshold. Default = 0.5");
    let futility_watch = get_input("Futility watch threshold (e.g. 0.5): ");
    
    // Futility analysis
    let run_futility = get_bool("Run futility analysis? (adds computation time)");
    
    // Seed
    let seed = get_optional_input("Seed for reproducibility (press Enter for random): ");
    
    // Fixed parameters (vanilla e-RT)
    let burn_in: usize = 50;
    let ramp: usize = 100;
    
    println!("\n--- Trial Design ---");
    println!("Control Rate:    {:.1}%", p_ctrl * 100.0);
    println!("Treatment Rate:  {:.1}%", p_trt * 100.0);
    println!("Design ARR:      {:.1}%", design_arr * 100.0);
    println!("Total N:         {}", n_patients);
    println!("Simulations:     {}", n_sims);
    println!("Success (1/α):   {}", success_threshold);
    println!("Futility Watch:  {}", futility_watch);
    println!("Burn-In:         {} (default)", burn_in);
    println!("Ramp:            {} (default)", ramp);
    if let Some(s) = seed {
        println!("Seed:            {}", s);
    } else {
        println!("Seed:            random");
    }

    // Initialize RNG
    let mut rng: Box<dyn RngCore> = match seed {
        Some(s) => Box::new(StdRng::seed_from_u64(s)),
        None => Box::new(rand::thread_rng()),
    };

    // === PHASE 1: TYPE I ERROR ===
    print!("\nPhase 1: Type I Error (Null)... ");
    io::stdout().flush().unwrap();
    let mut null_rejections = 0;
    
    for _ in 0..n_sims {
        let mut proc = ERTProcess::new(burn_in, ramp);
        for i in 1..=n_patients {
            let is_trt = rng.gen_bool(0.5);
            let outcome = if rng.gen_bool(p_ctrl) { 1.0 } else { 0.0 };
            proc.update(i, outcome, is_trt);
            if proc.wealth > success_threshold {
                null_rejections += 1;
                break; 
            }
        }
    }
    let type1_error = (null_rejections as f64 / n_sims as f64) * 100.0;
    println!("Done. Type I Error: {:.2}%", type1_error);

    // === PHASE 2: POWER & FUTILITY ANALYSIS ===
    print!("Phase 2: Power Analysis");
    if run_futility { print!(" + Futility"); }
    println!("...");
    io::stdout().flush().unwrap();
    
    let mut results: Vec<TrialResult> = Vec::with_capacity(n_sims);
    let mut trajectories: Vec<Vec<f64>> = vec![vec![0.0; n_patients + 1]; n_sims];
    
    // Store 30 sample trajectories for plotting
    let sample_indices: Vec<usize> = (0..30.min(n_sims)).collect();
    
    let pb_interval = (n_sims / 20).max(1);

    for sim in 0..n_sims {
        if sim % pb_interval == 0 {
            print!(".");
            io::stdout().flush().unwrap();
        }
        
        let mut proc = ERTProcess::new(burn_in, ramp);
        let mut stopped = false;
        let mut stop_step = None;
        let mut stop_diff = None;
        let mut futility_info: Option<FutilityInfo> = None;

        trajectories[sim][0] = 1.0;

        for i in 1..=n_patients {
            let is_trt = rng.gen_bool(0.5);
            let prob = if is_trt { p_trt } else { p_ctrl };
            let outcome = if rng.gen_bool(prob) { 1.0 } else { 0.0 };
            
            proc.update(i, outcome, is_trt);
            trajectories[sim][i] = proc.wealth;

            // Futility info (if enabled)
            if run_futility && futility_info.is_none() && proc.wealth < futility_watch && i > burn_in {
                let n_remaining = n_patients - i;
                let req_arr = required_effect_for_success(
                    &mut *rng,
                    proc.wealth, 
                    n_remaining, 
                    p_ctrl, 
                    burn_in, 
                    ramp, 
                    50
                );
                futility_info = Some(FutilityInfo {
                    patient_number: i,
                    wealth_at_trigger: proc.wealth,
                    required_arr: req_arr,
                    ratio_to_design: req_arr / design_arr,
                });
            }

            // Success
            if !stopped && proc.wealth > success_threshold {
                stopped = true;
                stop_step = Some(i);
                stop_diff = Some(proc.current_risk_diff());
            }
        }

        results.push(TrialResult {
            stopped_at: stop_step,
            success: stopped,
            risk_diff_at_stop: stop_diff,
            final_risk_diff: proc.current_risk_diff(),
            futility_info,
        });
    }
    println!(" Done.");

    // === COMPUTE STATISTICS ===
    let success_count = results.iter().filter(|r| r.success).count();
    let no_stop_count = n_sims - success_count;
    
    let (avg_stop_n, avg_diff_stop, avg_diff_final, type_m_error) = if success_count > 0 {
        let successes: Vec<&TrialResult> = results.iter().filter(|r| r.success).collect();
        let avg_n: f64 = successes.iter()
            .map(|r| r.stopped_at.unwrap() as f64).sum::<f64>() / success_count as f64;
        let avg_stop: f64 = successes.iter()
            .map(|r| r.risk_diff_at_stop.unwrap().abs()).sum::<f64>() / success_count as f64;
        let avg_final: f64 = successes.iter()
            .map(|r| r.final_risk_diff.abs()).sum::<f64>() / success_count as f64;
        (avg_n, avg_stop, avg_final, avg_stop / avg_final)
    } else {
        (0.0, 0.0, 0.0, 0.0)
    };

    // Futility stats (if enabled)
    let futility_stats = if run_futility {
        let trials_with_info: Vec<&TrialResult> = results.iter()
            .filter(|r| r.futility_info.is_some()).collect();
        
        if !trials_with_info.is_empty() {
            let n_triggered = trials_with_info.len();
            
            let mut trigger_patients: Vec<f64> = trials_with_info.iter()
                .map(|r| r.futility_info.as_ref().unwrap().patient_number as f64).collect();
            trigger_patients.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            let mut required_arrs: Vec<f64> = trials_with_info.iter()
                .map(|r| r.futility_info.as_ref().unwrap().required_arr).collect();
            required_arrs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            let mut ratios: Vec<f64> = trials_with_info.iter()
                .map(|r| r.futility_info.as_ref().unwrap().ratio_to_design).collect();
            ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            let triggered_success = trials_with_info.iter().filter(|r| r.success).count();
            
            Some((
                n_triggered,
                trigger_patients[n_triggered / 2],
                required_arrs[n_triggered / 2],
                ratios[n_triggered / 4],
                ratios[n_triggered / 2],
                ratios[(n_triggered * 3) / 4],
                triggered_success,
            ))
        } else {
            None
        }
    } else {
        None
    };

    // === PRINT CONSOLE SUMMARY ===
    println!("\n==========================================");
    println!("   RESULTS");
    println!("==========================================");
    println!("Type I Error:    {:.2}%", type1_error);
    println!("Power:           {:.1}%", (success_count as f64/n_sims as f64)*100.0);
    if success_count > 0 {
        println!("Avg Stop:        {:.0} ({:.0}% of N)", avg_stop_n, (avg_stop_n / n_patients as f64) * 100.0);
        println!("Type M Error:    {:.2}x", type_m_error);
    }

    // === GENERATE HTML REPORT ===
    println!("\nGenerating report...");
    
    // Prepare trajectory data for plots
    let mut x_axis: Vec<usize> = Vec::new();
    let mut y_median: Vec<f64> = Vec::new();
    let mut y_lower: Vec<f64> = Vec::new();
    let mut y_upper: Vec<f64> = Vec::new();

    for i in 0..=n_patients {
        if i % 5 != 0 && i != n_patients { continue; }
        x_axis.push(i);
        let mut step_vals: Vec<f64> = trajectories.iter().map(|v| v[i]).collect();
        step_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        y_lower.push(step_vals[(n_sims as f64 * 0.025) as usize]);
        y_median.push(step_vals[(n_sims as f64 * 0.50) as usize]);
        y_upper.push(step_vals[(n_sims as f64 * 0.975) as usize]);
    }

    // Sample trajectories (30 runs)
    let sample_trajectories: Vec<&Vec<f64>> = sample_indices.iter()
        .map(|&i| &trajectories[i]).collect();

    // Stopping times
    let stop_times: Vec<f64> = results.iter()
        .filter(|r| r.success)
        .map(|r| r.stopped_at.unwrap() as f64).collect();

    // Required ARRs (if futility enabled)
    let required_arrs: Vec<f64> = if run_futility {
        results.iter()
            .filter(|r| r.futility_info.is_some())
            .map(|r| r.futility_info.as_ref().unwrap().required_arr * 100.0)
            .collect()
    } else {
        Vec::new()
    };

    // Build HTML
    let html = build_html_report(
        // Parameters
        p_ctrl, p_trt, design_arr, n_patients, n_sims,
        success_threshold, futility_watch, burn_in, ramp, seed,
        // Results
        type1_error, success_count, no_stop_count,
        avg_stop_n, avg_diff_stop, avg_diff_final, type_m_error,
        futility_stats, run_futility,
        // Plot data
        &x_axis, &y_median, &y_lower, &y_upper,
        &sample_trajectories, &stop_times, &required_arrs,
    );

    let mut file = File::create("binary_report.html").unwrap();
    file.write_all(html.as_bytes()).unwrap();
    
    println!("\n>> Report saved: binary_report.html");
    println!("==========================================");
}

fn build_html_report(
    p_ctrl: f64, p_trt: f64, design_arr: f64, n_patients: usize, n_sims: usize,
    success_threshold: f64, futility_watch: f64, burn_in: usize, ramp: usize, seed: Option<u64>,
    type1_error: f64, success_count: usize, no_stop_count: usize,
    avg_stop_n: f64, avg_diff_stop: f64, avg_diff_final: f64, type_m_error: f64,
    futility_stats: Option<(usize, f64, f64, f64, f64, f64, usize)>,
    run_futility: bool,
    x_axis: &[usize], y_median: &[f64], y_lower: &[f64], y_upper: &[f64],
    sample_trajectories: &[&Vec<f64>], stop_times: &[f64], required_arrs: &[f64],
) -> String {
    let seed_str = match seed {
        Some(s) => s.to_string(),
        None => "random".to_string(),
    };

    let timestamp = chrono_lite();

    // Convert data to JSON strings for Plotly
    let x_json: String = format!("{:?}", x_axis);
    let median_json: String = format!("{:?}", y_median);
    let lower_json: String = format!("{:?}", y_lower);
    let upper_json: String = format!("{:?}", y_upper);
    let stop_times_json: String = format!("{:?}", stop_times);
    let required_arrs_json: String = format!("{:?}", required_arrs);

    // Sample trajectories JSON
    let mut sample_traces = String::new();
    for (idx, traj) in sample_trajectories.iter().enumerate() {
        let downsampled: Vec<f64> = traj.iter().enumerate()
            .filter(|(i, _)| i % 5 == 0 || *i == n_patients)
            .map(|(_, v)| *v).collect();
        let color = if idx == 0 { "rgba(100,100,100,0.4)" } else { "rgba(100,100,100,0.2)" };
        sample_traces.push_str(&format!(
            "{{type:'scatter',mode:'lines',x:{},y:{:?},line:{{color:'{}',width:1}},showlegend:false}},",
            x_json, downsampled, color
        ));
    }

    // Futility section HTML
    let futility_html = if run_futility {
        if let Some((n_trig, med_patient, med_arr, q25, q50, q75, trig_success)) = futility_stats {
            format!(r#"
            <h2>Futility Analysis</h2>
            <table>
                <tr><td>Trials triggering (wealth &lt; {:.1}):</td><td><strong>{} ({:.1}%)</strong></td></tr>
                <tr><td>Median patient at trigger:</td><td>{:.0} ({:.0}% of N)</td></tr>
                <tr><td>Median required ARR:</td><td>{:.1}%</td></tr>
                <tr><td>Design ARR:</td><td>{:.1}%</td></tr>
                <tr><td>Ratio (Required/Design) - 25th pctl:</td><td>{:.2}x</td></tr>
                <tr><td>Ratio (Required/Design) - Median:</td><td>{:.2}x</td></tr>
                <tr><td>Ratio (Required/Design) - 75th pctl:</td><td>{:.2}x</td></tr>
                <tr><td>Triggered trials that succeeded:</td><td>{} ({:.1}%)</td></tr>
            </table>
            "#, 
            futility_watch, n_trig, (n_trig as f64 / n_sims as f64) * 100.0,
            med_patient, (med_patient / n_patients as f64) * 100.0,
            med_arr * 100.0, design_arr * 100.0,
            q25, q50, q75,
            trig_success, (trig_success as f64 / n_trig as f64) * 100.0)
        } else {
            "<h2>Futility Analysis</h2><p>No trials triggered futility watch.</p>".to_string()
        }
    } else {
        "".to_string()
    };

    // Required ARR plot (only if futility enabled and data exists)
    let required_arr_plot = if run_futility && !required_arrs.is_empty() {
        format!(r#"
        <h3>Required ARR Distribution (at Futility Trigger)</h3>
        <div id="plot4" style="width:100%;height:400px;"></div>
        <script>
            Plotly.newPlot('plot4', [{{
                type: 'histogram',
                x: {},
                marker: {{color: 'steelblue'}},
                name: 'Required ARR'
            }}], {{
                shapes: [{{
                    type: 'line',
                    x0: {:.1}, x1: {:.1},
                    y0: 0, y1: 1,
                    yref: 'paper',
                    line: {{color: 'red', width: 2, dash: 'dash'}}
                }}],
                xaxis: {{title: 'Required ARR (%)'}},
                yaxis: {{title: 'Count'}},
                annotations: [{{
                    x: {:.1}, y: 1, yref: 'paper',
                    text: 'Design ARR', showarrow: false,
                    font: {{color: 'red'}}
                }}]
            }});
        </script>
        "#, required_arrs_json, design_arr * 100.0, design_arr * 100.0, design_arr * 100.0)
    } else {
        "".to_string()
    };

    format!(r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Binary e-RT Simulation Report</title>
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
        <h1>Binary e-RT Simulation Report</h1>
        <p class="timestamp">Generated: {}</p>
        
        <h2>Parameters</h2>
        <table>
            <tr><td>Control Event Rate:</td><td>{:.1}%</td></tr>
            <tr><td>Treatment Event Rate:</td><td>{:.1}%</td></tr>
            <tr><td>Design ARR:</td><td><strong>{:.1}%</strong></td></tr>
            <tr><td>Sample Size (N):</td><td>{}</td></tr>
            <tr><td>Simulations:</td><td>{}</td></tr>
            <tr><td>Success Threshold (1/α):</td><td>{}</td></tr>
            <tr><td>Futility Watch:</td><td>{}</td></tr>
            <tr><td>Burn-in:</td><td>{}</td></tr>
            <tr><td>Ramp:</td><td>{}</td></tr>
            <tr><td>Seed:</td><td>{}</td></tr>
        </table>

        <h2>Operating Characteristics</h2>
        <table>
            <tr class="highlight"><td>Type I Error:</td><td>{:.2}%</td></tr>
            <tr class="highlight"><td>Power (Success Rate):</td><td>{:.1}%</td></tr>
            <tr><td>No Stop:</td><td>{} ({:.1}%)</td></tr>
        </table>

        <h2>Success Analysis</h2>
        <table>
            <tr><td>Number of Successes:</td><td>{}</td></tr>
            <tr><td>Average Stopping Point:</td><td>{:.0} patients ({:.0}% of N)</td></tr>
            <tr><td>Effect at Stop:</td><td>{:.1}% ARR</td></tr>
            <tr><td>Effect at End:</td><td>{:.1}% ARR</td></tr>
            <tr><td>Type M Error (Magnification):</td><td>{:.2}x</td></tr>
        </table>

        {}

        <h2>Visualizations</h2>
        
        <h3>e-Value Trajectories (Median with 95% CI)</h3>
        <div id="plot1" style="width:100%;height:500px;"></div>
        
        <h3>Sample Trajectories (30 runs)</h3>
        <div id="plot2" style="width:100%;height:500px;"></div>
        
        <h3>Stopping Times Distribution</h3>
        <div id="plot3" style="width:100%;height:400px;"></div>
        
        {}
    </div>

    <script>
        // Plot 1: Median + 95% CI
        Plotly.newPlot('plot1', [
            {{type:'scatter',mode:'lines',x:{},y:{},line:{{width:0}},showlegend:false}},
            {{type:'scatter',mode:'lines',x:{},y:{},fill:'tonexty',fillcolor:'rgba(31,119,180,0.3)',line:{{width:0}},showlegend:false}},
            {{type:'scatter',mode:'lines',x:{},y:{},line:{{color:'blue',width:2.5}},name:'Median'}}
        ], {{
            yaxis: {{type:'log',title:'e-value'}},
            xaxis: {{title:'Patients Enrolled'}},
            shapes: [
                {{type:'line',x0:0,x1:1,xref:'paper',y0:{},y1:{},line:{{color:'green',width:2,dash:'dash'}}}},
                {{type:'line',x0:0,x1:1,xref:'paper',y0:{},y1:{},line:{{color:'orange',width:1.5,dash:'dot'}}}}
            ]
        }});

        // Plot 2: Sample trajectories
        Plotly.newPlot('plot2', [
            {}
            {{type:'scatter',mode:'lines',x:[0,{}],y:[{},{}],line:{{color:'green',width:2,dash:'dash'}},name:'Success'}},
            {{type:'scatter',mode:'lines',x:[0,{}],y:[{},{}],line:{{color:'orange',width:1.5,dash:'dot'}},name:'Futility Watch'}}
        ], {{
            yaxis: {{type:'log',title:'e-value'}},
            xaxis: {{title:'Patients Enrolled'}}
        }});

        // Plot 3: Stopping times
        Plotly.newPlot('plot3', [{{
            type: 'histogram',
            x: {},
            marker: {{color: 'green'}},
            name: 'Success'
        }}], {{
            xaxis: {{title: 'Patient Number at Stop'}},
            yaxis: {{title: 'Count'}}
        }});
    </script>
</body>
</html>"#,
        // Header
        timestamp,
        // Parameters
        p_ctrl * 100.0, p_trt * 100.0, design_arr * 100.0, n_patients, n_sims,
        success_threshold, futility_watch, burn_in, ramp, seed_str,
        // Operating characteristics
        type1_error, (success_count as f64 / n_sims as f64) * 100.0,
        no_stop_count, (no_stop_count as f64 / n_sims as f64) * 100.0,
        // Success analysis
        success_count, avg_stop_n, (avg_stop_n / n_patients as f64) * 100.0,
        avg_diff_stop * 100.0, avg_diff_final * 100.0, type_m_error,
        // Futility section
        futility_html,
        // Required ARR plot
        required_arr_plot,
        // Plot 1 data
        x_json, lower_json, x_json, upper_json, x_json, median_json,
        success_threshold, success_threshold, futility_watch, futility_watch,
        // Plot 2 data
        sample_traces,
        n_patients, success_threshold, success_threshold,
        n_patients, futility_watch, futility_watch,
        // Plot 3 data
        stop_times_json
    )
}

// Simple timestamp without external crate
fn chrono_lite() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let secs = duration.as_secs();
    // Very rough conversion - good enough for display
    let days = secs / 86400;
    let years = 1970 + days / 365;
    let remaining_days = days % 365;
    let months = remaining_days / 30 + 1;
    let day = remaining_days % 30 + 1;
    let hours = (secs % 86400) / 3600;
    let mins = (secs % 3600) / 60;
    format!("{}-{:02}-{:02} {:02}:{:02} UTC", years, months, day, hours, mins)
}

use rand::RngCore;