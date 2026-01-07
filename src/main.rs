mod ert_core;
mod binary;
mod continuous;
mod survival;
mod multistate;
mod agnostic;
mod analyze_binary;
mod analyze_continuous;
mod analyze_survival;
mod analyze_multistate;
mod compare_methods;

use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    // If arguments provided, use CLI mode
    if args.len() > 1 {
        run_cli(&args[1..]);
        return;
    }

    // Otherwise, interactive menu
    run_interactive();
}

fn run_cli(args: &[String]) {
    if args.is_empty() {
        print_usage();
        return;
    }

    match args[0].as_str() {
        "analyze" | "a" => {
            if args.len() < 2 {
                eprintln!("Error: CSV file required");
                eprintln!("Usage: ert analyze <file.csv> [options]");
                return;
            }
            let csv_path = &args[1];
            let opts = parse_analyze_options(&args[2..]);

            if let Err(e) = analyze_continuous::run_cli(csv_path, &opts) {
                eprintln!("Error: {}", e);
            }
        }
        "analyze-binary" | "ab" => {
            if args.len() < 2 {
                eprintln!("Error: CSV file required");
                eprintln!("Usage: ert analyze-binary <file.csv> [options]");
                return;
            }
            let csv_path = &args[1];
            let opts = parse_analyze_options(&args[2..]);

            if let Err(e) = analyze_binary::run_cli(csv_path, &opts) {
                eprintln!("Error: {}", e);
            }
        }
        "analyze-survival" | "as" => {
            if args.len() < 2 {
                eprintln!("Error: CSV file required");
                eprintln!("Usage: ert analyze-survival <file.csv> [options]");
                return;
            }
            let csv_path = &args[1];
            let opts = parse_analyze_options(&args[2..]);

            if let Err(e) = analyze_survival::run_cli(csv_path, &opts) {
                eprintln!("Error: {}", e);
            }
        }
        "analyze-multistate" | "am" => {
            if args.len() < 2 {
                eprintln!("Error: CSV file required");
                eprintln!("Usage: ert analyze-multistate <file.csv> [options]");
                return;
            }
            let csv_path = &args[1];
            let opts = parse_multistate_options(&args[2..]);

            analyze_multistate::run(
                csv_path,
                opts.state_names,
                opts.threshold.unwrap_or(20.0),
                opts.burn_in.unwrap_or(30),
                opts.ramp.unwrap_or(50),
                !opts.generate_report,
            );
        }
        "help" | "-h" | "--help" => print_usage(),
        _ => {
            eprintln!("Unknown command: {}", args[0]);
            print_usage();
        }
    }
}

fn parse_analyze_options(args: &[String]) -> AnalyzeOptions {
    let mut opts = AnalyzeOptions::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--method" | "-m" => {
                if i + 1 < args.len() {
                    opts.method = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--threshold" | "-t" => {
                if i + 1 < args.len() {
                    opts.threshold = args[i + 1].parse().ok();
                    i += 1;
                }
            }
            "--burn-in" | "-b" => {
                if i + 1 < args.len() {
                    opts.burn_in = args[i + 1].parse().ok();
                    i += 1;
                }
            }
            "--ramp" | "-r" => {
                if i + 1 < args.len() {
                    opts.ramp = args[i + 1].parse().ok();
                    i += 1;
                }
            }
            "--min" => {
                if i + 1 < args.len() {
                    opts.min_val = args[i + 1].parse().ok();
                    i += 1;
                }
            }
            "--max" => {
                if i + 1 < args.len() {
                    opts.max_val = args[i + 1].parse().ok();
                    i += 1;
                }
            }
            "--no-report" => {
                opts.generate_report = false;
            }
            _ => {}
        }
        i += 1;
    }
    opts
}

#[derive(Default)]
pub struct AnalyzeOptions {
    pub method: Option<String>,      // "rto", "rtc"
    pub threshold: Option<f64>,      // default 20
    pub burn_in: Option<usize>,      // default 50
    pub ramp: Option<usize>,         // default 100
    pub min_val: Option<f64>,        // for e-RTo
    pub max_val: Option<f64>,        // for e-RTo
    pub generate_report: bool,       // default true
}

impl AnalyzeOptions {
    fn default() -> Self {
        AnalyzeOptions {
            method: None,
            threshold: None,
            burn_in: None,
            ramp: None,
            min_val: None,
            max_val: None,
            generate_report: true,
        }
    }
}

#[derive(Default)]
pub struct MultistateOptions {
    pub state_names: Option<Vec<String>>,
    pub threshold: Option<f64>,
    pub burn_in: Option<usize>,
    pub ramp: Option<usize>,
    pub generate_report: bool,
}

fn parse_multistate_options(args: &[String]) -> MultistateOptions {
    let mut opts = MultistateOptions {
        state_names: None,
        threshold: None,
        burn_in: None,
        ramp: None,
        generate_report: true,
    };
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--states" | "-s" => {
                if i + 1 < args.len() {
                    opts.state_names = Some(
                        args[i + 1].split(',')
                            .map(|s| s.trim().to_string())
                            .collect()
                    );
                    i += 1;
                }
            }
            "--threshold" | "-t" => {
                if i + 1 < args.len() {
                    opts.threshold = args[i + 1].parse().ok();
                    i += 1;
                }
            }
            "--burn-in" | "-b" => {
                if i + 1 < args.len() {
                    opts.burn_in = args[i + 1].parse().ok();
                    i += 1;
                }
            }
            "--ramp" | "-r" => {
                if i + 1 < args.len() {
                    opts.ramp = args[i + 1].parse().ok();
                    i += 1;
                }
            }
            "--no-report" => {
                opts.generate_report = false;
            }
            _ => {}
        }
        i += 1;
    }
    opts
}

fn print_usage() {
    println!("e-RT: Sequential Randomization Tests");
    println!();
    println!("USAGE:");
    println!("  ert                                Interactive mode");
    println!("  ert analyze <file.csv>             Analyze continuous trial data");
    println!("  ert analyze-binary <file.csv>      Analyze binary trial data");
    println!("  ert analyze-survival <file.csv>    Analyze survival trial data");
    println!("  ert analyze-multistate <file.csv>  Analyze multi-state trial data");
    println!();
    println!("OPTIONS:");
    println!("  -m, --method <rto|rtc>   Method (default: rtc)");
    println!("  -t, --threshold <N>      Success threshold (default: 20)");
    println!("  -b, --burn-in <N>        Burn-in period (default: 50/30)");
    println!("  -r, --ramp <N>           Ramp period (default: 100/50)");
    println!("  --min <N>                Min bound (e-RTo only)");
    println!("  --max <N>                Max bound (e-RTo only)");
    println!("  -s, --states <names>     State names, comma-separated (multistate)");
    println!("  --no-report              Skip HTML report generation");
    println!();
    println!("CSV FORMAT:");
    println!("  continuous:  treatment,outcome");
    println!("  binary:      treatment,outcome (0/1)");
    println!("  survival:    treatment,time,status (status: 1=event, 0=censored)");
    println!("  multistate:  patient_id,time,state,treatment (state: 0=worst, N-1=best)");
    println!();
    println!("EXAMPLES:");
    println!("  ert analyze trial.csv --method rtc");
    println!("  ert analyze-binary mortality.csv --threshold 20");
    println!("  ert analyze-survival os_data.csv --burn-in 30");
    println!("  ert analyze-multistate icu.csv --states \"Dead,ICU,Ward,Home\"");
}

fn run_interactive() {
    println!("\n==========================================");
    println!("   e-RT: Sequential Randomization Tests");
    println!("==========================================");
    println!("\nSelect an option:");
    println!("  1. e-RT   (binary endpoint)");
    println!("  2. e-RTo/c (continuous endpoint)");
    println!("  3. e-RTs  (survival/time-to-event)");
    println!("  4. e-RTms (multi-state)");
    println!("  5. e-RTu  (universal/agnostic)");
    println!("  6. Analyze Binary Trial (CSV)");
    println!("  7. Analyze Continuous Trial (CSV)");
    println!("  8. Analyze Survival Trial (CSV)");
    println!("  9. Analyze Multi-State Trial (CSV)");
    println!(" 10. Compare e-RTo vs e-RTc");
    println!("  0. Exit");

    print!("\nSelect: ");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();

    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();

    match input.trim() {
        "1" => binary::run(),
        "2" => continuous::run(),
        "3" => survival::run(),
        "4" => multistate::run(),
        "5" => agnostic::run(),
        "6" => {
            if let Err(e) = analyze_binary::run() {
                eprintln!("Error: {}", e);
            }
        }
        "7" => {
            if let Err(e) = analyze_continuous::run() {
                eprintln!("Error: {}", e);
            }
        }
        "8" => {
            if let Err(e) = analyze_survival::run() {
                eprintln!("Error: {}", e);
            }
        }
        "9" => run_analyze_multistate_interactive(),
        "10" => compare_methods::run(),
        "0" => println!("Goodbye!"),
        _ => println!("Invalid option"),
    }
}

fn run_analyze_multistate_interactive() {
    use crate::ert_core::get_string;

    let path = get_string("\nPath to CSV file: ");
    let states_str = get_string("State names (comma-separated, worst to best, or Enter for auto): ");

    let state_names = if states_str.is_empty() {
        None
    } else {
        Some(states_str.split(',').map(|s| s.trim().to_string()).collect())
    };

    analyze_multistate::run(&path, state_names, 20.0, 30, 50, false);
}
