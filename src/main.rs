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
mod multistate_experiment;

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use clap::{Parser, Subcommand};

/// Sequential Randomization Tests using e-values (betting martingales)
#[derive(Parser)]
#[command(name = "ert")]
#[command(author = "Fernando G Zampieri")]
#[command(version)]
#[command(about = "Sequential Randomization Tests using e-values", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Auto-detect and analyze trial data (redirects binary data appropriately)
    #[command(visible_alias = "a")]
    Analyze {
        /// Path to CSV file
        file: PathBuf,
        #[command(flatten)]
        opts: AnalyzeArgs,
    },

    /// Analyze binary trial data (e-RT)
    #[command(name = "analyze-binary", visible_alias = "ab")]
    AnalyzeBinary {
        /// Path to CSV file
        file: PathBuf,
        #[command(flatten)]
        opts: AnalyzeArgs,
    },

    /// Analyze continuous trial data (e-RTc)
    #[command(name = "analyze-continuous", visible_alias = "ac")]
    AnalyzeContinuous {
        /// Path to CSV file
        file: PathBuf,
        #[command(flatten)]
        opts: AnalyzeArgs,
    },

    /// Analyze survival/time-to-event trial data (e-RTs)
    #[command(name = "analyze-survival", visible_alias = "as")]
    AnalyzeSurvival {
        /// Path to CSV file
        file: PathBuf,
        #[command(flatten)]
        opts: AnalyzeArgs,
    },

    /// Analyze multi-state trial data (e-RTms)
    #[command(name = "analyze-multistate", visible_alias = "am")]
    AnalyzeMultistate {
        /// Path to CSV file
        file: PathBuf,
        /// State names, comma-separated, worst to best (e.g., "Dead,ICU,Ward,Home")
        #[arg(short, long, value_delimiter = ',')]
        states: Option<Vec<String>>,
        #[command(flatten)]
        opts: AnalyzeArgs,
    },
}

#[derive(Parser, Clone)]
struct AnalyzeArgs {
    /// Success threshold (default: 20, i.e., alpha=0.05)
    #[arg(short, long)]
    threshold: Option<f64>,

    /// Burn-in period before betting starts
    #[arg(short, long)]
    burn_in: Option<usize>,

    /// Ramp period for gradual betting increase
    #[arg(short, long)]
    ramp: Option<usize>,

    /// Skip HTML report generation
    #[arg(long)]
    no_report: bool,
}

/// Options struct for passing to analysis modules (preserves existing interface)
#[derive(Default)]
pub struct AnalyzeOptions {
    pub threshold: Option<f64>,
    pub burn_in: Option<usize>,
    pub ramp: Option<usize>,
    pub generate_report: bool,
}

impl From<&AnalyzeArgs> for AnalyzeOptions {
    fn from(args: &AnalyzeArgs) -> Self {
        AnalyzeOptions {
            threshold: args.threshold,
            burn_in: args.burn_in,
            ramp: args.ramp,
            generate_report: !args.no_report,
        }
    }
}

/// Detect if CSV data appears to be binary (all outcome values are 0 or 1).
fn detect_binary_data(path: &PathBuf) -> Option<bool> {
    let file = File::open(path).ok()?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // Skip header
    let _ = lines.next()?;

    let mut seen_non_binary = false;
    let mut count = 0;

    for line in lines.take(100) {
        let line = line.ok()?;
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 2 {
            let outcome = parts[1].trim();
            if let Ok(val) = outcome.parse::<f64>() {
                count += 1;
                if val != 0.0 && val != 1.0 {
                    seen_non_binary = true;
                    break;
                }
            }
        }
    }

    if count == 0 {
        return None;
    }
    Some(!seen_non_binary)
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Some(cmd) => run_command(cmd),
        None => run_interactive(),
    }
}

fn run_command(cmd: Commands) {
    match cmd {
        Commands::Analyze { file, opts } => {
            let options: AnalyzeOptions = (&opts).into();
            let path_str = file.to_string_lossy();

            // Auto-detect binary data and redirect
            if let Some(is_binary) = detect_binary_data(&file) {
                if is_binary {
                    eprintln!("Note: Data appears to be binary (all values 0 or 1).");
                    eprintln!("      Redirecting to analyze-binary for appropriate analysis.");
                    eprintln!("      Use 'ert analyze-binary' or 'ert analyze-continuous' to override.\n");
                    if let Err(e) = analyze_binary::run_cli(&path_str, &options) {
                        eprintln!("Error: {}", e);
                    }
                    return;
                }
            }

            if let Err(e) = analyze_continuous::run_cli(&path_str, &options) {
                eprintln!("Error: {}", e);
            }
        }

        Commands::AnalyzeBinary { file, opts } => {
            let options: AnalyzeOptions = (&opts).into();
            if let Err(e) = analyze_binary::run_cli(&file.to_string_lossy(), &options) {
                eprintln!("Error: {}", e);
            }
        }

        Commands::AnalyzeContinuous { file, opts } => {
            let options: AnalyzeOptions = (&opts).into();
            if let Err(e) = analyze_continuous::run_cli(&file.to_string_lossy(), &options) {
                eprintln!("Error: {}", e);
            }
        }

        Commands::AnalyzeSurvival { file, opts } => {
            let options: AnalyzeOptions = (&opts).into();
            if let Err(e) = analyze_survival::run_cli(&file.to_string_lossy(), &options) {
                eprintln!("Error: {}", e);
            }
        }

        Commands::AnalyzeMultistate { file, states, opts } => {
            analyze_multistate::run(
                &file.to_string_lossy(),
                states,
                opts.threshold.unwrap_or(20.0),
                opts.burn_in.unwrap_or(30),
                opts.ramp.unwrap_or(50),
                opts.no_report,
            );
        }
    }
}

fn run_interactive() {
    println!("\n==========================================");
    println!("   e-RT: Sequential Randomization Tests");
    println!("==========================================");
    println!("\nSelect an option:");
    println!("  1. e-RT   (binary endpoint)");
    println!("  2. e-RTc  (continuous endpoint)");
    println!("  3. e-RTu  (universal/agnostic)");
    println!("  4. e-RTs  (survival/time-to-event)");
    println!("  5. e-RTms (multi-state)");
    println!("  6. Analyze Binary Trial (CSV)");
    println!("  7. Analyze Continuous Trial (CSV)");
    println!("  8. Analyze Survival Trial (CSV)");
    println!("  9. Analyze Multi-State Trial (CSV)");
    println!(" 10. [Demo] Why stratification works");
    println!("  0. Exit");

    print!("\nSelect: ");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();

    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();

    match input.trim() {
        "1" => binary::run(),
        "2" => continuous::run(),
        "3" => agnostic::run(),
        "4" => survival::run(),
        "5" => multistate::run(),
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
        "10" => multistate_experiment::run(),
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
