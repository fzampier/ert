mod ert_core;
mod binary;
mod continuous;
mod survival;
mod multistate;
mod agnostic;
mod analyze_binary;
mod analyze_continuous;
mod compare_methods;

fn main() {
    println!("\n==========================================");
    println!("   e-RT: Sequential Randomization Tests");
    println!("==========================================");
    println!("\nSelect an option:");
    println!("  1. Binary Endpoint Simulation");
    println!("  2. Continuous Endpoint Simulation");
    println!("  3. Survival Endpoint Simulation");
    println!("  4. Multi-State Simulation");
    println!("  5. Agnostic e-RT (Universal)");
    println!("  6. Analyze Binary Trial (CSV)");
    println!("  7. Analyze Continuous Trial (CSV)");
    println!("  8. Compare Methods (LinearERT vs MAD)");
    println!("  9. Exit");

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
        "8" => compare_methods::run(),
        "9" => println!("Goodbye!"),
        _ => println!("Invalid option"),
    }
}
