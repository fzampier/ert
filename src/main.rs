mod binary;
mod continuous;
mod survival;
mod multistate;

fn main() {
    println!("\n==========================================");
    println!("   e-RT: Sequential Randomization Tests");
    println!("==========================================");
    println!("\nSelect an option:");
    println!("  1. Binary Endpoint Simulation");
    println!("  2. Continuous Endpoint Simulation");
    println!("  3. Survival Endpoint Simulation");
    println!("  4. Multi-State Simulation");
    println!("  5. Analyze Binary Trial (CSV)");
    println!("  6. Analyze Continuous Trial (CSV)");
    println!("  7. Exit");
    
    print!("\nSelect: ");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
    
    match input.trim() {
        "1" => binary::run(),
        "2" => continuous::run(),
        "3" => survival::run(),
        "4" => multistate::run(),
        "5" => println!("Not yet implemented"),
        "6" => println!("Not yet implemented"),
        "7" => println!("Goodbye!"),
        _ => println!("Invalid option"),
    }
}