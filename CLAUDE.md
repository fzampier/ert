# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ERT is a Rust CLI tool implementing Sequential Randomization Tests using e-values (betting martingales) for clinical trial monitoring. The core idea: test treatment effects by wagering on treatment assignments given observed outcomes. When wealth exceeds a threshold (e.g., 20 for alpha=0.05), reject the null hypothesis with anytime-valid Type I error control.

Based on Zampieri FG. arXiv:2512.04366 [stat.ME]. 2026.

## Build and Run

```bash
cargo build --release
cargo run --release          # Interactive menu
./target/release/ert         # Direct binary

# Run single test
cargo test test_name

# Run all tests
cargo test
```

## CLI Commands

```bash
ert analyze <file.csv>              # Auto-detect binary/continuous
ert analyze-binary <file.csv>       # e-RT for binary outcomes
ert analyze-continuous <file.csv>   # e-RTc for continuous outcomes
ert analyze-survival <file.csv>     # e-RTs for time-to-event
ert analyze-multistate <file.csv>   # e-RTms for multi-state
ert analyze-deaths <file.csv>       # e-RTd for deaths-only monitoring

# Common options: -t threshold, -b burn-in, -r ramp, --no-report
```

## Architecture

**Module structure mirrors the six e-RT methods:**

| Module | Simulation | CSV Analysis | Purpose |
|--------|-----------|--------------|---------|
| `binary.rs` | `run()` | - | e-RT: Binary endpoints |
| `continuous.rs` | `run()` | - | e-RTc: Continuous endpoints |
| `survival.rs` | `run()` | - | e-RTs: Time-to-event |
| `multistate.rs` | `run()` | - | e-RTms: Multi-state ordinal |
| `agnostic.rs` | `run()` | - | e-RTu: Universal/domain-blind |
| `deaths_only.rs` | `run()` | - | e-RTd: Deaths-only monitoring |
| `analyze_binary.rs` | - | `run()`, `run_cli()` | Analyze binary CSV |
| `analyze_continuous.rs` | - | `run()`, `run_cli()` | Analyze continuous CSV |
| `analyze_survival.rs` | - | `run()`, `run_cli()` | Analyze survival CSV |
| `analyze_multistate.rs` | - | `run()` | Analyze multi-state CSV |
| `analyze_deaths.rs` | - | `run()`, `run_cli()` | Analyze deaths-only CSV |

**Core types in `ert_core.rs`:**
- `BinaryERTProcess`: Tracks wealth, counts, betting for binary data. Provides `confidence_sequence_rd()` and `confidence_sequence_or()` for anytime-valid CIs.
- `MADProcess`: MAD-based continuous e-process with robust scaling. Provides `confidence_sequence_d()` and `confidence_sequence_mean_diff()`.
- Helper functions: `median()`, `mad()`, `normal_cdf()`, `normal_quantile()`, sample size calculators.

**Deaths-only type in `deaths_only.rs`:**
- `DeathsOnlyERT`: Bets on P(death from treatment | death occurred). Under null, p=0.5. Provides `confidence_sequence_p()` and `confidence_sequence_rr()` for mortality rate ratio CIs. Trade-off: ~2.5x sample size inflation vs frequentist for equivalent power, but no survivor tracking required.

**Entry point (`main.rs`):**
- CLI parsing with subcommands (analyze, analyze-binary, etc.)
- Interactive menu dispatching to module `run()` functions
- Auto-detection of binary vs continuous data in `analyze` command

## Key Statistical Concepts

**Betting formula (all methods):**
```
E_n = E_{n-1} × M_n
M_n = λ_n / 0.5 (if treatment) or (1 - λ_n) / 0.5 (if control)
```

**Burn-in/ramp:** First `burn_in` patients: no betting. Next `ramp` patients: gradual increase to full betting. Prevents early catastrophic losses.

**Anytime-valid CIs:** Use time-uniform critical value `crit(n,α) = sqrt(2 × (ln(2/α) + ln(ln(n))))` instead of fixed 1.96.

**Multi-state stratification:** `multistate.rs` runs separate e-processes per "from state" then averages wealths - handles bouncing between non-absorbing states.

## CSV Formats

```
# Binary/Continuous: treatment,outcome
1,1
0,0

# Survival: treatment,time,status
1,12.5,1
0,8.3,0

# Multi-state: patient_id,time,state,treatment
1,0,1,1
1,1,2,1

# Deaths-only: arm,time (one row per death, time optional)
1,5
0,12
0,18
```

## HTML Reports

All analysis modules generate interactive HTML reports with Plotly trajectory plots. Output filenames: `*_report.html` (simulations) or `*_analysis_report.html` (CSV analysis).
