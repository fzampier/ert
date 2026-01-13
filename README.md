# ert

Sequential Randomization Tests using e-values (betting martingales) for clinical trial monitoring.

## Foundation

This implements the betting framework for hypothesis testing developed by Ramdas, Shafer, and collaborators. The core idea comes from Duan, Ramdas & Wasserman (2022): test treatment effects by wagering on treatment assignments given observed outcomes.

**The betting intuition:** After observing a patient's outcome but before learning their treatment assignment, we place a bet on which arm they belong to. If treatment works, outcomes should predict assignments—patients with good outcomes are more likely from the treatment arm. Under the null hypothesis (no effect), outcomes are uninformative about assignment, so no betting strategy can systematically profit. Wealth fluctuates randomly around 1.

When wealth exceeds a threshold (e.g., 20 for alpha=0.05), we reject the null. Ville's inequality guarantees Type I error control at any stopping time, regardless of when or why monitoring stops.

## Paper

> Zampieri FG. Sequential Randomization Tests Using e-values: Applications for trial monitoring. arXiv:2512.04366 [stat.ME]. 2026.

**[Read the preprint](https://arxiv.org/abs/2512.04366)**

The preprint has R code from a previous implementation. As of January 07 2026, the Rust implementation has some advantages not implemented in the R code.
R code remains valid.

## Methods

| Method | Endpoint | Use Case |
|--------|----------|----------|
| e-RT | Binary | Response rates, mortality |
| e-RTo | Continuous (bounded) | Ventilator-free days, pain scores |
| e-RTc | Continuous (unbounded) | Biomarkers, lab values |
| e-RTs | Time-to-event | Overall survival, PFS |
| e-RTms | Multi-state | ICU trajectories (Dead/ICU/Ward/Home) |

e-RTms uses stratified averaging across transition types to handle non-absorbing states where patients can bounce between levels.

## Futility Monitoring

The FutilityMonitor is a simulation-based decision support tool (not an e-process) that estimates P(recovery)—the probability of eventually crossing the success threshold at design effect.

**Survivorship bias correction:** Late stop recommendations were overconfident because trials reaching late checkpoints with low P(recovery) have "survived" earlier checkpoints. The monitor applies a calibrated correction: `survivorship_correction = 1.0 + 6.0 * t²` where t = enrollment fraction.

**Calibration (with correction):**
- Early recommendations (<50% of N): ~6% est vs ~8% actual
- Late recommendations (≥50% of N): ~4% est vs ~12% actual
- When stop recommended, actual recovery is always well below 50%

## Quick Start

```bash
cargo build --release
cargo run --release
```

See [MANUAL.md](MANUAL.md) for mathematical details and usage.

## Key References

- Ramdas A, Wang R. Hypothesis testing with e-values. Found Trends Stat 2025.
- Duan B, Ramdas A, Wasserman L. Interactive rank testing by betting. CLeaR 2022.
- Shafer G. Testing by betting. JRSS-A 2021.
- Grunwald P et al. The safe logrank test. AAAI 2021.

## Disclaimer

Experimental method under development. Not for clinical use without statistical oversight.
In case this isn't obvious, this is a side project for me that is heavily vibe coded with Claude Code while I try to learn Rust myself.

## License

MIT
