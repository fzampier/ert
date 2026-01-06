# ert

Sequential Randomization Tests using e-values (betting martingales).

## What is this?

`ert` implements anytime-valid inference for randomized clinical trials. Unlike traditional p-values that require fixed sample sizes, e-values allow continuous monitoring with strict Type I error control. It is based on the framework defined by Ramdas and collaborators.

## Features

| Module | Endpoint Type | Use Case |
|--------|---------------|----------|
| e-RT | Binary | Response rates, success/failure |
| e-RTo | Continuous (bounded) | VFD, pain scores, bounded scales |
| e-RTc | Continuous (unbounded) | Biomarkers, lab values |
| e-RTs | Survival | Time-to-event, OS, PFS |
| e-RTms | Multi-state | ICU->Ward->Home trajectories |
| e-RTu | Universal | Any outcome with good/bad signal |

Plus CSV analyzers for real trial data with optional futility monitoring.

## Quick Start

```bash
# Build
cargo build --release

# Run
cargo run --release
```

Select a module from the menu and follow the prompts.

## Documentation

See [MANUAL.md](MANUAL.md) for:
- Full mathematical formulas for each e-process
- Parameter explanations
- Futility monitoring details
- Limitations and caveats

## Example

```
==========================================
   e-RT: Sequential Randomization Tests
==========================================

Select an option:
  1. e-RT   (binary endpoint)
  2. e-RTo/c (continuous endpoint)
  3. e-RTs  (survival/time-to-event)
  4. e-RTms (multi-state)
  5. e-RTu  (universal/agnostic)
  6. Analyze Binary Trial (CSV)
  7. Analyze Continuous Trial (CSV)
  8. Compare e-RTo vs e-RTc
  9. Exit
```

## CSV Format

For analyzing real trial data:

```csv
treatment,outcome
1,1
0,0
1,0
0,1
```

- `treatment`: 0 (control) or 1 (treatment)
- `outcome`: 0/1 for binary, numeric for continuous

## License

MIT - see [LICENSE](LICENSE)

## Author

Fernando Gonzalez Zapata
