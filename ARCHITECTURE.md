# e-RT Architecture: The Power Hierarchy

## Three Layers of Sequential Testing

```
┌─────────────────────────────────────────────────────────────┐
│  TRADITIONAL (z-test, proportional odds, log-rank)          │
│  Fixed sample. Maximum power. Domain-specific.              │
│  One-shot analysis at predetermined N.                      │
│  ─────────────────────────────────────────────────────────  │
│                          ↓ pay for sequential               │
│  DEDICATED (e-RT, e-RTms, LinearERT, e-survival)            │
│  Sequential. Anytime-valid. Domain-aware betting.           │
│  Knows about binary outcomes, transitions, hazards, etc.    │
│  Uses domain knowledge to optimize the betting strategy.    │
│  ─────────────────────────────────────────────────────────  │
│                          ↓ pay for agnosticism              │
│  UNIVERSAL (agnostic e-RT)                                  │
│  Sequential. Anytime-valid. Domain-blind.                   │
│  Sees only: (arm, good/bad) signals.                        │
│  Doesn't know what generated them. The floor.               │
└─────────────────────────────────────────────────────────────┘
```

## What Each Layer Costs

- **Traditional → Dedicated**: Power cost for sequential flexibility
- **Dedicated → Universal**: Power cost for domain blindness

## The Philosophy

### Agnostic e-RT is the universal floor
- Zero domain knowledge
- Just bets on (arm, good/bad) signals
- Works for anything: binary, continuous, multistate, survival, kpop streams, Embraer jets
- Lowest power, maximum generality

### Dedicated methods justify themselves by beating the floor
- e-RT binary knows about event rates
- e-RTms knows about good/bad transitions
- LinearERT knows about continuous distributions
- e-survival knows about hazard ratios

### Traditional tests are the ceiling
- Maximum power for fixed sample
- No sequential flexibility
- Domain-specific assumptions

## Implementation Plan

### Current State
- Traditional benchmarks: Added to most modules (z-test, proportional odds, etc.)
- Dedicated e-RT: Fully implemented (binary, continuous, multistate, survival)
- Agnostic e-RT: Implemented as standalone module (option 5)

### Future Enhancement
Add agnostic e-RT as a **comparator within dedicated modules**, not the other way around.

Each dedicated module (binary.rs, continuous.rs, multistate.rs, survival.rs) would show:

```
--- Power at N=500 ---
Z-test (fixed):        80%   ← ceiling (traditional)
e-RT binary:           72%   ← domain-aware sequential (dedicated)
Agnostic (universal):  55%   ← floor (universal)

Domain knowledge buys you: +17%
Sequential costs you: -8%
```

This gives users the full spectrum:
1. What fixed-sample buys you (ceiling)
2. What domain knowledge buys you (dedicated vs universal)
3. What you get knowing nothing (floor)

## The Agnostic Core

```rust
// The universal signal - all it knows
pub struct Signal {
    pub arm: Arm,        // Treatment or Control
    pub good: bool,      // Good or bad outcome
}

// Translators convert domain → signals
// Binary: event happened → good/bad
// Continuous: above median → good
// Multistate: favorable transition → good
// Survival: longer survival → good
```

The agnostic e-process is a pure betting machine. It doesn't interpret.
Domain knowledge lives in the translator, not the e-process.

## Key Insight

e-RTms (multistate) is essentially agnostic e-RT with a specific translator:
- Translator: "ICU→Ward or Ward→Home = good, else = bad"
- Betting engine: identical to agnostic

This means the dedicated methods are really:
**Agnostic core + domain-specific translator + optimized betting**

The extra power comes from:
1. Smarter signal definition (what counts as "good")
2. Domain-aware betting calibration
3. Using magnitude, not just direction (for continuous)

## Summary

- Agnostic stays pure: no benchmarks added TO it
- Agnostic becomes benchmark: added as comparator WITHIN dedicated modules
- User sees full picture: ceiling (traditional) → dedicated → floor (universal)
