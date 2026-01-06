---
output:
  pdf_document: default
  html_document: default
---
# e-RT Manual

Sequential Randomization Tests using e-values (betting martingales).

## Quick Reference

| Module | Menu | Use Case | Effect Measure |
|--------|------|----------|----------------|
| e-RT | 1 | Binary outcomes (response/no response) | Risk Difference, OR |
| e-RTo | 2 | Bounded continuous (VFD 0-28, scores) | Mean Difference |
| e-RTc | 2 | Unbounded continuous (biomarkers) | Cohen's d |
| e-RTs | 3 | Time-to-event (survival) | Hazard-based |
| e-RTms | 4 | Multi-state (ICU->Ward->Home) | Transition OR |
| e-RTu | 5 | Universal/agnostic | Rate Difference |
| Analyze Binary | 6 | Real trial CSV (binary) | RD, OR |
| Analyze Continuous | 7 | Real trial CSV (continuous) | Mean Diff, d |
| Compare Methods | 8 | e-RTo vs e-RTc comparison | Both |

---

## Power Hierarchy

Three layers of sequential testing, from most to least powerful:

```
+-------------------------------------------------------------+
|  TRADITIONAL (z-test, log-rank, t-test)                     |
|  Fixed sample size. Maximum power. One-shot analysis.       |
|                                                             |
|                     | pay for sequential                    |
|                     v                                       |
|  DEDICATED (e-RT, e-RTo, e-RTs, e-RTms)                     |
|  Sequential. Anytime-valid. Domain-aware betting.           |
|  Uses domain knowledge (rates, means, hazards, transitions).|
|                                                             |
|                     | pay for agnosticism                   |
|                     v                                       |
|  UNIVERSAL (e-RTu)                                          |
|  Sequential. Anytime-valid. Domain-blind.                   |
|  Sees only: (arm, good/bad) signals. The floor.             |
+-------------------------------------------------------------+
```

**What each layer costs:**
- Traditional -> Dedicated: ~10-20% power for sequential flexibility
- Dedicated -> Universal: ~10-30% power for domain blindness

**Example at N=500, 10% effect:**
```
Z-test (fixed):        80%   <- ceiling
e-RT (dedicated):      72%   <- domain-aware sequential
e-RTu (universal):     55%   <- floor
```

**Key insight:** Dedicated methods are essentially:
`Universal core + domain-specific translator + optimized betting`

The extra power comes from:
1. Smarter signal definition (what counts as "good")
2. Domain-aware betting calibration
3. Using magnitude, not just direction

---

## Core Formula: The e-Process

All methods follow the same betting martingale structure:

```
E_n = E_{n-1} * M_n

where M_n = lambda_n / 0.5      if patient n is treatment
          = (1 - lambda_n) / 0.5  if patient n is control
```

- `E_n`: e-value after n patients (starts at E_0 = 1)
- `lambda_n`: betting fraction in [0, 1], represents belief treatment is better
- `lambda = 0.5`: neutral (no information)
- `lambda > 0.5`: betting on treatment being better
- `lambda < 0.5`: betting on control being better

**Rejection rule:** Reject H0 when E_n >= 1/alpha (e.g., E >= 20 for alpha = 0.05)

**Type I error guarantee:** P(E_n >= 1/alpha for any n | H0) <= alpha

---

## Modules and Formulas

### 1. e-RT (Binary Endpoint)

**Menu option:** 1

**Use case:** Trials with binary outcomes (responder/non-responder, success/failure).

**Formula:**

```
Let p_T(n) = observed treatment success rate up to patient n-1
    p_C(n) = observed control success rate up to patient n-1
    delta(n) = p_T(n) - p_C(n)

Ramp factor:
    c(n) = clamp((n - burn_in) / ramp, 0, 1)

For patient n with outcome Y_n in {0, 1}:
    If Y_n = 1 (success):
        lambda_n = 0.5 + 0.5 * c(n) * delta(n)
    If Y_n = 0 (failure):
        lambda_n = 0.5 - 0.5 * c(n) * delta(n)

    lambda_n = clamp(lambda_n, 0.01, 0.99)
```

**Intuition:** Bet that successes come from the arm with historically higher success rate.

**Parameters:**
- `P(success | control)`: Control arm response rate (e.g., 0.25)
- `P(success | treatment)`: Treatment arm response rate (e.g., 0.35)
- `Patients per trial`: Sample size for simulation
- `Simulations`: Number of Monte Carlo runs
- `Threshold`: e-value threshold for rejection (default 20 = alpha=0.05)
- `Seed`: Optional random seed for reproducibility

**Output:**
- Type I error rate (should be <= alpha)
- Power (rejection rate under alternative)
- Type M error (effect magnification at early stopping)
- HTML report with e-value trajectories

**Effect measures:** Risk Difference, Odds Ratio with 95% CI

---

### 2. e-RTo (Continuous Bounded/Ordinal)

**Menu option:** 2, then select option 1

**Use case:** Bounded continuous outcomes like Ventilator-Free Days (0-28), pain scores (0-10).

**Formula:**

```
Let mu_T(n) = observed treatment mean up to patient n-1
    mu_C(n) = observed control mean up to patient n-1
    delta(n) = mu_T(n) - mu_C(n)
    range = max_val - min_val

Ramp factor:
    c(n) = clamp((n - burn_in) / ramp, 0, 1)

Normalized outcome:
    x(n) = (Y_n - min_val) / range        # maps to [0, 1]
    scalar(n) = 2 * x(n) - 1               # maps to [-1, 1]

Normalized effect:
    delta_norm(n) = delta(n) / range

Betting fraction:
    lambda_n = 0.5 + 0.5 * c(n) * delta_norm(n) * scalar(n)
    lambda_n = clamp(lambda_n, 0.001, 0.999)
```

**Intuition:** High outcomes (scalar > 0) suggest treatment is better if delta > 0. Low outcomes suggest control is better if delta > 0.

**Parameters:**
- `Control mean`: Expected mean in control arm
- `Treatment mean`: Expected mean in treatment arm
- `SD`: Standard deviation
- `Min bound`: Minimum possible value (e.g., 0)
- `Max bound`: Maximum possible value (e.g., 28)
- `Patients per trial`: Sample size
- `Simulations`: Monte Carlo runs
- `Threshold`: e-value threshold (default 20)

---

### 3. e-RTc (Continuous Unbounded/MAD)

**Menu option:** 2, then select option 2

**Use case:** Unbounded continuous outcomes like biomarkers, lab values.

**Formula:**

```
Let mu_T(n) = observed treatment mean up to patient n-1
    mu_C(n) = observed control mean up to patient n-1

Direction (learned from data):
    dir(n) = +1 if mu_T(n) > mu_C(n)
           = -1 if mu_T(n) < mu_C(n)
           =  0 otherwise

Robust scaling using past outcomes Y_1, ..., Y_{n-1}:
    med(n) = median(Y_1, ..., Y_{n-1})
    MAD(n) = median(|Y_i - med(n)|) * 1.4826
    s(n) = MAD(n) if MAD(n) > 0, else 1

Standardized residual:
    r(n) = (Y_n - med(n)) / s(n)

Squashed signal (bounded):
    g(n) = r(n) / (1 + |r(n)|)           # maps to (-1, 1)

Ramp factor:
    c(n) = clamp((n - burn_in) / ramp, 0, 1)

Betting fraction:
    lambda_n = 0.5 + c(n) * c_max * g(n) * dir(n)
    lambda_n = clamp(lambda_n, 0.001, 0.999)
```

**Intuition:** Uses MAD for robust scaling. Extreme outcomes (large |r|) are squashed to prevent catastrophic bets.

**Parameters:**
- `Control mean`: Expected mean in control arm
- `Treatment mean`: Expected mean in treatment arm
- `SD`: Standard deviation
- `c_max`: Maximum betting fraction (default 0.6)
- `Patients per trial`: Sample size
- `Simulations`: Monte Carlo runs

---

### 4. e-RTs (Survival/Time-to-Event)

**Menu option:** 3

**Use case:** Trials with time-to-event outcomes (overall survival, progression-free survival).

**Formula:**

```
Score test signal based on observed vs expected events.

At each event time t_k, for patient i with event:
    Let n_T(t_k) = patients at risk in treatment at time t_k
        n_C(t_k) = patients at risk in control at time t_k
        n(t_k) = n_T(t_k) + n_C(t_k)

    Expected fraction from treatment:
        e_T(t_k) = n_T(t_k) / n(t_k)

    Signal:
        S_k = +1 if event in control (good for treatment hypothesis)
            = -1 if event in treatment (bad for treatment hypothesis)

    Observed vs expected:
        O_k - E_k = (1 - arm_k) - e_T(t_k)   # arm=0 for control, 1 for treatment

Cumulative signal drives betting:
    Z(n) = sum of (O_k - E_k) up to patient n

    lambda_n = 0.5 + c(n) * clamp(Z(n) * scale, -0.49, 0.49)
```

**Intuition:** If more events occur in control than expected under H0, evidence favors treatment.

**Parameters:**
- `Median survival (control)`: e.g., 12 months
- `Hazard ratio`: Treatment effect (e.g., 0.7 = 30% reduction)
- `Enrollment period`: Duration of patient accrual
- `Follow-up`: Additional follow-up after enrollment ends
- `Patients`: Total sample size
- `Simulations`: Monte Carlo runs

**Note:** e-RTu (universal) was removed from survival as it doesn't work - survival information is encoded in event timing, not binary good/bad signals.

---

### 5. e-RTms (Multi-State)

**Menu option:** 4

**Use case:** Trials tracking patients through multiple states (e.g., ICU -> Ward -> Home/Dead).

**States:** Ward (0), ICU (1), Home (2, absorbing), Dead (3, absorbing)

**Formula:**

```
Each day, patient transitions according to transition matrix P[from][to].

Favorable transitions (treatment working):
    ICU -> Ward (step-down)
    Ward -> Home (discharge)
    ICU -> Home (discharge from ICU)

Unfavorable transitions:
    Ward -> ICU (deterioration)
    Ward -> Dead
    ICU -> Dead

Signal for patient on day d:
    If favorable transition:   signal = +1
    If unfavorable transition: signal = -1
    If no change or absorbing: no update

Betting uses cumulative transition evidence:
    lambda_n based on observed transition rate difference between arms
```

**Effect sizes:**
- **Large:** OR ~1.6, +15% Home at day 28
- **Small:** OR ~1.2, +5% Home at day 28 (more realistic)

**Parameters:**
- `Effect size`: Large (1) or Small (2)
- `Days to simulate`: Follow-up duration (e.g., 28)
- `Patients per trial`: Sample size
- `Simulations`: Monte Carlo runs
- `Threshold`: e-value threshold

---

### 6. e-RTu (Universal/Agnostic)

**Menu option:** 5

**Use case:** Domain-agnostic e-process. Only sees (arm, good/bad) signals.

**Formula:**

```
Input: sequence of (arm, good) pairs where good in {true, false}

Let r_T(n) = good_T / n_T = observed "good" rate in treatment
    r_C(n) = good_C / n_C = observed "good" rate in control
    delta(n) = r_T(n) - r_C(n)

Ramp factor:
    c(n) = clamp((n - burn_in) / ramp, 0, 1)

For signal n with outcome good_n:
    If good_n = true:
        lambda_n = 0.5 + 0.5 * c(n) * delta(n)
    If good_n = false:
        lambda_n = 0.5 - 0.5 * c(n) * delta(n)

    lambda_n = clamp(lambda_n, 0.01, 0.99)
```

**Intuition:** Identical to e-RT but accepts any source that can map outcomes to good/bad.

**Works well for:**
- Binary outcomes (equivalent to e-RT)
- Multi-state transitions (good = favorable transition)

**Does NOT work for:**
- Survival (info is in timing, not binary signal)

---

### 7. Analyze Binary Trial (CSV)

**Menu option:** 6

**Use case:** Analyze real clinical trial data with binary outcomes.

**CSV format:**
```
treatment,outcome
1,1
0,0
1,0
...
```
- `treatment`: 0 (control) or 1 (treatment)
- `outcome`: 0 (failure) or 1 (success)

**Uses e-RT formula** applied to real data sequentially.

**Parameters:**
- `Burn-in`: Initial patients before betting starts (default 50)
- `Ramp`: Gradual increase to full betting (default 100)
- `Success threshold`: e-value for rejection (default 20)
- `Futility monitoring`: Optional, tracks recovery probability

**Futility analysis (if enabled):**
- `Futility threshold`: e-value below which to flag (e.g., 0.5)
- `Design control rate`: Expected control rate from protocol
- `Design treatment rate`: Expected treatment rate from protocol
- Uses Monte Carlo to find required ARR for 50% recovery probability

---

### 8. Analyze Continuous Trial (CSV)

**Menu option:** 7

**Use case:** Analyze real clinical trial data with continuous outcomes.

**CSV format:**
```
treatment,outcome
1,18.5
0,12.3
1,22.1
...
```

**Methods:**
- **e-RTo:** Uses bounded formula (requires min/max)
- **e-RTc:** Uses MAD formula (unbounded)

**Futility analysis:**
- `Futility threshold`: e.g., 0.5 or 0.9
- Uses Monte Carlo to find required effect for 50% recovery

---

### 9. Compare e-RTo vs e-RTc

**Menu option:** 8

Runs both methods on identical simulated data to compare Type I error and power.

---

## Key Concepts

### e-Values and Thresholds

An e-value is a measure of evidence against the null hypothesis:
- e = 1: No evidence
- e = 20: Strong evidence (alpha = 0.05)
- e = 100: Very strong evidence (alpha = 0.01)

**Threshold = 1/alpha:** For alpha = 0.05, use threshold = 20.

### Burn-in and Ramp

- **Burn-in:** Initial patients where no betting occurs (learning period)
- **Ramp:** Gradual increase from 0 to full betting strength

```
c(n) = 0                          if n <= burn_in
     = (n - burn_in) / ramp       if burn_in < n < burn_in + ramp
     = 1                          if n >= burn_in + ramp
```

Purpose: Prevents catastrophic early losses from noise before effect direction is learned.

Typical values: burn_in = 50, ramp = 100

### Type M Error

Effect magnification at early stopping:

```
Type_M = |effect at crossing| / |effect at final|
```

- Type M = 1.0: No magnification
- Type M = 1.5: Effect at stopping was 50% larger than final

Early stopping tends to overestimate effects. Type M quantifies this.

### Futility Monitoring

Tracks whether trial can still succeed:
- Monitors when e-value drops below threshold
- Uses Monte Carlo to find required effect for 50% recovery probability
- Reports ratio to design effect

**Interpretation:**
- Ratio < 1.0: Can recover with smaller-than-design effect
- Ratio > 1.5: Need larger-than-design effect (concerning)
- Ratio > 2.0: Strong signal to consider stopping

**Limitation:** Current implementation uses bidirectional betting, so futility mainly captures early noise rather than "treatment isn't working." For true futility detection, one-sided betting would be needed.

---

## HTML Reports

All modules generate HTML reports with:
- Console output summary
- e-value trajectory plots (log scale)
- Support plots (linear ln(e) scale)
- Method-specific results

Reports use:
- Plotly 2.35.0 for interactive plots
- Standardized styling (system-ui font, 1400px width)
- Threshold and futility lines on plots

---

## Files Generated

| File | Source |
|------|--------|
| `binary_report.html` | e-RT simulation |
| `continuous_report.html` | e-RTo/c simulation |
| `survival_report.html` | e-RTs simulation |
| `multistate_report.html` | e-RTms simulation |
| `agnostic_report.html` | e-RTu simulation |
| `binary_analysis_report.html` | Analyze Binary (CSV) |
| `continuous_analysis_report.html` | Analyze Continuous (CSV) |
| `comparison_report.html` | Compare Methods |

---

## Building and Running

```bash
# Build
cargo build --release

# Run
cargo run --release

# Or run the binary directly
./target/release/ert
```

---

## Limitations

1. **Bidirectional betting:** e-process detects effects in either direction. A harmful treatment will still cross threshold (just with negative effect).

2. **Futility with bidirectional:** Futility monitoring is weaker than ideal because the process learns direction and bets accordingly.

3. **e-RTu and survival:** Universal e-process doesn't work for survival outcomes - information is in event timing, not binary signals.

4. **No covariate adjustment:** Current implementation uses unadjusted analyses only.

5. **1:1 randomization assumed:** All modules assume equal allocation to arms.
