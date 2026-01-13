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
| e-RTms | 4 | Multi-state (ordinal trajectories) | Proportional OR |
| e-RTu | 5 | Universal/agnostic | Rate Difference |
| Analyze Binary | 6 | Real trial CSV (binary) | RD, OR |
| Analyze Continuous | 7 | Real trial CSV (continuous) | Mean Diff, d |
| Analyze Survival | 8 | Real trial CSV (survival) | HR |
| Analyze Multi-State | 9 | Real trial CSV (multi-state) | Proportional OR |
| Compare Methods | 10 | e-RTo vs e-RTc comparison | Both |

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

**Use case:** Trials tracking patients through multiple ordered states (e.g., Dead < ICU < Ward < Home).

**Configuration:**
- **ICU preset:** Dead(0) < ICU(1) < Ward(2) < Home(3), absorbing: Dead, Home
- **Custom:** User-defined N states, ordered worst to best

**State ordering:** States indexed 0 to N-1, where 0 = worst outcome, N-1 = best outcome.

**Transition classification:**

```
Good transition (treatment working):
    Moving to higher-indexed state (to > from)
    Examples: ICU -> Ward, Ward -> Home

Bad transition (treatment failing):
    Moving to lower-indexed state (to < from)
    Examples: Ward -> ICU, ICU -> Dead

Neutral: Staying in same state (no signal)
```

**The bouncing problem:**

When states are not absorbing (e.g., patients can move Ward -> ICU -> Ward), transitions from different states have different signal-to-noise ratios. A naive e-process that pools all transitions loses power because noisy bounce-back transitions dilute the treatment signal.

**Solution: Stratified averaging**

e-RTms runs a separate e-process for each "from state" (stratum), then averages their wealths:

```
For each from_state s = 0, 1, ..., N-1:
    Track stratum-specific counts:
        r_T^s = good transitions from s in treatment / total from s in treatment
        r_C^s = good transitions from s in control / total from s in control
        delta_s = r_T^s - r_C^s

    Stratum betting (for transition from state s):
        lambda_s = 0.5 + 0.5 * c(n) * delta_s * (good ? 1 : -1)
        W_s *= (arm == T ? lambda_s : 1 - lambda_s) / 0.5

Combined e-value (average of active strata):
    E_n = (W_0 + W_1 + ... + W_k) / k
    where k = number of strata with observations
```

**Why averaging works:**

Each stratum's e-process is a martingale with E[W_s] = 1 under H0. By linearity of expectation:

```
E[(W_0 + W_1 + ... + W_k) / k] = (E[W_0] + E[W_1] + ... + E[W_k]) / k = 1
```

This holds regardless of dependence between strata (same patient can contribute to multiple strata). Averaging is robust; multiplication would require independence.

**Performance:**

| Scenario | Naive | Stratified |
|----------|-------|------------|
| Absorbing states (ICU) | 82% | 80% |
| Bouncing states | 7% | 98% |

Stratification recovers power when patients bounce between non-absorbing states.

**Effect sizes (ICU preset):**
- **Large:** OR ~1.6, +15% Home at day 28
- **Small:** OR ~1.2, +5% Home at day 28 (more realistic)

**Parameters:**
- `Configuration`: ICU preset (1) or Custom states (2)
- For custom: State names, absorbing states, starting state, follow-up days
- `Transition matrices`: Control and treatment (rows must sum to 1.0)
- `Patients per trial`: Sample size
- `Simulations`: Monte Carlo runs
- `Threshold`: e-value threshold

**Benchmark metrics:**
- Proportional Odds Ratio (OR > 1 favors treatment)
- Mann-Whitney P(T>C) (probability treatment outcome better than control)

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
- `Design control rate`: Expected control rate from protocol
- `Design treatment rate`: Expected treatment rate from protocol
- Uses FutilityMonitor with default settings (recovery target 10%, stop ratio 1.75x)
- Reports whether stop was recommended and worst observed ratio

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

### 9. Analyze Survival Trial (CSV)

**Menu option:** 8

**CLI:** `ert analyze-survival <file.csv> [options]`

**Use case:** Analyze real clinical trial data with time-to-event outcomes.

**CSV format:**
```
treatment,time,status
1,12.5,1
0,8.3,0
1,24.0,0
...
```
- `treatment`: 0 (control) or 1 (treatment)
- `time`: Time to event or censoring
- `status`: 1 (event) or 0 (censored)

**Uses e-RTs formula** with fixed wager (λ_max = 0.25).

**Parameters:**
- `Burn-in`: Initial events before betting starts (default 30)
- `Ramp`: Gradual increase to full betting (default 50)
- `Success threshold`: e-value for rejection (default 20)

**Output:**
- e-value trajectory (by event number)
- HR at crossing (event ratio)
- Final HR with 95% CI (person-time adjusted)
- Type M error (effect magnification at early stopping)
- HTML report with trajectory plots

**Note:** HR at crossing uses event ratio as a simple proxy. Crude person-time HR can be misleading at early interim analyses due to differential follow-up.

---

### 10. Analyze Multi-State Trial (CSV)

**Menu option:** 9

**CLI:** `ert analyze-multistate <file.csv> [options]`

**Use case:** Analyze real clinical trial data with multi-state/ordinal outcomes.

**CSV format:**
```
patient_id,time,state,treatment
1,0,1,1
1,1,2,1
1,2,3,1
2,0,1,0
2,1,0,0
...
```
- `patient_id`: Patient identifier
- `time`: Day/timepoint
- `state`: State index (0 = worst, N-1 = best)
- `treatment`: 0 (control) or 1 (treatment)

**CLI options:**
- `--states` or `-s`: State names, comma-separated (e.g., "Dead,ICU,Ward,Home")
- `--threshold` or `-t`: e-value threshold (default 20)
- `--burn-in` or `-b`: Initial transitions before betting (default 30)
- `--ramp` or `-r`: Ramp period (default 50)
- `--no-report`: Skip HTML report

**Example:**
```bash
ert analyze-multistate icu_trial.csv --states "Dead,ICU,Ward,Home"
```

**Analysis:**
1. Extracts transitions from patient trajectories
2. Classifies each transition as good (to > from) or bad (to < from)
3. Applies e-RTms betting based on transition direction
4. Reports final state distribution and proportional odds

**Output:**
- Transition counts (good/bad/neutral)
- e-value trajectory
- Final state distribution by arm
- Proportional Odds Ratio
- Mann-Whitney P(T>C)

---

### 11. Compare e-RTo vs e-RTc

**Menu option:** 10

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

**Default values by module:**

| Module | Burn-in | Ramp | Rationale |
|--------|---------|------|-----------|
| e-RT (binary) | 50 | 100 | Binary data is noisy; more patients needed to learn direction |
| e-RTo/c (continuous) | 20 | 50 | Continuous outcomes carry more information per patient |
| e-RTs (survival) | 30 | 50 | Events are sparse; balance between learning and not waiting too long |
| e-RTms (multi-state) | 30 | 50 | Multiple transitions per patient; similar to survival |

The key tradeoff:
- **Higher burn-in:** More robust direction learning, but delays early stopping
- **Lower burn-in:** Faster potential stopping, but risk of betting wrong direction early

### Type M Error

Effect magnification at early stopping:

```
Type_M = |effect at crossing| / |effect at final|
```

- Type M = 1.0: No magnification
- Type M = 1.5: Effect at stopping was 50% larger than final

Early stopping tends to overestimate effects. Type M quantifies this.

### Futility Monitoring

**IMPORTANT:** The FutilityMonitor is a simulation-based decision support tool, NOT a martingale or e-process. It does not provide anytime-valid inference.

The monitor estimates P(recovery), the probability that the trial will eventually cross the success threshold assuming the true treatment effect equals the design effect.

**Key Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| Recovery target | 10% | Recommend stop if P(recovery) < 10% |
| Normal interval | 5% of N | Check every 5% of planned enrollment |
| Alert interval | 1% of N | Switch to frequent checking when in danger zone |
| Danger threshold | 0.5 | Below this e-value triggers alert mode |
| Hysteresis | 3 | Consecutive checks above threshold to exit alert mode |

**How it works:**

1. At regular checkpoints, uses Monte Carlo forward simulation to estimate P(recovery) at design effect
2. Compares P(recovery) against an enrollment-adjusted threshold
3. Reports ratio = base_threshold / P(recovery) for interpretability

**Enrollment-Adjusted Threshold:**

Trials that only cross below threshold late have been hovering near the margin—the crossing is more likely a pessimistic noise fluctuation than signal. To address this, the threshold becomes stricter as enrollment increases:

```
adjusted_threshold = recovery_target × (1 - 0.5 × √t)
```

where t = enrollment fraction (patient / N_total).

| Enrollment (t) | Adjusted Threshold |
|----------------|-------------------|
| 25% | 7.5% |
| 50% | 6.5% |
| 75% | 5.7% |
| 100% | 5.0% |

**Practical interpretation:**

- Early recommendations use lenient threshold (easier to trigger stop)
- Late recommendations require stronger evidence (lower P(recovery))
- When stop is recommended, actual recovery is always well below 50%

Under H0 (no effect):
- 100% of trials recommended for stop
- Type I error among stopped trials ~2%

**Limitation:** Uses bidirectional betting, so futility captures scenarios where the e-value is struggling, not specifically "treatment is harmful." A trial with a harmful treatment would still be recommended for futility stopping (which is appropriate).

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
| `survival_analysis_report.html` | Analyze Survival (CSV) |
| `multistate_analysis_report.html` | Analyze Multi-State (CSV) |
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
