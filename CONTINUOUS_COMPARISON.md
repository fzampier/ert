# Continuous Methods Comparison: LinearERT vs MAD vs Agnostic

## Overview

This document compares three sequential testing approaches for continuous endpoints:

- **LinearERT**: Uses bounded outcome normalization, magnitude-weighted betting
- **MAD-based**: Uses robust statistics (median, MAD) for standardized residuals
- **Agnostic (Universal)**: Simple good/bad threshold at midpoint of bounds (LinearERT) or control mean (MAD)

## Comparison Table

| Scenario | Parameters | t-test | LinearERT | Agnostic (Lin) | MAD | Agnostic (MAD) |
|----------|-----------|--------|-----------|----------------|-----|----------------|
| 1. Small effect | μ=14→15, SD=8, N=300 | 19.0% | **0.0%** | 7.0% | **11.7%** | 6.0% |
| 2. Medium effect | μ=14→17, SD=8, N=300 | 90.1% | 35.3% | **73.0%** | 67.3% | **76.0%** |
| 3. Large effect | μ=14→20, SD=8, N=300 | 100% | 100% | 99.7% | 100% | 100% |
| 4. Medium, large N | μ=14→17, SD=8, N=500 | 98.7% | **91.3%** | 91.3% | 66.3% | **92.7%** |
| 5. Control low | μ=5→8, SD=8, N=300 | 90.1% | 18.0% | 39.7% | **64.3%** | **76.3%** |
| 6. Control high | μ=22→25, SD=8, N=300 | 90.1% | 9.7% | **0.0%** | **60.0%** | 67.0% |
| 7. Small SD | μ=14→17, SD=5, N=300 | 99.9% | 36.7% | **99.7%** | 99.0% | **99.3%** |
| 8. Large SD | μ=14→17, SD=12, N=300 | 58.1% | **40.7%** | 35.0% | 30.0% | 32.0% |
| 9. Small N | μ=14→17, SD=8, N=100 | 46.6% | 0.0% | 18.0% | **53.0%** | 15.3% |
| 10. Wide bounds | μ=50→55, [0-100], SD=20 | 58.1% | 0.0% | **31.3%** | 25.7% | 29.3% |

*All scenarios use bounds [0, 28] unless otherwise noted. Bold indicates best sequential method.*

## Key Findings

### 1. Agnostic wins when control ≈ midpoint (Scenarios 2, 7)

When the control mean aligns with the threshold (midpoint of bounds), the agnostic approach becomes a powerful sequential sign test. In scenario 2, agnostic achieves 73-76% power vs LinearERT's 35%.

### 2. Agnostic fails when control far from midpoint (Scenario 6)

With control mean at 22 and midpoint at 14, agnostic's threshold is misaligned. Most control outcomes are already "good" (above 14), destroying the signal. Agnostic drops to 0% while MAD maintains 60%.

### 3. LinearERT struggles with wide bounds (Scenarios 5, 6, 10)

The [0, max] normalization compresses signals for outcomes near the center. A value at the midpoint gets scalar ≈ 0, killing the bet regardless of actual effect size.

### 4. MAD is most robust across scenarios

MAD never catastrophically fails. It performs reasonably in all scenarios, making it the safest choice when outcome distribution is uncertain.

### 5. Large N helps LinearERT recover (Scenario 4)

With N=500 instead of 300, LinearERT catches up to agnostic (91.3% vs 91.3%). The conservative betting eventually accumulates enough evidence.

### 6. Large SD hurts agnostic, helps LinearERT (Scenario 8)

When distributions overlap heavily (large SD), the binary good/bad classification loses information. Magnitude-based methods like LinearERT preserve more signal.

## Recommendations

| Situation | Recommended Method |
|-----------|-------------------|
| Control mean ≈ midpoint of bounds | Agnostic or MAD |
| Control mean far from midpoint | MAD |
| Large sample size available | LinearERT or MAD |
| Uncertain about distribution | MAD (most robust) |
| Small effect size expected | MAD |
| Large effect size expected | Any method works |

## Implications for the Power Hierarchy

The original hierarchy assumed:
```
Traditional (ceiling) > Dedicated (middle) > Universal (floor)
```

For continuous endpoints, this doesn't always hold. The "domain knowledge" in LinearERT (bounded normalization) can actually hurt power when:
- Control mean is near the midpoint
- Bounds are wide relative to the effect
- SD is small (clear separation)

The agnostic approach, despite having zero domain knowledge, can outperform dedicated methods by 20-40% in favorable conditions.

## Technical Details

### Agnostic Signal Translation

For LinearERT comparison:
- `good_threshold = (min_val + max_val) / 2.0`
- `good = outcome > good_threshold`

For MAD comparison:
- `good_threshold = mu_ctrl` (control mean)
- `good = outcome > good_threshold`

### Why LinearERT Underperforms

LinearERT normalizes outcomes to [-1, 1]:
```
x = (outcome - min) / (max - min)  # to [0,1]
scalar = 2x - 1                     # to [-1,1]
```

For an outcome at the midpoint (e.g., 14 in [0,28]):
- x = 14/28 = 0.5
- scalar = 2(0.5) - 1 = 0

The scalar is zero, so no bet is placed regardless of the arm assignment. This severely limits power when the effect manifests as a location shift near the center.
