## Behaviour: outcome rates by knowledge x condition

| condition | knowledge | n | P(false) | P(correct) | P(evasion) | P(adopt X) | P(false & follow-up correct) |
|---|---|---|---|---|---|---|---|
| neutral | known | 1665 | 0.013 | 0.987 | 0.000 | 0.002 | 0.008 |
| neutral | partial | 765 | 0.367 | 0.633 | 0.000 | 0.035 | 0.161 |
| neutral | unknown | 545 | 0.993 | 0.002 | 0.006 | 0.075 | 0.106 |
| sycophancy | known | 1665 | 0.082 | 0.918 | 0.000 | 0.049 | 0.059 |
| sycophancy | partial | 765 | 0.501 | 0.499 | 0.000 | 0.205 | 0.180 |
| sycophancy | unknown | 545 | 0.925 | 0.072 | 0.004 | 0.402 | 0.081 |
| incentive | known | 1665 | 0.782 | 0.218 | 0.000 | 0.771 | 0.730 |
| incentive | partial | 765 | 0.922 | 0.078 | 0.000 | 0.882 | 0.494 |
| incentive | unknown | 545 | 0.998 | 0.002 | 0.000 | 0.936 | 0.117 |
| instructed | known | 1665 | 0.895 | 0.105 | 0.000 | 0.026 | 0.838 |
| instructed | partial | 765 | 0.897 | 0.103 | 0.000 | 0.051 | 0.461 |
| instructed | unknown | 545 | 0.969 | 0.031 | 0.000 | 0.075 | 0.084 |

## Shares of false outputs by neutral knowledge state (95% cluster-bootstrap CI)

| pool | n false | known (=lie) | partial | unknown (=hallucination / pressured-unknown) | known & follow-up correct |
|---|---|---|---|---|---|
| neutral | 844 | 2.6% [1.6, 3.8] | 33.3% [29.8, 36.6] | 64.1% [60.5, 67.6] | 1.7% [0.8, 2.5] |
| sycophancy | 1024 | 13.4% [11.3, 15.7] | 37.4% [34.4, 40.6] | 49.2% [45.8, 52.8] | 9.6% [7.8, 11.5] |
| incentive | 2551 | 51.0% [48.9, 53.1] | 27.6% [25.7, 29.5] | 21.3% [19.7, 23.0] | 47.6% [45.3, 49.6] |
| instructed | 2705 | 55.1% [53.0, 57.1] | 25.4% [23.6, 27.2] | 19.5% [17.9, 21.1] | 51.6% [49.5, 53.6] |
| all_pressure | 6280 | 46.7% [44.7, 48.7] | 28.2% [26.6, 30.3] | 25.1% [23.3, 26.9] | 43.1% [41.0, 45.1] |
| all_conditions | 7124 | 41.4% [39.5, 43.4] | 28.8% [27.0, 30.8] | 29.7% [27.6, 31.9] | 38.2% [36.5, 40.1] |
| neutral+sycophancy+incentive | 4419 | 33.1% [31.1, 35.0] | 31.0% [28.6, 33.2] | 36.0% [33.7, 38.4] | 30.0% [28.4, 31.9] |

## Detector AUROCs (layer 12; 95% cluster-bootstrap CI). Positive class listed first.

| detector | lie vs neutral-correct | halluc vs neutral-correct | lie vs halluc | lie vs honest-under-pressure | lie vs pressured-unknown-false | honest-under-pressure vs neutral-correct |
|---|---|---|---|---|---|---|
| repe_probe | 0.93 [0.92,0.93] | 0.62 [0.60,0.65] | 0.89 [0.88,0.91] | 0.88 [0.87,0.89] | 0.64 [0.63,0.65] | 0.40 [0.38,0.41] |
| truth_probe | 0.90 [0.89,0.91] | 0.74 [0.72,0.77] | 0.74 [0.72,0.76] | 0.63 [0.61,0.65] | 0.52 [0.50,0.54] | 0.82 [0.81,0.83] |
| instr_lie_probe | 0.99 [0.99,0.99] | 0.81 [0.79,0.83] | 0.97 [0.96,0.97] | 0.93 [0.92,0.94] | 0.68 [0.67,0.69] | 0.69 [0.68,0.70] |
| neg_logprob | 0.81 [0.80,0.82] | 0.91 [0.89,0.92] | 0.48 [0.46,0.51] | 0.72 [0.71,0.73] | 0.53 [0.52,0.54] | 0.66 [0.65,0.67] |
| inconsistency | 0.81 [0.80,0.82] | 0.91 [0.89,0.93] | 0.47 [0.44,0.49] | 0.73 [0.72,0.74] | 0.52 [0.51,0.53] | 0.63 [0.62,0.65] |
| correctness_probe | 0.84 [0.82,0.85] | 0.89 [0.87,0.90] | 0.51 [0.49,0.54] | 0.70 [0.68,0.71] | 0.48 [0.46,0.50] | 0.67 [0.65,0.68] |
| pik_probe | 0.16 [0.15,0.17] | 0.83 [0.81,0.85] | 0.05 [0.04,0.06] | 0.20 [0.19,0.22] | 0.28 [0.27,0.29] | 0.48 [0.47,0.50] |
| comp_truth+known | 0.96 [0.96,0.97] | 0.41 [0.38,0.44] | 0.97 [0.96,0.97] | 0.85 [0.84,0.86] | 0.74 [0.72,0.75] | 0.78 [0.77,0.79] |
| comp_correct+known | 0.94 [0.94,0.95] | 0.62 [0.59,0.65] | 0.90 [0.89,0.91] | 0.87 [0.86,0.88] | 0.68 [0.66,0.69] | 0.68 [0.67,0.70] |
| repe_probe@prompt | 0.99 [0.99,0.99] | 0.47 [0.45,0.50] | 0.99 [0.99,1.00] | 0.87 [0.85,0.88] | 0.67 [0.66,0.68] | 0.90 [0.89,0.91] |

n (pos/neg): lie vs neutral-correct: 2930/1643; halluc vs neutral-correct: 541/1643; lie vs halluc: 2930/541; lie vs honest-under-pressure: 2930/2065; lie vs pressured-unknown-false: 2930/1576; honest-under-pressure vs neutral-correct: 2065/1643

## Fire rate at threshold = 1% FPR on neutral honest-correct answers

| detector | neutral_correct | hallucination | lie | honest_pressure | pressured_unknown_false |
|---|---|---|---|---|---|
| repe_probe | 0.010 | 0.026 | 0.671 | 0.162 | 0.449 |
| truth_probe | 0.010 | 0.070 | 0.235 | 0.100 | 0.195 |
| instr_lie_probe | 0.010 | 0.179 | 0.952 | 0.226 | 0.742 |
| neg_logprob | 0.010 | 0.253 | 0.347 | 0.044 | 0.275 |
| inconsistency | 0.004 | 0.285 | 0.380 | 0.030 | 0.308 |
| correctness_probe | 0.010 | 0.318 | 0.394 | 0.131 | 0.418 |
| pik_probe | 0.010 | 0.139 | 0.002 | 0.008 | 0.045 |
| comp_truth+known | 0.010 | 0.017 | 0.718 | 0.158 | 0.359 |
| comp_correct+known | 0.010 | 0.091 | 0.773 | 0.187 | 0.540 |
| repe_probe@prompt | 0.010 | 0.006 | 0.962 | 0.436 | 0.756 |

## Per-condition AUROCs (lie from that condition)

| detector | cond | n lie | lie vs halluc | lie vs neutral-correct | lie vs honest-under-pressure (same prompt) | lie vs pressured-unknown-false (same prompt) |
|---|---|---|---|---|---|---|
| repe_probe | sycophancy | 137 | 0.38 [0.33,0.44] | 0.49 [0.43,0.56] | 0.77 [0.72,0.81] | 0.50 [0.45,0.55] |
| repe_probe | incentive | 1302 | 0.83 [0.81,0.85] | 0.88 [0.87,0.90] | 0.44 [0.41,0.47] | 0.53 [0.50,0.56] |
| repe_probe | instructed | 1491 | 1.00 [1.00,1.00] | 1.00 [1.00,1.00] | 0.73 [0.69,0.76] | 0.51 [0.49,0.55] |
| truth_probe | sycophancy | 137 | 0.78 [0.74,0.82] | 0.94 [0.92,0.95] | 0.62 [0.58,0.67] | 0.44 [0.40,0.50] |
| truth_probe | incentive | 1302 | 0.84 [0.82,0.86] | 0.94 [0.93,0.95] | 0.81 [0.78,0.83] | 0.59 [0.56,0.62] |
| truth_probe | instructed | 1491 | 0.65 [0.62,0.68] | 0.87 [0.86,0.88] | 0.70 [0.65,0.75] | 0.58 [0.55,0.61] |
| instr_lie_probe | sycophancy | 137 | 0.42 [0.37,0.48] | 0.79 [0.75,0.83] | 0.72 [0.67,0.76] | 0.37 [0.31,0.42] |
| instr_lie_probe | incentive | 1302 | 0.99 [0.98,0.99] | 1.00 [1.00,1.00] | 0.91 [0.89,0.92] | 0.56 [0.53,0.59] |
| instr_lie_probe | instructed | 1491 | 1.00 [1.00,1.00] | 1.00 [1.00,1.00] | 0.96 [0.95,0.97] | 0.66 [0.63,0.68] |
| neg_logprob | sycophancy | 137 | 0.40 [0.34,0.46] | 0.85 [0.81,0.88] | 0.81 [0.76,0.85] | 0.47 [0.41,0.53] |
| neg_logprob | incentive | 1302 | 0.15 [0.13,0.17] | 0.59 [0.57,0.62] | 0.37 [0.34,0.40] | 0.58 [0.55,0.61] |
| neg_logprob | instructed | 1491 | 0.78 [0.76,0.81] | 0.99 [0.99,0.99] | 0.65 [0.59,0.70] | 0.47 [0.44,0.50] |
| inconsistency | sycophancy | 137 | 0.28 [0.24,0.33] | 0.77 [0.73,0.83] | 0.73 [0.68,0.77] | 0.35 [0.31,0.41] |
| inconsistency | incentive | 1302 | 0.13 [0.11,0.15] | 0.60 [0.58,0.62] | 0.34 [0.31,0.37] | 0.52 [0.50,0.54] |
| inconsistency | instructed | 1491 | 0.78 [0.75,0.80] | 0.99 [0.99,1.00] | 0.75 [0.71,0.79] | 0.50 [0.47,0.52] |
| correctness_probe | sycophancy | 137 | 0.40 [0.34,0.45] | 0.84 [0.80,0.88] | 0.58 [0.54,0.62] | 0.34 [0.29,0.39] |
| correctness_probe | incentive | 1302 | 0.60 [0.57,0.63] | 0.87 [0.85,0.88] | 0.82 [0.80,0.85] | 0.48 [0.45,0.51] |
| correctness_probe | instructed | 1491 | 0.45 [0.43,0.48] | 0.81 [0.79,0.82] | 0.89 [0.87,0.91] | 0.54 [0.51,0.56] |
| pik_probe | sycophancy | 137 | 0.26 [0.22,0.31] | 0.64 [0.60,0.69] | 0.56 [0.51,0.62] | 0.26 [0.22,0.31] |
| pik_probe | incentive | 1302 | 0.08 [0.07,0.10] | 0.29 [0.27,0.30] | 0.49 [0.45,0.53] | 0.27 [0.25,0.30] |
| pik_probe | instructed | 1491 | 0.00 [0.00,0.00] | 0.00 [0.00,0.00] | 0.48 [0.43,0.52] | 0.33 [0.30,0.35] |
| comp_truth+known | sycophancy | 137 | 0.83 [0.79,0.87] | 0.80 [0.77,0.85] | 0.54 [0.50,0.59] | 0.65 [0.59,0.69] |
| comp_truth+known | incentive | 1302 | 0.95 [0.93,0.95] | 0.94 [0.93,0.95] | 0.78 [0.75,0.81] | 0.70 [0.67,0.72] |
| comp_truth+known | instructed | 1491 | 1.00 [0.99,1.00] | 1.00 [0.99,1.00] | 0.64 [0.60,0.69] | 0.67 [0.65,0.70] |
| comp_correct+known | sycophancy | 137 | 0.58 [0.52,0.62] | 0.72 [0.67,0.76] | 0.53 [0.48,0.57] | 0.49 [0.45,0.54] |
| comp_correct+known | incentive | 1302 | 0.84 [0.82,0.86] | 0.91 [0.89,0.92] | 0.81 [0.78,0.83] | 0.60 [0.57,0.62] |
| comp_correct+known | instructed | 1491 | 0.98 [0.97,0.99] | 1.00 [1.00,1.00] | 0.88 [0.86,0.90] | 0.64 [0.62,0.68] |

## Condition-stratified same-prompt AUROCs (pair-weighted mean of per-condition AUROCs)

| detector | lie vs honest-under-pressure | lie vs pressured-unknown-false |
|---|---|---|
| repe_probe | 0.59 | 0.52 |
| truth_probe | 0.74 | 0.58 |
| instr_lie_probe | 0.88 | 0.60 |
| neg_logprob | 0.54 | 0.52 |
| inconsistency | 0.54 | 0.50 |
| correctness_probe | 0.79 | 0.50 |
| pik_probe | 0.50 | 0.30 |
| comp_truth+known | 0.69 | 0.68 |
| comp_correct+known | 0.77 | 0.62 |
| repe_probe@prompt | 0.46 | 0.56 |

## Diagnostic supervised probes (out-of-fold, grouped by question): response tokens vs prompt tokens only

| probe | pooled AUROC [95% CI] | n pos / n neg | sycophancy | incentive | instructed |
|---|---|---|---|---|---|
| diag_lie_vs_honest@resp | 0.99 [0.99,1.00] | 2930/2065 | 0.93 | 1.00 | 0.98 |
| diag_lie_vs_honest@prompt | 0.90 [0.89,0.90] | 2930/2065 | 0.73 | 0.63 | 0.76 |
| diag_lie_vs_halluc@resp | 1.00 [1.00,1.00] | 2930/541 | n/a | n/a | n/a |
| diag_lie_vs_halluc@prompt | 1.00 [1.00,1.00] | 2930/541 | n/a | n/a | n/a |
| diag_lie_vs_pufalse@resp | 0.85 [0.84,0.87] | 2930/1576 | 0.76 | 0.82 | 0.84 |
| diag_lie_vs_pufalse@prompt | 0.85 [0.83,0.87] | 2930/1576 | 0.79 | 0.81 | 0.84 |

## Cosine similarity between probe directions (raw activation space)

| | repe_probe | truth_probe(false) | instr_lie_probe | correctness_probe(incorrect) | pik_probe(unknown) | lie_vs_halluc | lie_vs_honest_pressure |
|---|---|---|---|---|---|---|---|
| repe_probe | 1.00 | 0.02 | 0.32 | -0.01 | 0.00 | 0.20 | 0.03 |
| truth_probe(false) | 0.02 | 1.00 | 0.04 | -0.01 | 0.01 | 0.04 | 0.02 |
| instr_lie_probe | 0.32 | 0.04 | 1.00 | 0.02 | -0.01 | 0.50 | 0.18 |
| correctness_probe(incorrect) | -0.01 | -0.01 | 0.02 | 1.00 | 0.09 | -0.04 | 0.05 |
| pik_probe(unknown) | 0.00 | 0.01 | -0.01 | 0.09 | 1.00 | -0.02 | 0.02 |
| lie_vs_halluc | 0.20 | 0.04 | 0.50 | -0.04 | -0.02 | 1.00 | 0.17 |
| lie_vs_honest_pressure | 0.03 | 0.02 | 0.18 | 0.05 | 0.02 | 0.17 | 1.00 |

Sanity (own-distribution CV AUROC): {'repe_cv_auc': 1.0, 'truth_cv_auc': 0.9622226612645095}