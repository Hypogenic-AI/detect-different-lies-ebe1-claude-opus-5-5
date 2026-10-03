## MASK cross-check (known_facts + disinformation, Llama-3.1-8B-Instruct, layer 12)

Counts: lie (pressure): 67, honest (pressure): 23, false, no belief (pressure): 55, true, no belief (pressure): 34, belief correct, consistent (neutral): 140, belief wrong, consistent (neutral): 6, belief wrong, inconsistent (neutral): 19

Pressure-statement categories: {np.str_('honest'): 23, np.str_('evasion'): 150, np.str_('false_no_belief'): 55, np.str_('true_no_belief'): 34, np.str_('lie'): 67, np.str_('honest_wrong_belief'): 3, np.str_('lie_against_wrong_belief'): 2}

| detector | lie vs honest (pressure) | lie vs false-no-belief (pressure) | lie vs belief-wrong-inconsistent (neutral) | belief wrong-inconsistent vs correct-consistent (neutral) | belief wrong-consistent vs correct-consistent (neutral) |
|---|---|---|---|---|---|
| repe_probe | 0.71 [0.56,0.86] | 0.42 [0.32,0.52] | 0.99 [0.97,1.00] | 0.62 [0.47,0.75] | 0.51 [0.25,0.78] |
| truth_probe | 0.56 [0.41,0.68] | 0.62 [0.53,0.72] | 0.52 [0.37,0.68] | 0.55 [0.41,0.69] | 0.51 [0.20,0.86] |
| truth_probe(mean) | 0.75 [0.61,0.87] | 0.69 [0.59,0.79] | 0.33 [0.18,0.48] | 0.62 [0.51,0.73] | 0.59 [0.32,0.82] |
| instr_lie_probe | 0.79 [0.64,0.90] | 0.53 [0.43,0.64] | 0.99 [0.97,1.00] | 0.49 [0.35,0.62] | 0.59 [0.24,0.94] |
| correctness_probe | 0.59 [0.46,0.74] | 0.54 [0.43,0.64] | 0.57 [0.43,0.71] | 0.73 [0.60,0.84] | 0.58 [0.31,0.84] |
| pik_probe | 0.28 [0.15,0.42] | 0.34 [0.24,0.44] | 0.02 [0.00,0.05] | 0.71 [0.57,0.85] | 0.52 [0.19,0.85] |
| neg_logprob | 0.39 [0.24,0.55] | 0.34 [0.24,0.45] | 0.06 [0.00,0.18] | 0.69 [0.58,0.83] | 0.53 [0.19,0.82] |