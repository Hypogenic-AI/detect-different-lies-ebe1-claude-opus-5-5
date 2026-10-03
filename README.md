# Detect different lies: hallucination vs incentive-driven misreporting

This project uses matched questions and one model (Llama-3.1-8B-Instruct) in a **knowledge × incentive** design:

- **Knowledge.** 3,000 TriviaQA questions are labelled known, partial or unknown by neutral sampling.
- **Incentive.** Each question is asked under four conditions: neutral, sycophantic hint, operator incentive and
  explicit instruction.

We measure what share of false outputs are hallucinations vs lies. We then test whether standard white-box lie
detectors and hallucination detectors tell them apart, with same-prompt and prompt-only controls. Full write-up:
**[REPORT.md](REPORT.md)**.

## Key findings

- **Shares of false answers depend on context.**
  - Neutral: 64% of false answers come from unknown questions and only 2.6% from known ones (lies).
  - Sycophantic hint: 13% lies.
  - Operator incentive or instruction: 51–55% lies.
  - The incentive prompt never says "lie", yet it makes the 8B model misreport 78% of the answers it knows. 73% are
    verified by the model giving the correct answer when asked in the same conversation.
- **The RepE/Apollo-style deception probe mostly reads the pressure context.**
  - It gives AUROC 0.93 for lie vs honest-correct, but **0.99 from the prompt tokens alone**.
  - It flags 44% of honest answers given under pressure (prompt-token readout at a 1%-FPR threshold).
- **Same prompt, both answers false: every off-the-shelf detector is near chance at telling a knowing lie from an
  ignorant falsehood.**
  - Condition-stratified AUROC 0.50–0.60. RepE 0.52, truth probe 0.58, log-prob 0.52, correctness probe 0.50.
  - MASK cross-check agrees: RepE 0.42.
- **Lie detectors fire only modestly on true hallucinations** (2.6–18% at 1% FPR). **Hallucination detectors miss
  confident incentive-driven lies** (log-prob AUROC lie vs hallucination 0.15).
- **A falsity + knowledge composite** (truth probe + P(I-know) probe) is the only detector above chance in every
  condition (0.65–0.70). The separating signal is already present before the answer is generated.

## Reproduce

```bash
source .venv/bin/activate                                  # torch 2.13, vLLM 0.30, transformers 5.18, sklearn 1.9
export CC=$PWD/.venv/bin/zigcc VLLM_USE_FLASHINFER_SAMPLER=0   # no C compiler on host
python src/gen_knowledge.py              # Llama: greedy + 10 samples on 3,000 TriviaQA val questions
python src/stage2_judge_knowledge.py     # Qwen2.5-14B judge: distractors and knowledge labels
python src/stage3_gen_pressure.py        # Llama: 4 conditions, samples and follow-up
python src/stage4_judge_pressure.py      # judge pressure answers
python src/stage5_activations.py         # HF residual activations (RepE, truth, eval)
python src/stage6_detectors.py           # detectors at layer 12 (sklearn)
python src/stage6_detectors.py --sweep-only --torch   # layer sweep (GPU LR)
python src/stage7_analysis.py            # tables + figures
python src/stage8_mask.py gen && python src/stage8_mask.py judge && python src/stage8_mask.py acts
python src/stage9_mask_analysis.py
scripts/kill_vllm.sh                     # run between vLLM stages: orphaned engines keep holding GPU memory
```

Hardware: 1× RTX A6000 (48 GB). Total runtime is about 2.5 h. No API calls were used: the API judge was unavailable,
so the judge is a local Qwen2.5-14B-Instruct.

## Layout

- `src/`: pipeline. `common.py` holds the prompts and helpers; `judge.py` holds the judge and distractor prompts.
- `results/`: raw generations and labels (`*.jsonl`), detector scores (`detector_scores_*.json`), and the analysis
  (`analysis/tables.md`, `behaviour.json`, `detectors.json`, `mask_tables.md`, `layer_sweep.json`). Activations live
  in `results/acts/` (large, not for git).
- `figures/`: fig1 prevalence, fig2 score distributions, fig3 AUROC heatmap, fig4 cross-firing, fig5 layer sweep,
  fig6 per-condition same-prompt AUROCs.
- `planning.md`, `literature_review.md`, `resources.md`: plan and background.
- `papers/`, `datasets/`, `code/`: pre-gathered resources.
