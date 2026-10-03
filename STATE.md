# Research State

- Current phase: `None`
- Pipeline completed: `True`

## Previous phases

resource_finder (succeeded), experiment_runner (succeeded)

## Current phase context

- Phase: `experiment_runner`
- Status: `completed`
- Started: `2026-10-03T05:21:37.690594Z`
- Next steps:
  - Validate the report and experimental artifacts before finalizing.

## Workspace check

- Root: `/workspaces/detect-different-lies-ebe1-claude-opus-5-5`
- Directory usable: `True`

## Output validation

- Valid: `True`
- Expected: `REPORT.md`
- Missing: None
- Outside workspace: None

## Agent notes

<!-- NEURICO_AGENT_NOTES_START -->
### resource_finder
<!-- NEURICO_AGENT_NOTES_START:resource_finder -->
**Phase: resource_finder — COMPLETE (2026-10-03).**

Done:
- 38 papers in papers/ (all 9 user arXiv IDs). The Anthropic blog is summarized in papers/notes/.
- 12 deep-read notes in papers/notes/. Synthesis in literature_review.md; catalog in resources.md.
- Datasets in datasets/: MASK (6 configs), TriviaQA rc.nocontext, sycophancy-eval, Noddybear/lies subset, probe_training.
  Docs in datasets/README.md; re-fetch with download.sh.
- 12 repos in code/ (Apollo, MASK, Liars' Bench, SEP, LLMsKnow, WACK, TTPD, GoT, RepE, Pacchiardi, ...). Docs in code/README.md.

Key findings:
- No paper evaluates lie detectors on hallucinations (honest-but-wrong) or hallucination detectors on known-answer lies.
  That is the core gap.
- Lie probes ≈ falsity/"deception-salient context" detectors (2603.10003, 2502.03407, 2602.01425).
- Hallucination detectors track recall/confidence (2510.09033), so they are predicted to call lies "known".

DIRECTION BUDGET (scored table in literature_review.md §7). KEPT:
- D1: matched TriviaQA knowledge × incentive factorial design, giving shares of false outputs by mechanism.
- D2: detector battery (Apollo RepE probe, truth probe, DYL; SE/SEP/logprob/P(True)/correctness probe).
  Measures AUROC lie-vs-hallucination, cross-firing at 1% FPR, direction geometry.
- D3: MASK cross-check on the same model.

PRUNED: SAE detectors, LoRA-fine-tuned detectors, model organisms, multi-model scaling, non-lying deception, CoT judges.
HK+ snowballing is folded into D1 as a control.

Environment / gotchas:
- .venv has torch 2.13 cu130, vLLM 0.30, transformers 5.18, sklearn. 1× A6000 48GB.
- No C compiler on host. Before vLLM, run: `export CC=$PWD/.venv/bin/zigcc VLLM_USE_FLASHINFER_SAMPLER=0`.
- Kill orphan VLLM::EngineCore by pid after runs.
- Llama-3.1-8B-Instruct is cached. Smoke tests pass (scripts/smoke_test_*.py).
  TriviaQA val n=200: 103 known / 55 unknown / 42 mixed.
- OPENAI_API_KEY and OPENROUTER_KEY are set, for the LLM judge.

Next phase (experiment_runner):
1. Knowledge-label ~3–5k TriviaQA val questions (greedy + 10 samples). Use a better alias match plus an LLM-judge fallback.
2. Generate neutral, incentive (sycophancy-eval incorrect_answer, system-prompt incentives) and control
   (snowballing, pressured-unknown) responses.
3. Label hallucination / lie / honest / evasion and compute shares with CIs.
4. Extract activations and fit detectors on disjoint data, then evaluate on the categories.
5. MASK subset cross-check.

Unresolved:
- Paper-finder service was down (HTTP 500), so search was manual via the arXiv API.
- Liars' Bench HF is gated; the repo data is available instead.
- Some HF IDs in agent notes are marked "unverified".
- TriviaQA alias incompleteness means an LLM judge is needed for labels.
<!-- NEURICO_AGENT_NOTES_END:resource_finder -->

### experiment_runner
<!-- NEURICO_AGENT_NOTES_START:experiment_runner -->
**Phase: experiment_runner — experiments complete, write-up in progress (2026-10-03).**
- Pipeline: src/gen_knowledge.py -> stage2..stage9 (see README). Subject Llama-3.1-8B-Instruct; LOCAL judge
  Qwen2.5-14B-Instruct (OpenRouter key hit its daily limit (403); OPENAI_API_KEY invalid (401)).
- 3000 TriviaQA val Qs: 1679 known / 548 unknown / 773 partial (greedy + 10 samples). 2975 have a valid distractor.
- Lie yield on known Qs: sycophancy 8.2%, incentive (no "lie" word) 78%, instructed 90%.
- Detectors at layer 12 (results/analysis/tables.md, detectors.json). Same-prompt lie vs pressured-unknown-false is
  near chance for all off-the-shelf detectors. RepE probe reads the prompt: AUROC 0.99 from prompt tokens for lie vs
  halluc. Composite falsity+P(IK) reaches 0.68.
- MASK cross-check: results/analysis/mask_tables.md (small n: 67 lies, 55 false-no-belief).
- Gotcha: `pkill -f`/`pgrep -f` inside a bash command kills that shell; use scripts/kill_vllm.sh.
- Next: finish layer sweep (logs/stage6_sweep.log), rerun stage7, write REPORT.md/README.md.
<!-- NEURICO_AGENT_NOTES_END:experiment_runner -->

<!-- NEURICO_AGENT_NOTES_END -->
