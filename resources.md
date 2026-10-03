# Resources Catalog

## Summary

These are the resources gathered to test whether LLM false outputs come from epistemic failure (hallucination) or
incentive-driven misreporting (lie), and whether white-box lie detectors and hallucination detectors can tell the two
apart. Totals:

- **38 papers** (PDF) plus 1 blog post.
- **5 dataset groups**: MASK, TriviaQA, sycophancy-eval, Noddybear/lies subset, and probe-training sets.
- **12 code repositories**.
- An environment that has been smoke-tested on the GPU (Llama-3.1-8B-Instruct, vLLM + HF hidden states).

## Papers

There are 38 downloaded PDFs. The full table is in `papers/README.md`, and deep notes for 12 papers are in `papers/notes/`.

| Title | Year | File | Key info |
|-------|------|------|----------|
| MASK: Disentangling Honesty From Accuracy ★ | 2025 | papers/2503.03750_*.pdf | Lie = statement ≠ belief. Neutral belief elicitation on the same proposition. Llama-3.1-8B P(lie) 23.5% |
| Liars' Bench ★ | 2025 | papers/2511.16035_*.pdf | 72,863 lies across 7 settings. Detectors fail on some lie types. Noise-floor belief verification |
| Detecting Strategic Deception Using Linear Probes ★ | 2025 | papers/2502.03407_*.pdf | Apollo RepE LR probe recipe. 1% FPR calibration on Alpaca |
| One Probe Won't Catch Them All ★ | 2026 | papers/2602.01425_*.pdf | Deception is heterogeneous. Default probe gets AUROC 0.374 on MASK known-facts |
| Probing the Limits of the Lie Detector Approach ★ | 2026 | papers/2603.10003_*.pdf | Truth probes detect falsity, not deception |
| Asymmetries in Spontaneous and Instructed Deception ★ | 2026 | papers/2609.00180_*.pdf | Instructed and spontaneous directions have cos≈0.5. Asymmetric transfer |
| Beyond Liars' Bench ★ | 2026 | papers/2607.20479_*.pdf | Lie typology of the training data drives detectability |
| Do LLMs Really Know What They Don't Know? ★ | 2025 | papers/2510.09033_*.pdf | Hidden states track recall, not truth. Confident hallucinations are undetectable |
| "Did you lie?" (UK AISI) ★ | 2026 | papers/2606.12618_*.pdf | TriviaQA prompted-lying testbed. Apollo, DYL and unrelated-Qs detectors |
| Fine-Tuned Lie Detectors Failed to Generalize (Anthropic blog) ★ | 2026 | papers/notes/anthropic_2026_lie_detectors_blog.md | Parallel interrogation. Cross-category AUROC 0.70–0.75 |
| Distinguishing Ignorance from Error (WACK) | 2024 | papers/2410.22071_*.pdf | Knowledge test (greedy + 5 @T=0.5). HK+ vs HK− |
| LLMs Know More Than They Show | 2024 | papers/2410.02707_*.pdf | Exact-answer-token correctness probes |
| Can LLMs Lie? Beyond Hallucination | 2025 | papers/2509.03518_*.pdf | Truth/hallucination/lie geometry in Llama-3.1-8B |
| …plus 25 more (truth probes, hallucination detectors, deception taxonomies) | | papers/ | see papers/README.md |

## Datasets

There are 5 dataset groups. Details and download instructions are in `datasets/README.md`, and everything can be re-fetched with `bash datasets/download.sh`.

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| MASK ★ | HF cais/MASK | 1,000 items (6 configs) | Pressure vs belief elicitation | datasets/mask/<cfg>/ | Test split only. An LLM judge is needed for open responses |
| TriviaQA ★ | HF mandarjoshi/trivia_qa rc.nocontext | 138k train / 17.9k val | Open QA with aliases | datasets/trivia_qa_rc_nocontext/ | Use val. Alias lists are incomplete |
| sycophancy-eval | HF meg-tong/sycophancy-eval | 1,000 TriviaQA q × 4 templates (+ are_you_sure) | User-pressure incentive | datasets/sycophancy_eval/ | Has `incorrect_answer` targets |
| Noddybear/lies (subset) | HF Noddybear/lies | ~1.8k Gemma-3-12B rows | Reference lie labels | datasets/noddybear_lies_subset/ | From the Anthropic blog |
| Probe-training sets | Apollo / TTPD repos | 611 RepE facts, ~371 roleplay scenarios, 41 TTPD csvs | Probe training | datasets/probe_training/ | Copied from code/ |
| (Liars' Bench HF) | Cadenza-Labs/liars-bench | — | — | not downloaded | Gated (403). Repo data in code/liars_bench/data |

## Code repositories

There are 12 cloned repositories, documented in `code/README.md`.

| Name | URL | Purpose | Location |
|------|-----|---------|----------|
| deception-detection | github.com/ApolloResearch/deception-detection | Deception probes (LR / MMS / LAT) and data | code/apollo_deception_detection/ |
| mask | github.com/centerforaisafety/mask | MASK generation, judge and metrics | code/mask/ |
| liars-bench | github.com/Cadenza-Labs/liars-bench | Lie generation, probes, black-box detectors and data | code/liars_bench/ |
| semantic-entropy-probes | github.com/OATML/semantic-entropy-probes | SE and SEP | code/semantic_entropy_probes/ |
| semantic_uncertainty | github.com/jlko/semantic_uncertainty | Semantic entropy | code/semantic_uncertainty/ |
| LLMsKnow | github.com/technion-cs-nlp/LLMsKnow | Correctness probes, P(True), logprob | code/llms_know/ |
| hallucination-mitigation | github.com/technion-cs-nlp/hallucination-mitigation | WACK HK+/HK− | code/wack_hallucination_mitigation/ |
| Truth_is_Universal | github.com/sciai-lab/Truth_is_Universal | TTPD truth probes | code/truth_is_universal/ |
| geometry-of-truth | github.com/saprmarks/geometry-of-truth | Truth probes and datasets | code/geometry_of_truth/ |
| representation-engineering | github.com/andyzoujm/representation-engineering | RepE honesty | code/representation_engineering/ |
| LLM-LieDetector | github.com/LoryPack/LLM-LieDetector | Black-box unrelated-questions detector | code/pacchiardi_lie_detector/ |
| sycophancy-eval | github.com/meg-tong/sycophancy-eval | Sycophancy data utilities | code/sycophancy_eval/ |

## Resource gathering notes

### Search strategy

- The paper-finder service (localhost:8000) returned HTTP 500 for every query, in both diligent and fast mode.
- Search therefore fell back to manual arXiv API queries (about 30 queries over lie detection, truth probes,
  hallucination detection, semantic entropy, sycophancy, deception taxonomies and knowledge-aware hallucination) plus
  known seminal papers. Logs are in `logs/search/`.
- The user's 9 arXiv IDs and 1 blog post were fetched first. All 9 were deep-read, along with 3 more discovered papers.
  Reading was split across 3 parallel reader agents, each of which read every main-body chunk.

### Selection criteria

Papers were chosen if they define or operationalize the lie vs hallucination distinction, provide the standard
detectors (both families), or provide matched-question or belief-verification methodology.

### Challenges

- Paper-finder was down.
- Liars' Bench on HF is gated.
- The host has no C compiler, so vLLM/Triton need the `zigcc` shim plus `VLLM_USE_FLASHINFER_SAMPLER=0` (see `code/README.md`).
- vLLM can leave an orphan EngineCore process holding GPU memory.
- TriviaQA alias matching needs care.

### Gaps and workarounds

- Liars' Bench data is available from the cloned repo instead of HF.
- AISI's Varied Deception prompts were not fetched (HF org `ai-safety-institute`). Comparable incentive prompts can be
  written in MASK style, or taken from sycophancy-eval.

## Recommendations for experiment design

1. **Primary dataset**: TriviaQA validation. Label knowledge per question for Llama-3.1-8B-Instruct, then run matched
   neutral vs incentive prompts on the same questions (incentives from sycophancy-eval `incorrect_answer`, plus
   system-prompt incentives).
2. **Secondary dataset**: MASK (known_facts, continuations, disinformation, statistics), to cross-check realistic
   implicit incentives. The judge can use OPENAI_API_KEY, which is set in the environment.
3. **Baselines**:
   - Lie detectors: Apollo RepE LR probe, truth probe (GoT/TTPD), DYL follow-up probe.
   - Hallucination detectors: semantic entropy, SEP, mean logprob, P(True), exact-answer correctness probe.
4. **Metrics**:
   - Shares of false outputs by mechanism, with bootstrap CIs.
   - AUROC for lie vs hallucination, lie vs honest, and hallucination vs correct.
   - Cross-firing rates at 1% FPR.
   - Cosine similarity between directions.
5. **Code to adapt**: probe recipes from `code/apollo_deception_detection/deception_detection/{data/repe.py,detectors.py}`,
   SE from `code/semantic_entropy_probes/semantic_uncertainty/`, the knowledge test from WACK / `scripts/smoke_test_knowledge.py`,
   and MASK judge prompts from `code/mask/mask/prompts/`.
