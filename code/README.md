# Cloned Repositories

All repositories are shallow clones (`--depth 1`). Nothing here was user-specified as a code reference; they are
the official implementations of the key papers. The user-specified paper repos are marked ★.

| Dir | URL | Paper | What we use it for |
|-----|-----|-------|--------------------|
| `apollo_deception_detection/` ★ | github.com/ApolloResearch/deception-detection | 2502.03407 | **Main lie-detector baseline.** Detectors: `deception_detection/detectors.py` (`LogisticRegressionDetector`, `MMSDetector`, `CovarianceAdjustedMMSDetector`, `LATDetector`). RepE training data: `deception_detection/data/repe.py` + `data/repe/true_false_facts.csv`. Roleplaying: `data/roleplaying/dataset.yaml`. |
| `mask/` ★ | github.com/centerforaisafety/mask | 2503.03750 | MASK pipeline: `mask/generate_responses.py` (pressure + belief prompts), `mask/evaluate.py` (LLM judge → proposition resolution), `mask/metric.py`, `mask/process_metrics.py`. Judge prompts: `mask/prompts/evaluation_prompts.py`. Uses OpenAI API for the judge. |
| `liars_bench/` ★ | github.com/Cadenza-Labs/liars-bench | 2511.16035 | Lie-generation and detector code (`src/probes`, `src/blackbox`, `src/mask`, `src/instruct-dishonesty`). `data/` (40 MB) has processed data, and `data/probes/{GoT,RepEng}` holds probe-training data. The HF dataset is gated, so this is the accessible copy. |
| `semantic_entropy_probes/` | github.com/OATML/semantic-entropy-probes | 2406.15927 | **Hallucination-detector baseline.** Generation + semantic entropy (`semantic_uncertainty/`), SEP training notebook (`semantic_entropy_probes/train-latent-probe.ipynb`). |
| `semantic_uncertainty/` | github.com/jlko/semantic_uncertainty | 2302.09664 / Farquhar 2024 | Reference semantic-entropy implementation (NLI clustering with DeBERTa or an LLM). |
| `llms_know/` | github.com/technion-cs-nlp/LLMsKnow | 2410.02707 | Correctness probes at exact-answer tokens (`src/probe.py`, `probe_all_layers_and_tokens.py`), `p_true_detection.py`, `logprob_detection.py`, `probe_type_of_error.py` (resampling error types). Includes TriviaQA support. |
| `wack_hallucination_mitigation/` | github.com/technion-cs-nlp/hallucination-mitigation | 2410.22071 | WACK knowledge test (greedy + 5 samples @T=0.5), HK+/HK− dataset construction, SVM probes. |
| `truth_is_universal/` | github.com/sciai-lab/Truth_is_Universal | 2407.12831 | TTPD truth probe (`probes.py`), activation extraction (`generate_acts.py`), true/false datasets. |
| `geometry_of_truth/` | github.com/saprmarks/geometry-of-truth | 2310.06824 | Mass-mean / LR truth probes (`probes.py`), datasets. |
| `representation_engineering/` | github.com/andyzoujm/representation-engineering | 2310.01405 | RepE reading vectors (honesty), LoRRA. |
| `pacchiardi_lie_detector/` | github.com/LoryPack/LLM-LieDetector | 2309.15840 | Black-box lie detector: unrelated follow-up questions + logistic regression. The 48–50 probe questions are in `lllm/` / `data/`. |
| `sycophancy_eval/` | github.com/meg-tong/sycophancy-eval | Sharma et al. 2023 | Loading utils for the sycophancy datasets (TriviaQA with user-suggested answers). |

## Environment notes (important for the experiment runner)

- The venv is `.venv` (uv), with torch 2.13 + cu130, vLLM 0.30, transformers 5.18 and scikit-learn. GPU: 1× RTX A6000 48 GB.
- **The host has no C compiler.** For vLLM/Triton:
  `export CC=$PWD/.venv/bin/zigcc VLLM_USE_FLASHINFER_SAMPLER=0`
  (`.venv/bin/zigcc` is a shim around the `ziglang` pip package that rewrites `-l:libcuda.so.1`).
  Without it, vLLM fails with "Failed to find C compiler" or a FlashInfer ninja build error.
- vLLM can hang at interpreter exit. If you `os._exit`, the `VLLM::EngineCore` child can stay alive holding about 29 GB
  of GPU memory. Check with `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`, then `kill <pid>`. Do **not** use
  `pkill -f <pattern>` that matches your own shell command line.
- Smoke tests: `scripts/smoke_test_knowledge.py` (vLLM, TriviaQA knowledge test) and `scripts/smoke_test_hidden_states.py`
  (HF `output_hidden_states`; Llama-3.1-8B-Instruct gives 33 hidden states of dim 4096). Both pass.
- Llama-3.1-8B-Instruct weights are cached in `~/.cache/huggingface/hub`. The HF token can access the gated Llama, Gemma and
  Mistral models, but **not** the gated `Cadenza-Labs/liars-bench` dataset.
- The repos' own dependencies (e.g. Apollo's pyproject) were **not** installed. It is simpler to reimplement the probe
  recipes (LR on standardized residual activations) with sklearn than to install those packages.
- Crashed vLLM runs drop ~20 MB `core.<pid>` files in the workspace root; run `ulimit -c 0` first (or delete them afterwards).
