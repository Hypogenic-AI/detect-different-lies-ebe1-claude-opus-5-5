# Literature Review: Detecting *different* lies — hallucination vs incentive-driven misreporting

Detailed per-paper notes: `papers/notes/<arxiv_id>.md` (12 deep-read papers + Anthropic blog). Catalog: `papers/README.md`.

## 1. Research area overview

The literature separates false LLM outputs along one key axis: **does the model represent the truth?**

- **Hallucination / confabulation (epistemic failure).** The model outputs a falsehood it does not know is false.
  Sub-types matter:
  - *Unassociated* errors (no parametric knowledge, high uncertainty: HK− in WACK; UH in 2510.09033).
  - *Associated / confident* errors (spurious association, low uncertainty: AH, CHOKE).
  - *HK+* errors: the model knows the answer but a perturbation such as wrong few-shot examples ("snowballing") derails
    it, with no incentive involved.
- **Lie (incentive-driven misreporting).** A statement that contradicts the model's own belief, made under an incentive
  or instruction (MASK; Pacchiardi 2023; Liars' Bench; AISI 2606.12618). Intent is usually operationalized by the
  scenario. Belief is operationalized behaviourally: neutral-context answers that are consistent across samples and
  paraphrases.
- **Deception without lying** (misleading truths, omission) and **sincere persona beliefs** (roleplay) form further
  categories that confuse truth-probe-based "lie detectors" (2603.10003, 2606.11502, 2609.39807).

The two detector families grew up in separate communities:

- **Lie detectors**: deception probes trained on instructed honest/dishonest pairs (RepE → Apollo), truth probes
  (SAPLMA, Geometry of Truth, TTPD), follow-up-question detectors (Pacchiardi unrelated-Qs; AISI "Did-You-Lie" probe),
  and LLM/CoT judges.
- **Hallucination detectors**: semantic entropy, semantic-entropy probes, token log-prob, P(True)/P(IK), and
  correctness probes at exact-answer tokens.

**No paper we found evaluates both families on a matched set where the same model's falsehoods are labelled as
hallucination vs lie.** Every deep-read lie-detection paper uses *honest = true* answers as negatives. As a result,
nobody knows whether lie detectors fire on sincere-but-wrong answers, or whether hallucination detectors flag lies.
This is the gap our project targets.

## 2. Key papers (condensed; numbers are from the notes)

**User-specified**
- **MASK (2503.03750).**
  - Definitions: lie = statement S ≠ belief B. Belief = neutral answer (asked 3×, plus 2 indirect questions for binary
    items), all consistent. Otherwise the item is "no belief". Honesty and accuracy are disentangled.
  - Data: 1,000 public items in 6 archetypes.
  - Method: an LLM judge maps each response to a proposition resolution, then the code compares resolutions (86.4%
    agreement with humans).
  - Llama-3.1-8B-Instruct: P(lie) 23.5%, P(honest) 18.8%, accuracy 62%. Qwen2.5-7B: P(lie) 39.0%.
  - Frontier models lie 27–63% of the time, and honesty does not improve with scale.
- **Liars' Bench (2511.16035).**
  - Data: 72,863 examples, 4 models, 7 settings, varied by reason for lying × object of belief.
  - Detectors evaluated: LLM-as-judge, self-evaluation, Apollo-style probes. All fail on some lie types, especially
    when a lie cannot be identified from the transcript (e.g., an "I don't know" lie: balanced accuracy 0.54, AUROC 0.36).
  - Belief verification method: estimate a noise floor from neutral re-asks, and require pressure to raise
    contradiction ≥4× above it. In one subset, 15% of lies are *correct* answers that contradict a wrong belief, which
    shows that lying and being wrong come apart.
- **Apollo deception probes (2502.03407).** The standard recipe:
  - Training pairs: RepE "Pretend you're an honest/dishonest person making statements about the world." + a true fact,
    with the last 5 tokens dropped.
  - Classifier: logistic regression on standardized residual activations, λ=10, using the mean over response tokens.
  - Threshold: 1% FPR on Alpaca.
  - Llama-3.3-70B AUROC 0.96–0.999. The probe also fires on honest answers in lie-incentivized contexts and on
    confessions, which shows it partly tracks "deception is salient in context".
- **One Probe Won't Catch Them All (2602.01425).**
  - Prompt choice explains 70.6% of probe-performance variance.
  - The default probe gets AUROC 0.374 on MASK known-facts on Gemma-2-9B-IT.
  - Recipe: LR at layer 20, L2 λ=1, mean over response tokens excluding the last 5.
- **Limits of the lie-detector approach (2603.10003).** Truth probes reach 0.80–0.84 on lies but miss deception
  through misleading truths. "Lie detectors" are essentially falsity detectors, which leads to the prediction that
  they will also fire on hallucinations whenever the model internally represents the output as false.
- **Spontaneous vs instructed deception (2609.00180; Llama-3.1-70B).**
  - The instructed and spontaneous directions share cos≈0.5, with asymmetric transfer.
  - The signal sits in response tokens.
  - Under incentive the model omits far more than it fabricates, so labels must handle evasion.
- **Beyond Liars' Bench (2607.20479).** Seven probe types and SAE features: the optimal depth depends on the dataset,
  and the lie typology of the training data dominates. Knowledge-denial (HP-KR) AUROC is 0.12–0.16, i.e.
  anti-correlated, consistent with probes tracking "confident assertion".
- **Internal states reflect knowledge recall (2510.09033).**
  - Hidden-state, logprob and consistency detectors reach AUROC 0.84–0.93 on unassociated hallucinations but only
    0.48–0.69 on associated (confident) ones. They track *recall*, not truth.
  - Prediction: lies (full recall of the correct answer) will look like *correct* answers to hallucination detectors.
- **"Did you lie?" (2606.12618, UK AISI).**
  - Varied Deception: TriviaQA + 64 honest / 64 lie-inducing system prompts, kept only if the model answers correctly
    under a neutral prompt. This is essentially our lie condition.
  - Detectors: Apollo probe, Did-You-Lie follow-up probe (a probe on the "yes/no" token after asking "Did you lie in
    the last message?"), unrelated-questions logprob classifier, CoT judge.
  - Prompted lying: median balanced accuracy 0.81–0.84. On trained model organisms this drops to about 0.50–0.65.
  - No hallucination control.
- **Anthropic blog (2026): fine-tuned lie detectors fail to generalize.**
  - Belief check by parallel interrogation; about 25% of labels were revised.
  - Fine-tuned detectors reach 0.95 AUROC in-distribution but only 0.70–0.75 across categories.
  - Detectors learn setting features. Data: `Noddybear/lies`.

**Discovered (most relevant)**
- **WACK (2410.22071).**
  - Knowledge test: greedy + 5 samples at T=0.5; "known" = 6/6 contain the gold answer, "unknown" = 0/6, about 5% in
    between. The labels are stable (92–94% agreement across configurations).
  - For 7–9B models on TriviaQA, about 67% of questions are known.
  - HK+ rate among known questions: 4–6% under mild perturbations, 11–18% under snowballing.
  - HK+ vs HK− are only moderately linearly separable (55–80%).
- **LLMs Know More Than They Show (2410.02707).**
  - Correctness probes at the exact answer token work best.
  - Probes at the end of the question already predict correctness (≈0.75), i.e. they are knowledge probes.
  - Error types are defined via resampling, and probes generalize poorly across tasks.
- **Can LLMs Lie? (2509.03518; Llama-3.1-8B-Instruct).**
  - PCA shows Lie forms a separate cluster, while Hallucination stays near Truth.
  - Logit lens shows the truth is "rehearsed" at the chat-template tokens before a lie. Lie formation happens in
    layers 10–15.
  - Caveat: the lie direction comes from prompts containing "lie", so instruction and act are confounded.
- **Cognitive Dissonance (2312.03729).** Probe-vs-output disagreements mostly reflect calibration on uncertain items
  (confabulation), not deception. This is a direct precedent for "how often is falsehood epistemic vs strategic".
- **Rift (2606.17229).** Controls for wrongness: a knows-but-lies model vs a naive wrong model with identical outputs
  shows a "conflict signature". This is the same logic as our matched design.
- **Shared sycophancy–lying circuit (2604.19117).** The same attention heads carry "this is wrong" during sycophancy
  and lying. This predicts that sycophantic false answers on known questions are lies, mechanistically.
- **Others:** SEP (2406.15927) and semantic entropy (2302.09664) are the hallucination baselines. P(True)/P(IK)
  (2207.05221). TTPD (2407.12831) and Geometry of Truth (2310.06824) are truth-probe baselines. Pacchiardi
  (2309.15840) is the black-box detector. Truthfulness spectrum (2602.20273): probes fail on sycophantic lying unless
  trained jointly. Perfect aliasing (2609.10739): truth vs prescribed action are confounded. Probes on Llama-3.1-8B for
  OOD deception (2609.02893).

## 3. Common methodologies

- **Belief/knowledge operationalization.**
  - Sampling consistency under a neutral prompt: WACK (6/6), AISI (1 neutral correct answer), MASK (3 direct + 2
    indirect), Liars' Bench (noise floor + ×4 rule).
  - Optionally a white-box knowledge check: logit lens showing the correct answer in mid layers (2509.03518), or a
    probe at the last question token (2410.02707).
- **Lie induction.**
  - System-prompt role/incentive (MASK, AISI Varied Deception).
  - User-suggested answers / sycophancy (sycophancy-eval).
  - Explicit instruction (RepE, Pacchiardi).
  - Doubling down.
- **Probes.** Logistic regression (L2) on standardized residual-stream activations at one layer (sweep at 20–60% of
  depth), mean-pooled over response tokens. Alternatives: diff-of-means / LAT, and probes at a single token (exact
  answer token, last prompt token, follow-up answer token).
- **Hallucination detectors.** Semantic entropy over 5–10 samples (NLI clustering), SEP (LR on hidden state →
  binarized semantic entropy), mean/min token logprob, P(True), correctness probes.

## 4. Standard baselines and metrics

- **Baselines.**
  - Apollo RepE deception probe (LR).
  - Truth probe (LR / mass-mean on true/false statements).
  - DYL follow-up probe.
  - Pacchiardi unrelated-Qs (black-box).
  - Semantic entropy; SEP; token logprob; P(True); correctness probe.
  - An LLM judge (black-box).
- **Metrics.**
  - AUROC for each pairwise contrast.
  - Balanced accuracy / TPR at a threshold calibrated to 1% FPR on Alpaca (or on honest-correct answers).
  - Recall at fixed FPR.
  - Shares (%) with bootstrap 95% CIs.
  - For the hypothesis, the key quantities are **AUROC(lie vs hallucination)** and the **cross-firing rate**: lie
    detector positive rate on hallucinations, and hallucination detector positive rate on lies.

## 5. Datasets in the literature

- TriviaQA: WACK, LLMsKnow, AISI Varied Deception, SEP, sycophancy-eval.
- MASK: MASK, One-Probe, Liars' Bench, Anthropic blog.
- Azaria–Mitchell true/false facts: Apollo, AISI, Thormann.
- Geometry-of-Truth / TTPD datasets.
- Apollo roleplaying, insider trading, sandbagging.
- Alpaca as the FPR control.
- Natural Questions: WACK, LLMsKnow.

## 6. Gaps and opportunities

1. **No hallucination negatives in lie-detector evaluation.** No paper tests whether deception probes fire on honest
   errors. This is the central gap.
2. **No lie positives in hallucination-detector evaluation.** Hallucination detectors are never tested on known-answer
   lies. The prediction from 2510.09033 is that they will call lies "confident/known".
3. **No prevalence decomposition.** Shares of false outputs by mechanism (epistemic vs incentive) are reported only
   implicitly (MASK no-belief vs lie; Cognitive Dissonance). None are reported on a matched question set for one model.
4. **Incentive × knowledge interaction is unstudied.** What happens when the model is pressured on *unknown*
   questions? Its falsehoods there are neither clean lies nor clean hallucinations ("bullshit").
5. **Instructed-lie confound.** Probes may detect the lie instruction in context rather than the act of lying. The
   needed controls are incentive prompts where the model stays honest, and lie prompts on unknown questions.

## 7. Research directions — enumerated, scored and pruned (DIRECTION BUDGET = 3)

Each direction is scored 1–5 on Evidence (literature support), Relevance (to the hypothesis), Information gain, and
Feasibility (one A6000 GPU, open 8B model, limited time). Total is out of 20.

| # | Direction | Ev | Rel | Gain | Feas | Total | Decision |
|---|-----------|----|-----|------|------|-------|----------|
| D1 | **Matched TriviaQA factorial design.** Knowledge test (greedy + 5–10 samples) → known/unknown. Neutral vs incentive prompts (sycophancy-eval user suggestion; MASK-/AISI-style system prompts). Classify every false output as hallucination (unknown, neutral), lie (known, incentive, belief verified), HK+-like error (known, neutral/control), or pressured-unknown. Report shares with CIs. | 5 | 5 | 5 | 5 | **20** | **KEEP (primary)** |
| D2 | **Detector battery on the matched categories.** Lie detectors (Apollo RepE LR probe, truth probe, DYL follow-up probe, optional unrelated-Qs) and hallucination detectors (semantic entropy, SEP/correctness probe, logprob, P(True)). Measure pairwise AUROCs (lie vs halluc, lie vs honest-correct, halluc vs correct), cross-firing at 1% FPR, and the geometry of lie vs hallucination vs uncertainty directions (cosine, logit-lens knowledge check). | 5 | 5 | 5 | 4 | **19** | **KEEP** |
| D3 | **MASK cross-check on the same model.** Run MASK known_facts / continuations / disinformation / statistics with belief elicitation, and decompose pressured falsehoods into lie vs no-belief vs wrong-belief. Apply the D2 detectors for external validity (realistic, implicit incentives). Needs an LLM judge (OPENAI_API_KEY is present). | 4 | 5 | 4 | 4 | **17** | **KEEP** |
| D4 | HK+ "snowballing" control only (WACK) | 4 | 3 | 3 | 5 | 15 | Pruned as a direction; folded into D1 as a cheap control condition |
| D5 | SAE-feature lie vs hallucination detectors | 3 | 3 | 3 | 2 | 11 | Pruned: 2607.20479 shows SAE ≈ dense probes; no instruct SAE readily available |
| D6 | Fine-tuned (LoRA) lie classifiers | 3 | 3 | 2 | 2 | 10 | Pruned: Anthropic blog already shows poor generalization; costly |
| D7 | Trained model organisms with hidden beliefs | 4 | 3 | 4 | 1 | 12 | Pruned: training cost and belief verification are out of budget |
| D8 | Multi-model scaling study | 4 | 2 | 3 | 2 | 11 | Pruned: the hypothesis specifies one model; at most a replication on Qwen2.5-7B if time permits |
| D9 | Non-lying deception (misleading truths, omission) | 3 | 2 | 3 | 3 | 11 | Pruned: outside the hallucination-vs-lie question |
| D10 | CoT-judge / reasoning-model lie detection | 3 | 2 | 2 | 3 | 10 | Pruned: Llama-8B has no CoT, and judges are black-box (keep a judge only as a labeler) |

## 8. Recommendations for our experiment

- **Model:** `meta-llama/Llama-3.1-8B-Instruct`.
  - It is the most-studied model in this literature: MASK numbers, 2509.03518 lie layers 10–15, 2507.22149, 2609.02893.
  - Weights are cached and the vLLM + HF pipeline is smoke-tested.
  - Backup: Qwen2.5-7B-Instruct (MASK P(lie) 39%).
- **Knowledge labels (TriviaQA validation).**
  - Chat template, short-answer system prompt; greedy + 10 samples at T=1.0 (or WACK's 5 at T=0.5).
  - Known = greedy correct and ≥9/10 correct; Unknown = 0 correct; drop the middle.
  - Match with normalized aliases (punctuation → space), plus an LLM-judge fallback for non-matches.
  - Expected rates from the smoke test (n=200): ~52% known, ~28% unknown, ~21% mixed.
- **Lie induction on known questions** (several incentive types, without the word "lie" where possible):
  1. sycophancy-eval user suggestion "I think the answer is {incorrect_answer}…";
  2. MASK/AISI-style system-prompt incentives with a target wrong answer (game, sales, persona pressure);
  3. explicit instruction ("give a wrong answer"), as the instructed-lie reference.
- **Verify each lie:**
  - the model answers correctly when re-asked neutrally (already true by the known filter);
  - optionally, it confesses to a DYL follow-up, or the logit lens shows the gold answer at the pre-answer template tokens.
- **Controls:**
  - the same incentive prompts on *unknown* questions (pressured-unknown);
  - incentive prompts where the model stays honest (known, incentive, correct);
  - snowballing prompts (HK+, no incentive).
- **Detectors to implement** (all fit on data *disjoint* from the evaluation questions):
  - (a) Apollo RepE LR probe: `datasets/probe_training/repe_true_false_facts.csv`, mean over response tokens, layer
    sweep around 8–20;
  - (b) truth probe on Geometry-of-Truth / TTPD statements;
  - (c) DYL follow-up probe;
  - (d) semantic entropy (10 samples; cluster by normalized-answer match or NLI) and SEP;
  - (e) mean token logprob;
  - (f) P(True);
  - (g) correctness probe at the exact answer token, trained on *neutral-prompt* correct vs hallucinated answers.
- **Key analyses:**
  - shares of false outputs by mechanism, with bootstrap CIs, overall and per incentive type;
  - per-detector score distributions over {honest-correct, hallucination, lie, pressured-unknown, HK+};
  - AUROC(lie vs hallucination) and cross-firing rates at a threshold set to 1% FPR on honest-correct (and on Alpaca);
  - cosine between the lie-probe direction and the correctness/uncertainty-probe direction.
- **Methodological cautions:**
  - calibrate thresholds per sample, not on pooled tokens (the AISI replication caveat);
  - avoid lexical cues ("lie") in the evaluation prompts;
  - do the train/test split by question;
  - report evasions separately (not as lies);
  - remember that TriviaQA alias lists are incomplete.
