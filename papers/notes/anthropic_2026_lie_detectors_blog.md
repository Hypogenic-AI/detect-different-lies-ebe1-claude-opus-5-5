# Fine-Tuned Lie Detectors Failed to Generalize (Anthropic Alignment Science Blog, 2026-08-21)
URL: https://alignment.anthropic.com/2026/lie-detectors/  (not a PDF; summary from the web page)
Authors: Jack Hopkins, Dipika Khullar (MATS / Anthropic Fellows), Rowan Wang, Fabien Roger (Anthropic)

- **Question**: Can supervised fine-tuning on on-policy lies produce lie detectors that generalize across lie types?
- **Lie definition**: false statement + model knows it is false. Belief verified by "parallel interrogation":
  elicit belief in a neutral context, compare with the pressured response; lie only if it contradicts the
  demonstrated belief and persists under challenge. ~25% of initial labels were revised by a GPT-5 judge
  (shows how hard it is to separate deception from confusion / context-dependent belief).
- **Models**: Gemma-3 (4B-27B), Qwen-3 (80B-235B), GPT-OSS (20B/120B), plus GPT/Claude as judges.
- **Data**: 12 on-policy elicitation settings, ~200k labelled examples: factual lies (MASK-based), sandbagging
  (ASCII art, tool use, domain knowledge, chess), CoT unfaithfulness, sycophancy, self-sycophancy, roleplay;
  plus off-policy DolusChat & Truth-Is-Universal. Released: https://huggingface.co/datasets/Noddybear/lies
- **Detectors**: zero-shot prompting (self-report "did you lie?", third-person monitor, rubric variants) and
  LoRA fine-tuned Gemma-3-27B classifier (r=16, alpha=32, dropout 0.05).
- **Results**: Qwen3-235B prompting AUROC 0.98-0.99; Gemma-3-27B prompting 0.46-0.80 depending on lie type.
  Third-person monitoring > self-report. Fine-tuning: in-distribution 0.60 -> 0.95 AUROC, but cross-category
  generalization plateaus at 0.70-0.75. Sandbagging nearly undetectable by detectors trained on other lies;
  ASCII-denial <-> chess-sandbagging transfer, but not to factual lies. Lie rates 20-42% overall; sandbagging >76%.
- **Explanations offered**: limited introspective access; reflexive (stimulus-driven) vs strategic deception
  (46% label instability in self-sycophancy); context-dependent beliefs in roleplay.
- **Relevance**: shows detectors learn surface/setting features rather than "deception"; motivates our test of
  whether detectors respond to *falsehood/uncertainty* vs *knowing misreport*. Parallel-interrogation belief check
  is the same logic we use: neutral belief elicitation on the matched question, then pressured response.
