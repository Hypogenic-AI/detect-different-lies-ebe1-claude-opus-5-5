# Downloaded Papers

38 arXiv PDFs + 1 blog post (summary only). Detailed notes for the 12 deep-read papers are in
`papers/notes/<arxiv_id>.md`. `papers/pages/` holds 4-page PDF chunks of those 12 papers.
★ = specified by the user in the research topic. 📖 = deep-read (all main-body chunks), notes file exists.

## A. User-specified (all deep-read)
| # | File | Title | Authors (year) | Why relevant |
|---|------|-------|----------------|--------------|
| 1 ★📖 | 2503.03750_mask_benchmark_honesty_vs_accuracy.pdf | The MASK Benchmark: Disentangling Honesty From Accuracy | Ren et al. (2025) | Lie = statement ≠ belief; neutral belief elicitation + pressure prompt on the same proposition. Core dataset. |
| 2 ★📖 | 2511.16035_liars_bench.pdf | Liars' Bench: Evaluating Lie Detectors for LMs | Kretschmar, Laurito, Maiya, Marks (2025) | 72,863 lies/honest responses from 4 models across 7 settings; lie detectors fail on some lie types; noise-floor belief verification. |
| 3 ★📖 | 2502.03407_detecting_strategic_deception_linear_probes.pdf | Detecting Strategic Deception Using Linear Probes | Goldowsky-Dill et al. (Apollo, 2025) | Standard white-box deception probe recipe (RepE instructed pairs, roleplaying); our main lie-detector baseline. |
| 4 ★📖 | 2602.01425_one_probe_wont_catch_them_all.pdf | One Probe Won't Catch Them All | Natarajan et al. (2026) | Deception is heterogeneous; prompt choice explains 70.6% of probe variance; MASK known-facts AUROC 0.374 for default probe. |
| 5 ★📖 | 2603.10003_limits_of_lie_detector_approach.pdf | Probing the Limits of the Lie Detector Approach | Thormann (2026) | Truth probes detect falsity, miss non-lying deception; lie detector ≈ falsehood detector. |
| 6 ★📖 | 2609.00180_spontaneous_vs_instructed_deception.pdf | Asymmetries in Spontaneous and Instructed Deception | Luikham (2026) | Instructed vs incentive-driven deception directions share cos≈0.5; asymmetric transfer. |
| 7 ★📖 | 2607.20479_beyond_liars_bench_lie_typology.pdf | Beyond Liars' Bench: Lie Typology, Depth, Sparsity | Moustafa, Feser, Mai (2026) | Lie typology of training data changes detectability; some sets anti-correlated (AUROC 0.12-0.16). |
| 8 ★📖 | 2510.09033_internal_states_reflect_knowledge_recall.pdf | Do LLMs Really Know What They Don't Know? | Cheang et al. (2025) | Hidden states reflect knowledge recall, not truthfulness; associated vs unassociated hallucinations. |
| 9 ★📖 | 2606.12618_did_you_lie_lie_detectors_scale.pdf | "Did you lie?" Lie Detectors across Model Scale | Cooney, Africa, Irving (UK AISI, 2026) | TriviaQA prompted-lying testbed (Varied Deception); Apollo, DYL follow-up probe, Unrelated-Qs detectors. |
| 10 ★ | notes/anthropic_2026_lie_detectors_blog.md | Fine-Tuned Lie Detectors Failed to Generalize (Anthropic blog) | Hopkins, Khullar, Wang, Roger (2026) | Parallel interrogation for belief; detectors learn setting features; data `Noddybear/lies`. |

## B. Discovered — hallucination vs knowledge (deep-read 📖 or skimmed)
| File | Title | Year | Relevance |
|------|-------|------|-----------|
| 📖 2410.22071_distinguishing_ignorance_from_error.pdf | Distinguishing Ignorance from Error in LLM Hallucinations (WACK; HK− vs HK+) | 2024 | Operational "knows" test on TriviaQA (greedy + 5 samples @T=0.5); HK+ = wrong despite knowing. |
| 📖 2410.02707_llms_know_more_than_they_show.pdf | LLMs Know More Than They Show | 2024 | Exact-answer-token probes; error types from resampling; probes generalize poorly across tasks. |
| 📖 2509.03518_can_llms_lie_beyond_hallucination.pdf | Can LLMs Lie? Investigation beyond Hallucination | 2025 | Directly contrasts lies vs hallucinations in representations (but lie direction from prompts containing "lie"). |
| 2502.12964_trust_me_im_wrong_chak.pdf | Trust Me, I'm Wrong (CHOKE) | 2025 | Certain hallucinations despite knowledge — third false-output type. |
| 2406.15927_semantic_entropy_probes.pdf | Semantic Entropy Probes | 2024 | Cheap hallucination detector from hidden states; baseline. |
| 2302.09664_semantic_uncertainty.pdf | Semantic Uncertainty (semantic entropy) | 2023 | Sampling-based hallucination detector baseline. |
| 2207.05221_llms_mostly_know_what_they_know.pdf | Language Models (Mostly) Know What They Know (P(True), P(IK)) | 2022 | Self-evaluation baselines. |
| 2312.03729_cognitive_dissonance_truthfulness.pdf | Cognitive Dissonance (confabulation vs deception vs heterogeneity) | 2023 | Taxonomy of query/probe disagreement; most is calibration not deception. |

## C. Discovered — truth probes / lie detection
| File | Title | Year | Relevance |
|------|-------|------|-----------|
| 2304.13734_internal_state_knows_when_lying.pdf | The Internal State of an LLM Knows When It's Lying (SAPLMA) | 2023 | True/false statement probes; dataset used by Apollo. |
| 2310.06824_geometry_of_truth.pdf | The Geometry of Truth | 2023 | Mass-mean probes, curated true/false datasets. |
| 2407.12831_truth_is_universal.pdf | Truth is Universal (TTPD) | 2024 | 2-D truth subspace; robust lie detection; datasets in repo. |
| 2310.01405_representation_engineering.pdf | Representation Engineering | 2023 | Honest/dishonest contrast pairs (source of the Apollo training set). |
| 2212.03827_discovering_latent_knowledge_ccs.pdf | Discovering Latent Knowledge (CCS) | 2022 | Unsupervised truth direction. |
| 2306.03341_inference_time_intervention.pdf | Inference-Time Intervention | 2023 | Truthfulness directions in attention heads. |
| 2309.15840_catch_an_ai_liar_blackbox.pdf | How to Catch an AI Liar (unrelated follow-up questions) | 2023 | Black-box lie detector baseline. |
| 2507.12691_black_to_white_deception_probes.pdf | Benchmarking Deception Probes via Black-to-White Boosts | 2025 | Probe evaluation methodology. |
| 2507.22149_truthful_representations_flip.pdf | When Truthful Representations Flip Under Deceptive Instructions | 2025 | Llama-3.1-8B / Gemma-2-9B: instructed lies keep truth representation decodable. |
| 2602.20273_truthfulness_spectrum_hypothesis.pdf | The Truthfulness Spectrum Hypothesis | 2026 | Probes fail on sycophantic lying unless trained jointly. |
| 2608.12852_falsehood_vs_impossibility_directions.pdf | Falsehood and Impossibility Are Different Directions | 2026 | Distinct directions for kinds of "wrongness". |
| 2609.02893_probe_generalization_subspace_selection.pdf | Probe Generalization as Subspace Selection (Llama-3.1-8B) | 2026 | PCA-subspace probes for OOD deception. |
| 2511.17408_off_policy_training_data_probes.pdf | Impact of Off-Policy Training Data on Probe Generalisation | 2025 | On- vs off-policy probe training data. |
| 2609.39807_stress_testing_lie_detectors.pdf | Stress-Testing LLM Lie Detectors: Role-Play Failures | 2026 | 8 probes fail under anti-factual personas. |
| 2609.21996_lie_detector_test_reading_knowledge.pdf | A Lie Detector Test for LMs (Concealed Information Test, PIR) | 2026 | Reads which answer the model recognizes — knowledge vs concealment. |
| 2609.10739_truth_never_gone_aliasing.pdf | The Truth Was Never Gone: Perfect Aliasing | 2026 | Truth vs prescribed-action probes are confounded in compliant contexts. |
| 2606.17229_rift_conflict_signature_deception.pdf | Rift: A Conflict Signature for Deception | 2026 | Controls for wrongness: liar vs naive-wrong model with identical outputs. |
| 2604.19117_shared_sycophancy_lying_circuit.pdf | LLMs Know They're Wrong and Agree Anyway | 2026 | Shared "this is wrong" heads for sycophancy and lying. |
| 2608.26372_knowledge_verified_emergent_deception.pdf | Knowledge-Verified Emergent Deception (KnownLieBench) | 2026 | Neutral knowledge probe then incentive — same design logic. |
| 2604.04788_sycophancy_to_deception_taxonomy.pdf | From Sycophancy to Deception: Unified Taxonomy | 2026 | Taxonomy (goal-directedness × object × mechanism). |
| 2606.11502_roleplay_do_models_believe.pdf | When Role-playing, Do Models Believe What They Say? | 2026 | Roleplay may change internal truth representation (confound). |

Search logs: `logs/search/` (paper-finder service returned HTTP 500, so the search was done manually via the arXiv API).
