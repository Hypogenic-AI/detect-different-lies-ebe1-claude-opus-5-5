#!/bin/bash
# Re-create all datasets in datasets/. Run from the workspace root with .venv active:
#   source .venv/bin/activate && bash datasets/download.sh
set -e
python - << 'PY'
from datasets import load_dataset, get_dataset_config_names
from huggingface_hub import snapshot_download
# 1. MASK (all 6 configs; test split only)
for cfg in get_dataset_config_names("cais/MASK"):
    load_dataset("cais/MASK", cfg).save_to_disk(f"datasets/mask/{cfg}")
# 2. TriviaQA rc.nocontext (questions + answer aliases, no evidence docs)
load_dataset("mandarjoshi/trivia_qa", "rc.nocontext").save_to_disk("datasets/trivia_qa_rc_nocontext")
# 3. sycophancy-eval (TriviaQA questions with user-suggested wrong/right answers)
snapshot_download("meg-tong/sycophancy-eval", repo_type="dataset",
                  allow_patterns=["answer.jsonl", "are_you_sure.jsonl", "README.md"], local_dir="datasets/sycophancy_eval")
# 4. Noddybear/lies (Anthropic-fellows lie-detector data), small Gemma-3-12B subsets
snapshot_download("Noddybear/lies", repo_type="dataset", local_dir="datasets/noddybear_lies_subset",
                  allow_patterns=["README.md"] + [f"google-gemma-3-12b-it_task-group_{g}/*" for g in
                                                  ["mask-factual", "sycophancy", "unanswerable", "offpolicy"]])
PY
# 5. Probe-training data copied from cloned repos (needs code/ clones; see code/README.md)
mkdir -p datasets/probe_training
cp code/apollo_deception_detection/data/repe/true_false_facts.csv datasets/probe_training/repe_true_false_facts.csv
cp code/apollo_deception_detection/data/roleplaying/dataset.yaml datasets/probe_training/apollo_roleplaying_dataset.yaml
cp -r code/truth_is_universal/datasets datasets/probe_training/truth_is_universal
# NOTE: Cadenza-Labs/liars-bench is gated on HF (403 without accepting terms). The cloned repo
# code/liars_bench/data contains its processed/probe data instead.
