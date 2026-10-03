"""Exp 1: neutral knowledge test. Greedy + 10 samples (T=1.0) per TriviaQA val question with the subject model.
Run: CC=$PWD/.venv/bin/zigcc VLLM_USE_FLASHINFER_SAMPLER=0 python src/gen_knowledge.py"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from datasets import load_from_disk
from vllm import LLM, SamplingParams

set_seed()
ds = load_from_disk("datasets/trivia_qa_rc_nocontext")["validation"]
idx = np.random.RandomState(SEED).permutation(len(ds))[:N_QUESTIONS]
qs = []
for i in idx:
    ex = ds[int(i)]
    al = sorted({norm(a) for a in ex["answer"]["normalized_aliases"] + ex["answer"]["aliases"]} - {""})
    qs.append({"qid": ex["question_id"], "question": ex["question"], "gold": ex["answer"]["value"], "aliases": al})

llm = LLM(SUBJECT_MODEL, gpu_memory_utilization=0.85, max_model_len=1024, seed=SEED, enforce_eager=True)
tok = llm.get_tokenizer()
prompts = [tok.apply_chat_template(build_messages("neutral", q["question"]), tokenize=False, add_generation_prompt=True) for q in qs]
g = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=32))
s = llm.generate(prompts, SamplingParams(temperature=1.0, n=10, max_tokens=32, seed=SEED))
for q, gi, si in zip(qs, g, s):
    q["greedy"] = gi.outputs[0].text.strip()
    q["samples"] = [o.text.strip() for o in si.outputs]
os.makedirs(RESULTS, exist_ok=True)
write_jsonl(qs, f"{RESULTS}/knowledge_gen.jsonl")
print("done", len(qs))
os._exit(0)
