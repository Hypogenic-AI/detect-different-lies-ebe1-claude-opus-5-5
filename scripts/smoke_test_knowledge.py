"""Run with: CC=$PWD/.venv/bin/zigcc VLLM_USE_FLASHINFER_SAMPLER=0 python scripts/smoke_test_knowledge.py 200
Smoke test (resource_finder phase): WACK-style knowledge test on 200 TriviaQA val questions
with Llama-3.1-8B-Instruct via vLLM, then hidden-state extraction with HF transformers.
Known = greedy + 5 samples@T=0.5 all contain a gold alias; Unknown = none do."""
import json, re, string, sys
from datasets import load_from_disk
from vllm import LLM, SamplingParams

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200

def norm(s):
    s = s.lower(); s = "".join(" " if c in set(string.punctuation) else c for c in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s); return " ".join(s.split())

ds = load_from_disk("datasets/trivia_qa_rc_nocontext")["validation"].select(range(N))
llm = LLM(MODEL, gpu_memory_utilization=0.6, max_model_len=1024, seed=0, enforce_eager=True)  # no C compiler on host -> no torch.compile
tok = llm.get_tokenizer()
sysmsg = "Answer the question with just the answer (a few words), no explanation."
prompts = [tok.apply_chat_template([{"role": "system", "content": sysmsg}, {"role": "user", "content": q}],
                                   tokenize=False, add_generation_prompt=True) for q in ds["question"]]
greedy = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=24))
samp = llm.generate(prompts, SamplingParams(temperature=0.5, n=5, max_tokens=24, seed=0))
known = unknown = mid = 0; rows = []
for ex, g, s in zip(ds, greedy, samp):
    aliases = [norm(a) for a in ex["answer"]["normalized_aliases"]]
    outs = [g.outputs[0].text] + [o.text for o in s.outputs]
    hits = [any(a and a in norm(o) for a in aliases) for o in outs]
    k = sum(hits); lab = "known" if k == 6 else ("unknown" if k == 0 else "mid")
    known += lab == "known"; unknown += lab == "unknown"; mid += lab == "mid"
    rows.append({"q": ex["question"], "gold": ex["answer"]["value"], "greedy": outs[0].strip(), "n_correct": k, "label": lab})
print(json.dumps({"N": N, "known": known, "unknown": unknown, "mid": mid}))
json.dump(rows, open("logs/smoke_test_knowledge.json", "w"), indent=1)
import os; os._exit(0)  # vLLM engine can hang on interpreter shutdown
