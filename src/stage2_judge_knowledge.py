"""Stage 2: (a) generate a distractor (plausible wrong answer) per question with the judge model,
(b) judge every neutral greedy answer and every non-alias-matching neutral sample, (c) assign knowledge labels.
Known   = greedy CORRECT and >= 9/10 samples CORRECT.
Unknown = greedy INCORRECT and <= 1/10 samples CORRECT.   (EVASION counts as not correct.)
Partial = everything else."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from judge import load_judge, judge, distractors

qs = read_jsonl(f"{RESULTS}/knowledge_gen.jsonl")
llm = load_judge()

# (a) distractors, rejected if they alias-match the gold or the judge says they are correct
ds = distractors(llm, qs)
dj = judge(llm, [dict(q, resp=d) for q, d in zip(qs, ds)])
for q, d, lab in zip(qs, ds, dj):
    q["distractor"] = d
    q["distractor_ok"] = bool(d) and lab == "INCORRECT" and not alias_match(d, q["aliases"]) and len(d) < 80

# (b) judge neutral outputs: dedupe by (qid, normalised response); alias match -> CORRECT without judge
cache, todo = {}, []
for q in qs:
    for r in [q["greedy"]] + q["samples"]:
        key = (q["qid"], norm(r))
        if key in cache:
            continue
        if alias_match(r, q["aliases"]):
            cache[key] = "CORRECT"
        else:
            cache[key] = None; todo.append((key, dict(question=q["question"], gold=q["gold"], aliases=q["aliases"], resp=r)))
print("judge calls:", len(todo))
labs = judge(llm, [t[1] for t in todo])
for (key, _), l in zip(todo, labs):
    cache[key] = l

for q in qs:
    q["greedy_label"] = cache[(q["qid"], norm(q["greedy"]))]
    q["sample_labels"] = [cache[(q["qid"], norm(r))] for r in q["samples"]]
    nc = sum(l == "CORRECT" for l in q["sample_labels"])
    q["n_sample_correct"] = nc
    if q["greedy_label"] == "CORRECT" and nc >= 9:
        q["knowledge"] = "known"
    elif q["greedy_label"] != "CORRECT" and nc <= 1:
        q["knowledge"] = "unknown"
    else:
        q["knowledge"] = "partial"
write_jsonl(qs, f"{RESULTS}/knowledge_labels.jsonl")
import collections
print(collections.Counter(q["knowledge"] for q in qs), collections.Counter(q["greedy_label"] for q in qs),
      "distractor_ok", sum(q["distractor_ok"] for q in qs))
os._exit(0)
