"""Stage 4: judge every greedy response and same-context follow-up from stage 3; flag distractor adoption."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from judge import load_judge, judge

rows = read_jsonl(f"{RESULTS}/pressure_gen.jsonl")
llm = load_judge()
items = [dict(question=r["question"], gold=r["gold"], aliases=r["aliases"], resp=r["response"]) for r in rows]
items += [dict(question=r["question"], gold=r["gold"], aliases=r["aliases"], resp=r["followup"]) for r in rows]
labs = judge(llm, items)
n = len(rows)
for r, l, fl in zip(rows, labs[:n], labs[n:]):
    r["label"] = l
    r["followup_label"] = fl
    r["adopted_distractor"] = alias_match(r["response"], [norm(r["distractor"])])
    # consistency of 5 samples with the greedy answer (normalised containment either way)
    g = norm(r["response"])
    r["consistency"] = float(np.mean([bool(g) and (g in norm(s) or norm(s) in g) and bool(norm(s)) for s in r["samples"]]))
write_jsonl(rows, f"{RESULTS}/pressure_labeled.jsonl")
import collections
print(collections.Counter((r["cond"], r["knowledge"], r["label"]) for r in rows))
os._exit(0)
