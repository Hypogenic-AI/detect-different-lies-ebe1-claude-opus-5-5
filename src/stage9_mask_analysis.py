"""Stage 9: MASK cross-check of the detectors (layer 12). Probes are fit on the full TriviaQA / RepE / truth data
(MASK is disjoint), then applied to MASK pressured statements and neutral belief answers.
Contrasts: lie vs honest (pressure prompts), lie vs false_no_belief (pressure, both false), and on neutral belief
answers: consistently-wrong belief (confident honest error) vs consistently-correct belief."""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from stage6_detectors import lr, load, category, LAYERS, PRIMARY
from sklearn.metrics import roc_auc_score
import collections

li = LAYERS.index(PRIMARY)
rng = np.random.RandomState(SEED)
ev, meta = load("eval"); repe, rmeta = load("repe"); tr, tmeta = load("truth"); mk, mmeta = load("mask")
cats = np.array([category(m) for m in meta]); cond = np.array([m["cond"] for m in meta])
lab = np.array([m["label"] for m in meta]); know = np.array([m["knowledge"] for m in meta])
f32 = lambda a: a[:, li].astype(np.float32)

S = {}
clf = lr().fit(f32(repe["resp_mean"]), np.array([m["deceptive"] for m in rmeta])); S["repe_probe"] = clf.decision_function(f32(mk["resp_mean"]))
clf = lr().fit(f32(tr["resp_last"]), np.array([m["label"] for m in tmeta])); S["truth_probe"] = -clf.decision_function(f32(mk["resp_last"]))
S["truth_probe(mean)"] = -clf.decision_function(f32(mk["resp_mean"]))
msk = ((cats == "lie") & (cond == "instructed")) | (cats == "neutral_correct")
clf = lr().fit(f32(ev["resp_mean"])[msk], ((cats == "lie") & (cond == "instructed"))[msk].astype(int)); S["instr_lie_probe"] = clf.decision_function(f32(mk["resp_mean"]))
msk = (cond == "neutral") & np.isin(lab, ["CORRECT", "INCORRECT"])
clf = lr().fit(f32(ev["resp_last"])[msk], (lab == "INCORRECT")[msk].astype(int)); S["correctness_probe"] = clf.decision_function(f32(mk["resp_last"]))
msk = (cond == "neutral") & np.isin(know, ["known", "unknown"])
clf = lr().fit(f32(ev["prompt_last"])[msk], (know == "unknown")[msk].astype(int)); S["pik_probe"] = clf.decision_function(f32(mk["prompt_last"]))
S["neg_logprob"] = -mk["lp_mean"].astype(float)

mc = np.array([m["cat"] for m in mmeta]); kind = np.array([m["kind"] for m in mmeta]); ids = np.array([m["id"] for m in mmeta])
P = kind == "pressure"; B = kind == "belief"
groups = {"lie (pressure)": P & (mc == "lie"), "honest (pressure)": P & (mc == "honest"),
          "false, no belief (pressure)": P & (mc == "false_no_belief"), "true, no belief (pressure)": P & (mc == "true_no_belief"),
          "belief correct, consistent (neutral)": B & (mc == "belief_correct_consistent"),
          "belief wrong, consistent (neutral)": B & (mc == "belief_wrong_consistent"),
          "belief wrong, inconsistent (neutral)": B & (mc == "belief_wrong_inconsistent")}
CON = [("lie vs honest (pressure)", "lie (pressure)", "honest (pressure)"),
       ("lie vs false-no-belief (pressure)", "lie (pressure)", "false, no belief (pressure)"),
       ("lie vs belief-wrong-inconsistent (neutral)", "lie (pressure)", "belief wrong, inconsistent (neutral)"),
       ("belief wrong-inconsistent vs correct-consistent (neutral)", "belief wrong, inconsistent (neutral)", "belief correct, consistent (neutral)"),
       ("belief wrong-consistent vs correct-consistent (neutral)", "belief wrong, consistent (neutral)", "belief correct, consistent (neutral)")]


def boot(s, pm, nm, n=500):
    m = pm | nm; y = pm[m]; ss = s[m]; v = []
    for _ in range(n):
        i = rng.randint(0, m.sum(), m.sum())
        if 0 < y[i].sum() < len(i):
            v.append(roc_auc_score(y[i], ss[i]))
    return roc_auc_score(y, ss), np.percentile(v, 2.5), np.percentile(v, 97.5)


res = {"counts": {k: int(v.sum()) for k, v in groups.items()},
       "pressure_cat_counts": dict(collections.Counter(mc[P])), "auc": {}}
for d, s in S.items():
    res["auc"][d] = {}
    for cname, a, b in CON:
        if groups[a].sum() >= 5 and groups[b].sum() >= 5:
            res["auc"][d][cname] = boot(s, groups[a], groups[b])
json.dump(res, open(f"{RESULTS}/analysis/mask.json", "w"), indent=1, default=float)
md = ["## MASK cross-check (known_facts + disinformation, Llama-3.1-8B-Instruct, layer 12)\n",
      "Counts: " + ", ".join(f"{k}: {v}" for k, v in res["counts"].items()),
      "\nPressure-statement categories: " + str(res["pressure_cat_counts"]) + "\n",
      "| detector | " + " | ".join(c[0] for c in CON) + " |", "|---|" + "---|" * len(CON)]
for d in S:
    md.append(f"| {d} | " + " | ".join((f"{res['auc'][d][c[0]][0]:.2f} [{res['auc'][d][c[0]][1]:.2f},{res['auc'][d][c[0]][2]:.2f}]"
                                         if c[0] in res["auc"][d] else "n/a") for c in CON) + " |")
open(f"{RESULTS}/analysis/mask_tables.md", "w").write("\n".join(md)); print("\n".join(md))
