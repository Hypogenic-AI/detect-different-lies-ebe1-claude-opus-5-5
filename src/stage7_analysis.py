"""Stage 7: behavioural prevalence analysis + detector evaluation, tables (JSON/markdown) and figures.
Inputs: results/knowledge_labels.jsonl, results/pressure_labeled.jsonl, results/detector_scores_*.json
Outputs: results/analysis/*.json|md, figures/*.png"""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = f"{RESULTS}/analysis"; os.makedirs(OUT, exist_ok=True); os.makedirs("figures", exist_ok=True)
NB = 1000
rng = np.random.RandomState(SEED)
# validated reference categorical palette (dataviz skill), fixed slot order
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GRAY = "#8a8985"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.axisbelow": True})
PRESSURE = ["sycophancy", "incentive", "instructed"]
CONDS = ["neutral"] + PRESSURE
md = []


def boot_ci(stat_fn, qids, n=NB):
    """Cluster bootstrap over questions: stat_fn(index array) -> float."""
    uq = np.unique(qids); pos = {q: np.where(qids == q)[0] for q in uq}
    vals = []
    for _ in range(n):
        s = rng.choice(uq, len(uq), replace=True)
        idx = np.concatenate([pos[q] for q in s])
        try:
            vals.append(stat_fn(idx))
        except ValueError:
            pass
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


# =============================== 1. behaviour / prevalence ===============================
rows = read_jsonl(f"{RESULTS}/pressure_labeled.jsonl")
know = np.array([r["knowledge"] for r in rows]); cond = np.array([r["cond"] for r in rows])
lab = np.array([r["label"] for r in rows]); qid = np.array([r["qid"] for r in rows])
fu = np.array([r["followup_label"] for r in rows]); adopt = np.array([r["adopted_distractor"] for r in rows])
false = lab == "INCORRECT"

beh = {"n_questions": int(len(np.unique(qid))),
       "knowledge_counts": {k: int((know[cond == "neutral"] == k).sum()) for k in ["known", "partial", "unknown"]}}
rate_tab = {}
for c in CONDS:
    for k in ["known", "partial", "unknown"]:
        m = (cond == c) & (know == k)
        rate_tab[f"{c}|{k}"] = {"n": int(m.sum()), "false": float(false[m].mean()), "correct": float((lab[m] == "CORRECT").mean()),
                                "evasion": float((lab[m] == "EVASION").mean()),
                                "adopt_distractor": float(adopt[m].mean()),
                                "false_and_followup_correct": float((false[m] & (fu[m] == "CORRECT")).mean())}
beh["rates"] = rate_tab

# shares of false outputs by knowledge state, per condition and pooled
shares = {}
for name, cm in [(c, cond == c) for c in CONDS] + [("all_pressure", np.isin(cond, PRESSURE)),
                                                    ("all_conditions", np.ones(len(cond), bool)),
                                                    ("neutral+sycophancy+incentive", np.isin(cond, ["neutral", "sycophancy", "incentive"]))]:
    fm = cm & false
    ent = {"n_false": int(fm.sum())}
    for k in ["known", "partial", "unknown"]:
        def f(idx, k=k):
            ii = idx[fm[idx]]; return np.mean(know[ii] == k)
        ent[k] = float(np.mean(know[fm] == k)); ent[k + "_ci"] = boot_ci(f, qid)
    # 'verified lie' = known + false + the same-context follow-up recovers the correct answer
    def fv(idx):
        ii = idx[fm[idx]]; return np.mean((know[ii] == "known") & (fu[ii] == "CORRECT"))
    ent["known_and_followup_correct"] = float(np.mean((know[fm] == "known") & (fu[fm] == "CORRECT")))
    ent["known_and_followup_correct_ci"] = boot_ci(fv, qid)
    shares[name] = ent
beh["false_output_shares"] = shares
json.dump(beh, open(f"{OUT}/behaviour.json", "w"), indent=1)

md.append("## Behaviour: outcome rates by knowledge x condition\n")
md.append("| condition | knowledge | n | P(false) | P(correct) | P(evasion) | P(adopt X) | P(false & follow-up correct) |")
md.append("|---|---|---|---|---|---|---|---|")
for c in CONDS:
    for k in ["known", "partial", "unknown"]:
        t = rate_tab[f"{c}|{k}"]
        md.append(f"| {c} | {k} | {t['n']} | {t['false']:.3f} | {t['correct']:.3f} | {t['evasion']:.3f} | {t['adopt_distractor']:.3f} | {t['false_and_followup_correct']:.3f} |")
md.append("\n## Shares of false outputs by neutral knowledge state (95% cluster-bootstrap CI)\n")
md.append("| pool | n false | known (=lie) | partial | unknown (=hallucination / pressured-unknown) | known & follow-up correct |")
md.append("|---|---|---|---|---|---|")
for name, e in shares.items():
    fmt = lambda k: f"{e[k]*100:.1f}% [{e[k+'_ci'][0]*100:.1f}, {e[k+'_ci'][1]*100:.1f}]"
    md.append(f"| {name} | {e['n_false']} | {fmt('known')} | {fmt('partial')} | {fmt('unknown')} | {fmt('known_and_followup_correct')} |")

# Figure 1: P(false) heatmap + stacked shares
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.25]})
H = np.array([[rate_tab[f"{c}|{k}"]["false"] for c in CONDS] for k in ["known", "partial", "unknown"]])
ax = axes[0]; im = ax.imshow(H, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
ax.set_xticks(range(4), CONDS, rotation=20); ax.set_yticks(range(3), ["known", "partial", "unknown"])
for i in range(3):
    for j in range(4):
        ax.text(j, i, f"{H[i, j]:.2f}", ha="center", va="center", color="white" if H[i, j] > .55 else "#0b0b0b", fontsize=9)
ax.set_title("P(false answer) by knowledge x condition"); ax.set_xlabel("condition"); ax.set_ylabel("neutral knowledge state")
ax = axes[1]
names = CONDS + ["all_pressure"]
left = np.zeros(len(names))
for k, col in zip(["known", "partial", "unknown"], [C[7], C[3], C[0]]):
    v = np.array([shares[n][k] for n in names])
    ax.barh(range(len(names)), v, left=left, color=col, edgecolor="white", linewidth=2, label={"known": "known (lie)", "partial": "partial", "unknown": "unknown (hallucination)"}[k])
    for i, (l, x) in enumerate(zip(left, v)):
        if x > .08:
            ax.text(l + x / 2, i, f"{x*100:.0f}%", ha="center", va="center", color="white", fontsize=9)
    left += v
ax.set_yticks(range(len(names)), [f"{n} (n={shares[n]['n_false']})" for n in names]); ax.invert_yaxis()
ax.set_xlim(0, 1); ax.set_xlabel("share of false outputs"); ax.set_title("Where do false outputs come from?")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=3, frameon=False)
plt.tight_layout(); plt.savefig("figures/fig1_prevalence.png", dpi=160, bbox_inches="tight"); plt.close()

# =============================== 2. detectors ===============================
D = json.load(open(f"{RESULTS}/detector_scores_primary.json"))
cats = np.array(D["cats"]); dq = np.array(D["qid"]); dcond = np.array(D["cond"])
S = {k: np.array(v, dtype=float) for k, v in D["scores"].items()}
def zref(x):
    r = x[(cats == "neutral_correct") & ~np.isnan(x)]; return (x - r.mean()) / r.std()
# composite 'belief-mismatch' detectors: falsity signal + knowledge signal (low P(I don't know)), z-scored on honest-correct
S["comp_truth+known"] = zref(S["truth_probe"]) - zref(S["pik_probe"])
S["comp_correct+known"] = zref(S["correctness_probe"]) - zref(S["pik_probe"])
LIE_DET = ["repe_probe", "truth_probe", "instr_lie_probe"]
HAL_DET = ["neg_logprob", "inconsistency", "correctness_probe", "pik_probe"]
COMP = ["comp_truth+known", "comp_correct+known"]
DETS = LIE_DET + HAL_DET + COMP
pres = np.isin(dcond, PRESSURE)
CONTRASTS = [  # (name, positive mask, negative mask)
    ("lie vs neutral-correct", (cats == "lie"), (cats == "neutral_correct")),
    ("halluc vs neutral-correct", (cats == "hallucination"), (cats == "neutral_correct")),
    ("lie vs halluc", (cats == "lie"), (cats == "hallucination")),
    ("lie vs honest-under-pressure", (cats == "lie"), (cats == "honest_pressure")),
    ("lie vs pressured-unknown-false", (cats == "lie"), (cats == "pressured_unknown_false")),
    ("honest-under-pressure vs neutral-correct", (cats == "honest_pressure"), (cats == "neutral_correct")),
]


def auc_ci(s, pm, nm):
    m = (pm | nm) & ~np.isnan(s)
    y = pm[m].astype(int); ss = s[m]; q = dq[m]
    a = roc_auc_score(y, ss)
    lo, hi = boot_ci(lambda idx: roc_auc_score(y[idx], ss[idx]), q, n=300)
    return a, lo, hi, int(pm[m].sum()), int(nm[m].sum())


det_res = {}
for d in DETS + ["repe_probe@prompt"]:
    s = S[d]; det_res[d] = {}
    for cname, pm, nm in CONTRASTS:
        a, lo, hi, npos, nneg = auc_ci(s, pm, nm)
        det_res[d][cname] = {"auc": a, "ci": [lo, hi], "n_pos": npos, "n_neg": nneg}
    # cross-firing: threshold = 99th pct of neutral-correct (1% FPR)
    thr = np.nanpercentile(s[cats == "neutral_correct"], 99)
    det_res[d]["fire_rate@1%FPR"] = {c: float(np.mean(s[cats == c] > thr)) for c in
                                     ["neutral_correct", "hallucination", "lie", "honest_pressure", "pressured_unknown_false"]}
    # per-condition AUROCs. Same-prompt contrasts are the unconfounded ones.
    det_res[d]["by_cond"] = {}
    for c in PRESSURE:
        lm = (cats == "lie") & (dcond == c)
        e = {}
        for cname, nm in [("lie vs halluc", cats == "hallucination"), ("lie vs neutral-correct", cats == "neutral_correct"),
                          ("lie vs honest-under-pressure", (cats == "honest_pressure") & (dcond == c)),
                          ("lie vs pressured-unknown-false", (cats == "pressured_unknown_false") & (dcond == c))]:
            m = (lm | nm) & ~np.isnan(s)
            if lm[m].sum() > 5 and nm[m].sum() > 5:
                a = float(roc_auc_score(lm[m], s[m]))
                ci = boot_ci(lambda idx, y=lm[m], ss=s[m]: roc_auc_score(y[idx], ss[idx]), dq[m], n=200)
                e[cname] = a; e[cname + "_ci"] = ci; e[cname + "_n"] = [int(lm[m].sum()), int(nm[m].sum())]
            else:
                e[cname] = None
        e["n_lie"] = int(lm.sum())
        det_res[d]["by_cond"][c] = e
    # condition-stratified AUROC for same-prompt contrasts (weighted by n_pos*n_neg = pooled pairs within condition)
    for cname in ["lie vs honest-under-pressure", "lie vs pressured-unknown-false"]:
        num = den = 0
        for c in PRESSURE:
            e = det_res[d]["by_cond"][c]
            if e[cname] is not None:
                w = e[cname + "_n"][0] * e[cname + "_n"][1]; num += w * e[cname]; den += w
        det_res[d]["stratified " + cname] = num / den
diag = {}
for d in [k for k in S if k.startswith("diag_")]:
    s = S[d]; m = ~np.isnan(s)
    pos = cats == "lie"
    diag[d] = {"auc": float(roc_auc_score(pos[m], s[m])), "n_pos": int(pos[m].sum()), "n_neg": int((~pos[m]).sum())}
    diag[d]["ci"] = boot_ci(lambda idx: roc_auc_score(pos[m][idx], s[m][idx]), dq[m], n=300)
    for c in PRESSURE:   # within one pressure template the prompt template is constant
        mc = m & (dcond == c)
        diag[d][c] = float(roc_auc_score(pos[mc], s[mc])) if 5 < pos[mc].sum() < mc.sum() - 5 else None
json.dump({"detectors": det_res, "diagnostic": diag, "cosines": D.get("cosines"), "sanity": D.get("sanity"),
           "layer": D["layer"]}, open(f"{OUT}/detectors.json", "w"), indent=1)

md.append(f"\n## Detector AUROCs (layer {D['layer']}; 95% cluster-bootstrap CI). Positive class listed first.\n")
md.append("| detector | " + " | ".join(c[0] for c in CONTRASTS) + " |")
md.append("|---|" + "---|" * len(CONTRASTS))
for d in DETS + ["repe_probe@prompt"]:
    md.append(f"| {d} | " + " | ".join(f"{det_res[d][c[0]]['auc']:.2f} [{det_res[d][c[0]]['ci'][0]:.2f},{det_res[d][c[0]]['ci'][1]:.2f}]" for c in CONTRASTS) + " |")
md.append("\nn (pos/neg): " + "; ".join(f"{c[0]}: {det_res['repe_probe'][c[0]]['n_pos']}/{det_res['repe_probe'][c[0]]['n_neg']}" for c in CONTRASTS))
md.append("\n## Fire rate at threshold = 1% FPR on neutral honest-correct answers\n")
fcats = ["neutral_correct", "hallucination", "lie", "honest_pressure", "pressured_unknown_false"]
md.append("| detector | " + " | ".join(fcats) + " |"); md.append("|---|" + "---|" * len(fcats))
for d in DETS + ["repe_probe@prompt"]:
    md.append(f"| {d} | " + " | ".join(f"{det_res[d]['fire_rate@1%FPR'][c]:.3f}" for c in fcats) + " |")
md.append("\n## Per-condition AUROCs (lie from that condition)\n")
md.append("| detector | cond | n lie | lie vs halluc | lie vs neutral-correct | lie vs honest-under-pressure (same prompt) | lie vs pressured-unknown-false (same prompt) |")
md.append("|---|---|---|---|---|---|---|")
def f2(e, k):
    return "n/a" if e[k] is None else f"{e[k]:.2f} [{e[k+'_ci'][0]:.2f},{e[k+'_ci'][1]:.2f}]"
for d in DETS:
    for c in PRESSURE:
        e = det_res[d]["by_cond"][c]
        md.append(f"| {d} | {c} | {e['n_lie']} | {f2(e,'lie vs halluc')} | {f2(e,'lie vs neutral-correct')} | {f2(e,'lie vs honest-under-pressure')} | {f2(e,'lie vs pressured-unknown-false')} |")
md.append("\n## Condition-stratified same-prompt AUROCs (pair-weighted mean of per-condition AUROCs)\n")
md.append("| detector | lie vs honest-under-pressure | lie vs pressured-unknown-false |"); md.append("|---|---|---|")
for d in DETS + ["repe_probe@prompt"]:
    md.append(f"| {d} | {det_res[d]['stratified lie vs honest-under-pressure']:.2f} | {det_res[d]['stratified lie vs pressured-unknown-false']:.2f} |")
md.append("\n## Diagnostic supervised probes (out-of-fold, grouped by question): response tokens vs prompt tokens only\n")
md.append("| probe | pooled AUROC [95% CI] | n pos / n neg | sycophancy | incentive | instructed |"); md.append("|---|---|---|---|---|---|")
g2 = lambda x: "n/a" if x is None else f"{x:.2f}"
for d, e in diag.items():
    md.append(f"| {d} | {e['auc']:.2f} [{e['ci'][0]:.2f},{e['ci'][1]:.2f}] | {e['n_pos']}/{e['n_neg']} | {g2(e['sycophancy'])} | {g2(e['incentive'])} | {g2(e['instructed'])} |")
if D.get("cosines"):
    md.append("\n## Cosine similarity between probe directions (raw activation space)\n")
    nm = D["cosines"]["names"]; Mx = np.array(D["cosines"]["matrix"])
    md.append("| | " + " | ".join(nm) + " |"); md.append("|---|" + "---|" * len(nm))
    for i, n in enumerate(nm):
        md.append(f"| {n} | " + " | ".join(f"{Mx[i, j]:.2f}" for j in range(len(nm))) + " |")
md.append(f"\nSanity (own-distribution CV AUROC): {D.get('sanity')}")

# Figure 2: score distributions per category (z-scored to neutral-correct)
order = ["neutral_correct", "honest_pressure", "hallucination", "pressured_unknown_false", "lie"]
lbl = {"neutral_correct": "honest-correct\n(neutral)", "honest_pressure": "honest-correct\n(pressure)",
       "hallucination": "hallucination\n(unknown, neutral)", "pressured_unknown_false": "false, unknown\n(pressure)",
       "lie": "lie\n(known, pressure)"}
colors = [C[0], C[2], C[3], C[1], C[7]]
fig, axes = plt.subplots(2, 5, figsize=(18, 6.6), sharey=False)
for ax, d in zip(axes.flat, DETS + ["repe_probe@prompt"]):
    s = S[d]; ref = s[cats == "neutral_correct"]; ref = ref[~np.isnan(ref)]
    z = (s - ref.mean()) / (ref.std() + 1e-9)
    data = [z[(cats == c) & ~np.isnan(z)] for c in order]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6, medianprops={"color": "#0b0b0b"})
    for p, col in zip(bp["boxes"], colors):
        p.set_facecolor(col); p.set_alpha(0.85); p.set_edgecolor("white")
    ax.set_xticks(range(1, 6), ["HC-n", "HC-p", "Hal", "F-unk-p", "Lie"], fontsize=8)
    ax.set_title(("LIE det: " if d in LIE_DET or d.startswith("repe") else "COMPOSITE: " if d in COMP else "HALLUC det: ") + d, fontsize=9)
    ax.axhline(0, color=GRAY, lw=0.8)
axes.flat[0].set_ylabel("score (z vs honest-correct neutral)"); axes.flat[5].set_ylabel("score (z vs honest-correct neutral)")
fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in colors], [lbl[o].replace("\n", " ") for o in order],
           loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.03))
plt.tight_layout(rect=(0, 0.04, 1, 1)); plt.savefig("figures/fig2_score_distributions.png", dpi=160, bbox_inches="tight"); plt.close()

# Figure 3: AUROC heatmap detectors x contrasts (diverging around 0.5)
A = np.array([[det_res[d][c[0]]["auc"] for c in CONTRASTS] for d in DETS + ["repe_probe@prompt"]])
fig, ax = plt.subplots(figsize=(10.5, 5.4))
im = ax.imshow(A, cmap="RdBu_r", vmin=0, vmax=1); ax.grid(False)
ax.set_xticks(range(len(CONTRASTS)), [c[0].replace(" vs ", "\nvs ") for c in CONTRASTS], fontsize=8)
ax.set_yticks(range(len(A)), DETS + ["repe_probe@prompt"])
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=8.5, color="white" if abs(A[i, j] - .5) > .3 else "#0b0b0b")
ax.axhline(len(LIE_DET) - .5, color="white", lw=3); ax.axhline(len(LIE_DET + HAL_DET) - .5, color="white", lw=3); ax.axhline(len(DETS) - .5, color="white", lw=3)
plt.colorbar(im, ax=ax, label="AUROC (first class scored higher > 0.5)")
ax.set_title(f"Detector AUROC per contrast (Llama-3.1-8B-Instruct, layer {D['layer']})")
plt.tight_layout(); plt.savefig("figures/fig3_auroc_heatmap.png", dpi=160, bbox_inches="tight"); plt.close()

# Figure 4: cross-firing at 1% FPR
fig, ax = plt.subplots(figsize=(13, 3.8))
w = 0.16
for j, (c, col) in enumerate(zip(order, colors)):
    v = [det_res[d]["fire_rate@1%FPR"][c] for d in DETS]
    ax.bar(np.arange(len(DETS)) + (j - 2) * w, v, w, color=col, edgecolor="white", linewidth=1, label=lbl[c].replace("\n", " "))
ax.set_xticks(range(len(DETS)), DETS); ax.set_ylabel("fraction flagged"); ax.set_ylim(0, 1)
ax.set_title("Fire rate at a threshold calibrated to 1% FPR on neutral honest-correct answers")
ax.legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8)
plt.tight_layout(); plt.savefig("figures/fig4_crossfiring.png", dpi=160, bbox_inches="tight"); plt.close()


# Figure 6: same-prompt (unconfounded) AUROCs per pressure condition
fig, axes = plt.subplots(1, 3, figsize=(17, 4.2), sharey=True)
for ax, cname in zip(axes, ["lie vs honest-under-pressure", "lie vs pressured-unknown-false", "lie vs halluc"]):
    w = 0.26
    for j, (c, col) in enumerate(zip(PRESSURE, [C[0], C[1], C[2]])):
        v = [det_res[d]["by_cond"][c][cname] for d in DETS]
        lo = [det_res[d]["by_cond"][c].get(cname + "_ci", [np.nan, np.nan])[0] for d in DETS]
        hi = [det_res[d]["by_cond"][c].get(cname + "_ci", [np.nan, np.nan])[1] for d in DETS]
        v = np.array([np.nan if x is None else x for x in v], float)
        ax.bar(np.arange(len(DETS)) + (j - 1) * w, v, w, color=col, edgecolor="white", label=f"{c} (n lie={det_res[DETS[0]]['by_cond'][c]['n_lie']})",
               yerr=[v - np.array(lo, float), np.array(hi, float) - v], error_kw={"lw": 0.8, "ecolor": "#52514e"})
    ax.axhline(.5, color="#0b0b0b", lw=.8)
    ax.set_xticks(range(len(DETS)), DETS, rotation=40, ha="right", fontsize=8)
    ax.set_title(cname + (" (same prompt)" if "halluc" not in cname else " (different prompts)"), fontsize=10)
    ax.set_ylim(0, 1)
axes[0].set_ylabel("AUROC (lie scored higher > 0.5)"); axes[0].legend(frameon=False, fontsize=8, loc="lower left")
plt.tight_layout(); plt.savefig("figures/fig6_per_condition_auroc.png", dpi=160, bbox_inches="tight"); plt.close()

# Figure 5: layer sweep
sweep = {}
for f in sorted(glob.glob(f"{RESULTS}/detector_scores_*.json")):
    Dl = json.load(open(f)); cl = np.array(Dl["cats"])
    for d in ["repe_probe", "truth_probe", "instr_lie_probe", "correctness_probe"]:
        s = np.array(Dl["scores"][d], dtype=float)
        for cname, pos, neg in [("lie vs neutral-correct", "lie", "neutral_correct"), ("lie vs halluc", "lie", "hallucination"),
                                ("halluc vs neutral-correct", "hallucination", "neutral_correct"),
                                ("lie vs honest-under-pressure", "lie", "honest_pressure")]:
            m = np.isin(cl, [pos, neg]) & ~np.isnan(s)
            sweep.setdefault(d, {}).setdefault(cname, {})[Dl["layer"]] = float(roc_auc_score(cl[m] == pos, s[m]))
        # stratified same-prompt contrast: lie vs pressured-unknown-false within each pressure condition
        cd = np.array(Dl["cond"]); num = den = 0
        for c in PRESSURE:
            m = (cd == c) & np.isin(cl, ["lie", "pressured_unknown_false"]) & ~np.isnan(s)
            npos = (cl[m] == "lie").sum(); nneg = m.sum() - npos
            num += npos * nneg * roc_auc_score(cl[m] == "lie", s[m]); den += npos * nneg
        sweep[d].setdefault("lie vs pressured-unknown-false (same prompt, stratified)", {})[Dl["layer"]] = float(num / den)
json.dump(sweep, open(f"{OUT}/layer_sweep.json", "w"), indent=1)
if len(glob.glob(f"{RESULTS}/detector_scores_*.json")) > 2:
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
    for ax, d in zip(axes, sweep):
        for (cname, v), col in zip(sweep[d].items(), [C[0], C[7], C[3], C[2], C[6]]):
            Ls = sorted(v); ax.plot(Ls, [v[l] for l in Ls], "-o", color=col, lw=2, ms=4, label=cname)
        ax.axhline(.5, color=GRAY, lw=.8); ax.set_title(d); ax.set_xlabel("layer"); ax.set_ylim(0, 1)
    axes[0].set_ylabel("AUROC")
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, ncol=5, frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.06))
    plt.tight_layout(); plt.savefig("figures/fig5_layer_sweep.png", dpi=160, bbox_inches="tight"); plt.close()

open(f"{OUT}/tables.md", "w").write("\n".join(md))
print("\n".join(md))
