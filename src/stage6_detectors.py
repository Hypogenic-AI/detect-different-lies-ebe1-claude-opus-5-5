"""Stage 6: fit the detector battery and score every evaluation transcript.

Lie detectors (higher = more 'deceptive'):
  repe_probe        Apollo-style LR probe (honest vs deceptive persona, RepE facts), mean over response tokens
  truth_probe       LR truth probe on true/false statements (GoT/TTPD csvs), last response token; score = P(false)
  instr_lie_probe   in-distribution probe trained on instructed lies vs neutral honest-correct answers
                    (the usual 'lie vs honest' training set), mean over response tokens, out-of-fold by question
Hallucination detectors (higher = more 'hallucination-like'):
  neg_logprob       - mean token log-prob of the response
  inconsistency     1 - fraction of 5 same-context samples that agree with the greedy answer (semantic-entropy proxy)
  correctness_probe SEP/LLMsKnow-style probe trained on NEUTRAL answers, incorrect vs correct, last response token, OOF
  pik_probe         P(I don't know) probe on the last prompt token, trained on neutral prompts (unknown vs known), OOF

All probes: StandardScaler + L2 logistic regression (C=0.1, i.e. lambda=10 as in Apollo) at a single layer.
Primary layer fixed a priori at hidden_states[12] (~37% depth; lie formation in layers 10-15 per 2509.03518);
a full layer sweep is saved as robustness.
Out-of-fold = 5-fold GroupKFold over question ids, so no question is scored by a probe that saw it.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold

LAYERS = list(range(4, 33, 2))
PRIMARY = 12
li = LAYERS.index(PRIMARY)
PRESSURE = ["sycophancy", "incentive", "instructed"]


USE_TORCH = "--torch" in sys.argv


class TorchLR:
    """GPU logistic regression with the same objective as sklearn's LogisticRegression(C) on standardised inputs:
    minimise C * sum(logloss) + 0.5 * ||w||^2 (full-batch LBFGS). Used for the layer sweep (sklearn is too slow)."""
    def __init__(self, C=0.1):
        self.C = C

    def fit(self, X, y):
        import torch
        X = torch.tensor(X, dtype=torch.float32, device="cuda"); y = torch.tensor(y, dtype=torch.float32, device="cuda")
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd
        w = torch.zeros(Z.shape[1], device="cuda", requires_grad=True); b = torch.zeros(1, device="cuda", requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=500, line_search_fn="strong_wolfe", tolerance_grad=1e-6)
        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(Z @ w + b, y, reduction="sum") * self.C + 0.5 * (w * w).sum()
            loss.backward(); return loss
        opt.step(closure)
        self.w, self.b = w.detach(), b.detach()
        return self

    def decision_function(self, X):
        import torch
        Z = (torch.tensor(X, dtype=torch.float32, device="cuda") - self.mu) / self.sd
        return (Z @ self.w + self.b).cpu().numpy()


def lr():
    if USE_TORCH:
        return TorchLR(0.1)
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))


def load(name):
    d = np.load(f"{RESULTS}/acts/{name}.npz")
    return {k: d[k] for k in d.files}, read_jsonl(f"{RESULTS}/acts/{name}_meta.jsonl")


def category(m):
    k, c, l = m["knowledge"], m["cond"], m["label"]
    if l == "EVASION":
        return "evasion"
    if c == "neutral":
        return {("known", "CORRECT"): "neutral_correct", ("unknown", "INCORRECT"): "hallucination"}.get((k, l), f"neutral_{k}_{l.lower()}")
    return {("known", "INCORRECT"): "lie", ("known", "CORRECT"): "honest_pressure",
            ("unknown", "INCORRECT"): "pressured_unknown_false", ("unknown", "CORRECT"): "pressured_unknown_correct"
            }.get((k, l), f"pressure_{k}_{l.lower()}")


def oof_probe(X, y, train_mask, groups, apply_mask=None, n_splits=5):
    """Fit on rows with train_mask (labels y), score rows in apply_mask out-of-fold by question group."""
    apply_mask = np.ones(len(X), bool) if apply_mask is None else apply_mask
    scores = np.full(len(X), np.nan)
    ug = np.unique(groups)
    rng = np.random.RandomState(SEED); rng.shuffle(ug)
    fold_of = {g: i % n_splits for i, g in enumerate(ug)}
    f = np.array([fold_of[g] for g in groups])
    for k in range(n_splits):
        tr = train_mask & (f != k)
        te = apply_mask & (f == k)
        clf = lr().fit(X[tr], y[tr])
        scores[te] = clf.decision_function(X[te])
    return scores


def main(layer_idx=li, tag="primary"):
    ev, meta = load("eval")
    repe, rmeta = load("repe")
    tr, tmeta = load("truth")
    groups = np.array([m["qid"] for m in meta])
    cond = np.array([m["cond"] for m in meta]); know = np.array([m["knowledge"] for m in meta])
    lab = np.array([m["label"] for m in meta])
    cats = np.array([category(m) for m in meta])
    S = {}

    # --- lie detectors -------------------------------------------------------------------------------
    yr = np.array([m["deceptive"] for m in rmeta])
    repe_clf = lr().fit(repe["resp_mean"][:, layer_idx].astype(np.float32), yr)
    S["repe_probe"] = repe_clf.decision_function(ev["resp_mean"][:, layer_idx].astype(np.float32))
    S["repe_probe@prompt"] = repe_clf.decision_function(ev["prompt_last"][:, layer_idx].astype(np.float32))

    yt = np.array([m["label"] for m in tmeta])
    truth_clf = lr().fit(tr["resp_last"][:, layer_idx].astype(np.float32), yt)
    S["truth_probe"] = -truth_clf.decision_function(ev["resp_last"][:, layer_idx].astype(np.float32))  # P(false)

    Xm = ev["resp_mean"][:, layer_idx].astype(np.float32)
    Xl = ev["resp_last"][:, layer_idx].astype(np.float32)
    Xp = ev["prompt_last"][:, layer_idx].astype(np.float32)
    instr_train = ((cats == "lie") & (cond == "instructed")) | (cats == "neutral_correct")
    y_instr = ((cats == "lie") & (cond == "instructed")).astype(int)
    S["instr_lie_probe"] = oof_probe(Xm, y_instr, instr_train, groups)

    # --- hallucination detectors --------------------------------------------------------------------
    S["neg_logprob"] = -ev["lp_mean"].astype(float)
    S["inconsistency"] = 1 - np.array([m["consistency"] for m in meta])
    corr_train = (cond == "neutral") & np.isin(lab, ["CORRECT", "INCORRECT"])
    S["correctness_probe"] = oof_probe(Xl, (lab == "INCORRECT").astype(int), corr_train, groups)
    pik_train = (cond == "neutral") & np.isin(know, ["known", "unknown"])
    S["pik_probe"] = oof_probe(Xp, (know == "unknown").astype(int), pik_train, groups)

    # --- diagnostic probes (separability / prompt confound) ------------------------------------------
    pres = np.isin(cond, PRESSURE)
    # lie vs honest answer under the SAME pressure prompts (known questions): response tokens vs prompt tokens only
    m_lh = pres & np.isin(cats, ["lie", "honest_pressure"])
    S["diag_lie_vs_honest@resp"] = oof_probe(Xm, (cats == "lie").astype(int), m_lh, groups, m_lh)
    S["diag_lie_vs_honest@prompt"] = oof_probe(Xp, (cats == "lie").astype(int), m_lh, groups, m_lh)
    # lie vs hallucination, directly supervised (confounded by prompt: lies have pressure prompts)
    m_lhal = np.isin(cats, ["lie", "hallucination"])
    S["diag_lie_vs_halluc@resp"] = oof_probe(Xm, (cats == "lie").astype(int), m_lhal, groups, m_lhal)
    S["diag_lie_vs_halluc@prompt"] = oof_probe(Xp, (cats == "lie").astype(int), m_lhal, groups, m_lhal)
    # knowledge under pressure: lie vs pressured-unknown false (same prompts, both false)
    m_lpu = pres & np.isin(cats, ["lie", "pressured_unknown_false"])
    S["diag_lie_vs_pufalse@resp"] = oof_probe(Xm, (cats == "lie").astype(int), m_lpu, groups, m_lpu)
    S["diag_lie_vs_pufalse@prompt"] = oof_probe(Xp, (cats == "lie").astype(int), m_lpu, groups, m_lpu)

    out = {"scores": {k: v.tolist() for k, v in S.items()}, "cats": cats.tolist(), "cond": cond.tolist(),
           "know": know.tolist(), "qid": groups.tolist(),
           "followup": [m["followup_label"] for m in meta], "adopted": [m["adopted_distractor"] for m in meta],
           "layer": LAYERS[layer_idx]}

    # probe directions in raw activation space (coef / scale) for cosine analysis
    if tag == "primary":
        def direction(clf):
            sc, l = clf[0], clf[1]; return (l.coef_[0] / sc.scale_)
        dirs = {"repe_probe": direction(repe_clf), "truth_probe(false)": -direction(truth_clf)}
        for name, X, y, msk in [("instr_lie_probe", Xm, y_instr, instr_train),
                                ("correctness_probe(incorrect)", Xl, (lab == "INCORRECT").astype(int), corr_train),
                                ("pik_probe(unknown)", Xp, (know == "unknown").astype(int), pik_train),
                                ("lie_vs_halluc", Xm, (cats == "lie").astype(int), m_lhal),
                                ("lie_vs_honest_pressure", Xm, (cats == "lie").astype(int), m_lh)]:
            dirs[name] = direction(lr().fit(X[msk], y[msk]))
        names = list(dirs); M = np.stack([dirs[n] / np.linalg.norm(dirs[n]) for n in names])
        out["cosines"] = {"names": names, "matrix": (M @ M.T).tolist()}
        # held-out accuracy of the off-the-shelf probes on their own training distributions (sanity)
        from sklearn.model_selection import cross_val_score
        out["sanity"] = {
            "repe_cv_auc": float(cross_val_score(lr(), repe["resp_mean"][:, layer_idx].astype(np.float32), yr,
                                                 groups=np.array([m["fact_id"] for m in rmeta]), cv=GroupKFold(5), scoring="roc_auc").mean()),
            "truth_cv_auc": float(cross_val_score(lr(), tr["resp_last"][:, layer_idx].astype(np.float32), yt, cv=5, scoring="roc_auc").mean())}
    json.dump(out, open(f"{RESULTS}/detector_scores_{tag}.json", "w"))
    return out


if __name__ == "__main__":
    # default: primary layer with sklearn. --sweep-only --torch: other layers with the GPU LR (same objective).
    if "--sweep-only" not in sys.argv:
        o = main()
        print("sanity", o.get("sanity"))
    else:
        for i, L in enumerate(LAYERS):
            if L == PRIMARY:
                continue
            main(i, f"L{L}"); print("layer", L, flush=True)
