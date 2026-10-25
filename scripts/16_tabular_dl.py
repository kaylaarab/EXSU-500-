"""
Step 16: Stronger deep-learning baselines for tabular data.

  RealMLP   (Holzmueller et al., NeurIPS 2024; via pytabkit)  - an MLP with tuned defaults
  TabM      (Gorishniy et al.; via pytabkit)                  - parameter-efficient ensemble of MLPs
  FT-Transformer (Gorishniy et al., 2021; rtdl_revisiting_models) - transformer over feature tokens

Same patient-level splits as scripts 02-06 (run 01_preprocess.py first). Validation is used for
early stopping and for choosing the decision threshold; the test set is scored once.
N_SEEDS seeds per model (shortened for CPU time: RealMLP 64 epochs; TabM 50 epochs with 8 ensemble members and batch size 1024; FT-Transformer 15 epochs; early stopping); the seed-averaged probabilities are the model's prediction.
Reference: CatBoost (depth 4, lr 0.1), trained here so we can run a paired patient-level
bootstrap on the AUROC difference. Sensitivity is always shown with accuracy.
SMOKE=1: tiny run to check the code only (nothing is saved).
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

SMOKE = bool(os.environ.get("SMOKE"))
N_SEEDS = 1 if SMOKE else int(os.environ.get("SEEDS", 3))
TARGET = "readmit_30d"
torch.set_num_threads(max(1, (os.cpu_count() or 4)))
tr, va, te = [pd.read_csv(f"data/{s}.csv", keep_default_na=False) for s in ("train", "val", "test")]
if SMOKE:
    tr = tr.sample(4000, random_state=0)
y = {k: d[TARGET].to_numpy().astype(int) for k, d in (("tr", tr), ("va", va), ("te", te))}
groups = te["patient_nbr"].to_numpy()
X = {k: d.drop(columns=[TARGET, "patient_nbr"]).reset_index(drop=True) for k, d in (("tr", tr), ("va", va), ("te", te))}
cats = X["tr"].select_dtypes(exclude="number").columns.tolist()
nums = [c for c in X["tr"].columns if c not in cats]

# --- frames for pytabkit (raw columns, categoricals as pandas 'category') -------------------
allx = pd.concat([X[k] for k in ("tr", "va", "te")], keys=["tr", "va", "te"])
for c in cats:
    allx[c] = allx[c].astype("category")
XP = {k: allx.loc[k].reset_index(drop=True) for k in ("tr", "va", "te")}

# --- tensors for the FT-Transformer -----------------------------------------------------------
SKEWED = [c for c in ["number_outpatient", "number_emergency", "number_inpatient", "total_prior_visits", "num_med_changes"] if c in nums]
vocab = {c: {v: i + 1 for i, v in enumerate(sorted(X["tr"][c].unique()))} for c in cats}
card = [len(vocab[c]) + 1 for c in cats]
M = X["tr"][nums].astype(float).copy(); M[SKEWED] = np.log1p(M[SKEWED]); mu, sd = M.mean(), M.std().replace(0, 1)
def numeric(k):
    A = X[k][nums].astype(float).copy(); A[SKEWED] = np.log1p(A[SKEWED]); return torch.tensor(((A - mu) / sd).to_numpy(np.float32))
def catcodes(k):
    return torch.tensor(np.stack([X[k][c].map(vocab[c]).fillna(0).astype(int).to_numpy() for c in cats], 1))
T = {k: (numeric(k), catcodes(k)) for k in ("tr", "va", "te")}


def ft_run(seed):
    import rtdl_revisiting_models as r
    torch.manual_seed(seed); np.random.seed(seed)
    m = r.FTTransformer(n_cont_features=len(nums), cat_cardinalities=card, d_out=1, n_blocks=2, d_block=64,
                        attention_n_heads=8, attention_dropout=0.1, ffn_d_hidden=None, ffn_d_hidden_multiplier=1.33,
                        ffn_dropout=0.1, residual_dropout=0.0)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-5)
    pw = torch.tensor((y["tr"] == 0).sum() / (y["tr"] == 1).sum(), dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    ytr = torch.tensor(y["tr"], dtype=torch.float32)
    n, best, state, bad = len(ytr), -1, None, 0
    def pred(k):
        m.eval()
        with torch.no_grad():
            return torch.sigmoid(torch.cat([m(T[k][0][i:i + 4096], T[k][1][i:i + 4096]).squeeze(1)
                                            for i in range(0, len(T[k][0]), 4096)])).numpy()
    for ep in range(2 if SMOKE else 15):                          # reduced from 30 for CPU time (machine restarts)
        m.train(); perm = torch.randperm(n)
        for i in range(0, n, 512):
            b = perm[i:i + 512]
            if len(b) < 2: continue
            opt.zero_grad(); lossf(m(T["tr"][0][b], T["tr"][1][b]).squeeze(1), ytr[b]).backward(); opt.step()
        s = average_precision_score(y["va"], pred("va"))
        if s > best: best, bad, state = s, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 4: break
    m.load_state_dict(state)
    return pred("va"), pred("te")


def pytab_run(cls, seed):
    kw = dict(device="cpu", random_state=seed, verbosity=0, n_threads=os.cpu_count())
    if SMOKE: kw["n_epochs"] = 2
    elif cls.__name__.startswith("RealMLP"): kw["n_epochs"] = 64      # default is 256; reduced for CPU time
    else: kw.update(n_epochs=50, tabm_k=8, batch_size=1024)         # TabM: 8 ensemble members (default 32) and larger batches, for CPU time
    mdl = cls(**kw)
    mdl.fit(XP["tr"], y["tr"], XP["va"], y["va"], cat_col_names=cats)
    return mdl.predict_proba(XP["va"])[:, 1], mdl.predict_proba(XP["te"])[:, 1]


from pytabkit import RealMLP_TD_Classifier, TabM_D_Classifier
MODELS = {"RealMLP": lambda s: pytab_run(RealMLP_TD_Classifier, s),
          "TabM": lambda s: pytab_run(TabM_D_Classifier, s),
          "FT-Transformer": ft_run}

# --- CatBoost reference -------------------------------------------------------------------------
Xc = {k: X[k].copy() for k in X}
for k in Xc:
    for c in cats: Xc[k][c] = Xc[k][c].astype(str)
pools = {k: Pool(Xc[k], y[k], cat_features=cats) for k in Xc}
cb = CatBoostClassifier(depth=4, learning_rate=0.1, iterations=20 if SMOKE else 1000, loss_function="Logloss",
                        eval_metric="AUC", scale_pos_weight=(y["tr"] == 0).sum() / (y["tr"] == 1).sum(), random_seed=42,
                        verbose=False, early_stopping_rounds=50).fit(pools["tr"], eval_set=pools["va"])
P = {"CatBoost (reference)": (cb.predict_proba(pools["va"])[:, 1], cb.predict_proba(pools["te"])[:, 1], [])}
for name, fn in MODELS.items():
    vs, ts, per = [], [], []
    for s in range(N_SEEDS):
        f = Path(f"data/cache16{'_smoke' if SMOKE else ''}_{name}_{s}.npz")
        if f.exists(): z = np.load(f); v, t = z['v'], z['t']
        else: v, t = fn(s); np.savez(f, v=v, t=t)
        vs.append(v); ts.append(t); per.append(roc_auc_score(y["te"], t))
        print(f"  {name} seed {s}: test AUROC {per[-1]:.4f}", flush=True)
    P[name] = (np.mean(vs, 0), np.mean(ts, 0), per)

yt = y["te"]
idx = pd.Series(np.arange(len(yt))).groupby(groups).apply(lambda s: s.to_numpy()).tolist()
rng = np.random.default_rng(1)
resamples = [np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))]) for _ in range(1000)]
base = P["CatBoost (reference)"][1]
print(f"\nalways-'no' accuracy {1 - yt.mean():.3f}")
rows = []
for name, (pv, pt, per) in P.items():
    fpr, tpr, th = roc_curve(y["va"], pv); t = th[np.argmax(tpr - fpr)]; yp = pt >= t
    d = [roc_auc_score(yt[r], pt[r]) - roc_auc_score(yt[r], base[r]) for r in resamples if yt[r].min() != yt[r].max()]
    lo, hi = np.percentile(d, [2.5, 97.5])
    rows.append(dict(model=name, auroc_seed_ensemble=roc_auc_score(yt, pt), auroc_per_seed=";".join(f"{a:.4f}" for a in per),
                     auprc=average_precision_score(yt, pt), accuracy=np.mean(yp == yt), sensitivity=np.mean(yp[yt == 1]),
                     specificity=np.mean(~yp[yt == 0]), threshold=t,
                     diff_vs_catboost=roc_auc_score(yt, pt) - roc_auc_score(yt, base), diff_lo=lo, diff_hi=hi))
    r = rows[-1]
    print(f"{name:22s} AUROC={r['auroc_seed_ensemble']:.3f} AUPRC={r['auprc']:.3f} | vs CatBoost {r['diff_vs_catboost']:+.4f} "
          f"[{lo:+.4f},{hi:+.4f}] | Acc={r['accuracy']:.3f} Sens={r['sensitivity']:.3f} Spec={r['specificity']:.3f}")
if not SMOKE:
    pd.DataFrame(rows).to_csv("5_results/ablations/tabular_dl.csv", index=False)
    print("\nSaved 5_results/ablations/tabular_dl.csv")
