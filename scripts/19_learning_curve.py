"""
Step 19: Learning curve, a test of the "information ceiling" claim.

Question: if more patients were available, would the model improve? Train on growing random subsets
of TRAINING PATIENTS (all rows of each sampled patient), with fixed hyperparameters (CatBoost depth 4,
lr 0.1, early stopping on the full validation set), and score on the same untouched test set.
If test AUROC flattens well before 100% of the training data, more data of the same kind will not help:
the limit is the information in the variables, not the sample size.

  CatBoost: 5 random subsets for fractions up to 40%, 3 for 60% and 80%, 1 for 100%.
  Logistic regression (one-hot, class-weighted): same fractions, for contrast.
  A saturating curve  AUROC(n) = a - b * n^(-c)  is fitted to the CatBoost means to estimate the asymptote a
  and what doubling the data would give. Extrapolation is indicative only.
Checkpointed to data/cache19.json so a machine restart does not lose finished fits.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.optimize import curve_fit
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET, OUT = "readmit_30d", Path("5_results/evaluation")
CACHE = Path("data/cache19.json")
cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
tr, va, te = [pd.read_csv(f"data/{s}.csv", keep_default_na=False) for s in ("train", "val", "test")]
feats = [c for c in tr.columns if c not in (TARGET, "patient_nbr")]
cats = tr[feats].select_dtypes(exclude="number").columns.tolist()
nums = [c for c in feats if c not in cats]
patients = tr["patient_nbr"].unique()
FRACS = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.0]
REPS = {0.05: 5, 0.10: 5, 0.20: 5, 0.40: 5, 0.60: 3, 0.80: 3, 1.0: 1}
Xva, yva, Xte, yte = va[feats], va[TARGET].to_numpy(), te[feats], te[TARGET].to_numpy()
pool_va, pool_te = Pool(Xva, yva, cat_features=cats), Pool(Xte, yte, cat_features=cats)


def subset(frac, rep):
    if frac == 1.0: return tr
    rng = np.random.default_rng(1000 * rep + int(frac * 100))
    keep = rng.choice(patients, int(round(frac * len(patients))), replace=False)
    return tr[tr["patient_nbr"].isin(keep)]


def fit_catboost(d):
    y = d[TARGET].to_numpy(); pool = Pool(d[feats], y, cat_features=cats)
    m = CatBoostClassifier(depth=4, learning_rate=0.1, iterations=1000, loss_function="Logloss", eval_metric="AUC",
                           scale_pos_weight=(y == 0).sum() / (y == 1).sum(), random_seed=42, verbose=False,
                           early_stopping_rounds=50).fit(pool, eval_set=pool_va)
    return dict(n_rows=len(d), n_patients=int(d.patient_nbr.nunique()), auroc=roc_auc_score(yte, m.predict_proba(pool_te)[:, 1]),
                auprc=average_precision_score(yte, m.predict_proba(pool_te)[:, 1]),
                train_auroc=roc_auc_score(y, m.predict_proba(pool)[:, 1]), iters=int(m.get_best_iteration()))


def fit_logreg(d):
    y = d[TARGET].to_numpy()
    pipe = Pipeline([("prep", ColumnTransformer([("n", StandardScaler(), nums), ("c", OneHotEncoder(handle_unknown="ignore"), cats)])),
                     ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]).fit(d[feats], y)
    p = pipe.predict_proba(Xte)[:, 1]
    return dict(n_rows=len(d), n_patients=int(d.patient_nbr.nunique()), auroc=roc_auc_score(yte, p), auprc=average_precision_score(yte, p),
                train_auroc=roc_auc_score(y, pipe.predict_proba(d[feats])[:, 1]), iters=0)


rows = []
for model, fn in (("CatBoost", fit_catboost), ("Logistic regression", fit_logreg)):
    for frac in FRACS:
        for rep in range(REPS[frac]):
            key = f"{model}|{frac}|{rep}"
            if key not in cache:
                cache[key] = fn(subset(frac, rep)); CACHE.write_text(json.dumps(cache))
            r = cache[key]; rows.append(dict(model=model, fraction=frac, rep=rep, **r))
            print(f"{model:20s} {frac:4.0%} rep{rep}: n={r['n_rows']:6d} test AUROC={r['auroc']:.4f} train AUROC={r['train_auroc']:.3f}", flush=True)
df = pd.DataFrame(rows); df.to_csv(OUT / "learning_curve.csv", index=False)

summ = df.groupby(["model", "fraction"]).agg(n_rows=("n_rows", "mean"), auroc=("auroc", "mean"), auroc_sd=("auroc", "std"),
                                              auprc=("auprc", "mean"), train_auroc=("train_auroc", "mean")).reset_index()
summ.round(4).to_csv(OUT / "learning_curve_summary.csv", index=False)
print("\n", summ.round(4).to_string(index=False))

cb = summ[summ.model == "CatBoost"]
f = lambda n, a, b, c: a - b * (n / 1e4) ** (-c)
try:
    popt, _ = curve_fit(f, cb.n_rows, cb.auroc, p0=[0.69, 0.05, 0.5], bounds=([0.6, 0, 0.05], [0.8, 1, 3]), maxfev=20000)
    a, b, c = popt; nfull = cb.n_rows.max()
    fit = dict(asymptote=a, auroc_at_full=f(nfull, *popt), auroc_at_2x=f(2 * nfull, *popt), auroc_at_5x=f(5 * nfull, *popt), b=b, c=c)
except Exception as e:
    fit = dict(error=str(e)); popt = None
print("\nsaturating fit:", {k: round(v, 4) if isinstance(v, float) else v for k, v in fit.items()})
pd.DataFrame([fit]).round(5).to_csv(OUT / "learning_curve_fit.csv", index=False)

fig, ax = plt.subplots(figsize=(6.4, 4.4))
for model, col in (("CatBoost", "#1f77b4"), ("Logistic regression", "#2ca02c")):
    s = summ[summ.model == model]
    ax.errorbar(s.n_rows, s.auroc, yerr=s.auroc_sd.fillna(0), fmt="o-", color=col, capsize=3, label=f"{model}: test")
    ax.plot(s.n_rows, s.train_auroc, "--", color=col, alpha=.5, label=f"{model}: train (in-sample)")
if popt is not None:
    xs = np.linspace(cb.n_rows.min(), 3 * cb.n_rows.max(), 200); ax.plot(xs, f(xs, *popt), ":", color="#d62728", label=f"fitted curve, limit about {a:.3f}")
ax.set_xscale("log"); ax.set_xlabel("Training stays (log scale)"); ax.set_ylabel("AUROC on the held-out test set"); ax.legend(fontsize=7)
ax.set_title("Learning curve: does more data help?"); plt.tight_layout(); plt.savefig(OUT / "learning_curve.png", dpi=150)
print("saved figure and tables")
