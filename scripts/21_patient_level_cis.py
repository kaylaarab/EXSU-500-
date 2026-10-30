"""
Step 21: patient-level 95% confidence intervals for every model in model_comparison.csv,
and paired differences against CatBoost.

Retrains each headline model with the settings used in scripts 02, 04, 05 and 06 (same split,
same seeds), stores the test-set probabilities in data/preds_21.npz (git-ignored) so the bootstrap
can be rerun without retraining, then resamples PATIENTS (not encounters) with replacement.
Reports AUROC, AUPRC, accuracy, sensitivity and specificity (threshold 0.5, as in the main table).
Output: 5_results/main_models/model_comparison_patient_ci.csv and model_paired_differences.csv.
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostClassifier, Pool
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

TARGET, ID = "readmit_30d", "patient_nbr"
CACHE = Path("data/preds_21.npz")
N_BOOT = 1000
OUT = Path("5_results/main_models")

train = pd.read_csv("data/train.csv", keep_default_na=False)
val = pd.read_csv("data/val.csv", keep_default_na=False)
test = pd.read_csv("data/test.csv", keep_default_na=False)
y_tr, y_va, y_te = train[TARGET], val[TARGET], test[TARGET]
X_tr, X_va, X_te = (d.drop(columns=[TARGET, ID]) for d in (train, val, test))
cat_cols = X_tr.select_dtypes(exclude="number").columns.tolist()
num_cols = [c for c in X_tr.columns if c not in cat_cols]
pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()


def prep():
    return ColumnTransformer([("num", StandardScaler(), num_cols),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])


def fit_all():
    P = {"Baseline (majority-class)": np.zeros(len(y_te))}
    lr = Pipeline([("prep", prep()), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0))]).fit(X_tr, y_tr)
    P["Logistic Regression"] = lr.predict_proba(X_te)[:, 1]
    lr_va = lr.predict_proba(X_va)[:, 1]
    xg = Pipeline([("prep", prep()), ("clf", XGBClassifier(
        max_depth=5, n_estimators=400, learning_rate=0.03, eval_metric="aucpr", subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=pos_weight, random_state=42, n_jobs=-1))]).fit(X_tr, y_tr)
    P["XGBoost (tuned on val)"] = xg.predict_proba(X_te)[:, 1]
    xg_va = xg.predict_proba(X_va)[:, 1]
    # CatBoost: script 05 selects depth/lr on validation AUPRC with early stopping; script 06 records the winner
    cb_tr, cb_va, cb_te = (d.astype({c: str for c in cat_cols}) for d in (X_tr, X_va, X_te))
    cb = CatBoostClassifier(depth=6, learning_rate=0.03, iterations=791, loss_function="Logloss", eval_metric="AUC",
                            scale_pos_weight=pos_weight, random_seed=42, verbose=False)
    cb.fit(Pool(cb_tr, y_tr, cat_features=cat_cols))
    P["CatBoost (engineered features)"] = cb.predict_proba(Pool(cb_te, cat_features=cat_cols))[:, 1]
    cb_val = cb.predict_proba(Pool(cb_va, cat_features=cat_cols))[:, 1]
    # Ensemble: script 06 grid over weights, step 0.1, maximising validation AUPRC
    pv = {"lr": lr_va, "xg": xg_va, "cb": cb_val}
    pt = {"lr": P["Logistic Regression"], "xg": P["XGBoost (tuned on val)"], "cb": P["CatBoost (engineered features)"]}
    best, bw = -1, None
    for w1, w2 in itertools.product([i / 10 for i in range(11)], repeat=2):
        w3 = round(1 - w1 - w2, 2)
        if w3 < 0:
            continue
        w = {"lr": w1, "xg": w2, "cb": w3}
        a = average_precision_score(y_va, sum(w[k] * pv[k] for k in w))
        if a > best:
            best, bw = a, w
    P["Ensemble (CatBoost+XGBoost+LogReg blend)"] = sum(bw[k] * pt[k] for k in bw)
    print("ensemble weights", bw)
    P["MLP (deep learning)"] = fit_mlp()
    return P


def fit_mlp():
    torch.manual_seed(42); np.random.seed(42)
    pp = prep()
    A = [pp.fit_transform(X_tr)] + [pp.transform(X) for X in (X_va, X_te)]
    A = [(a.toarray() if hasattr(a, "toarray") else a).astype(np.float32) for a in A]
    Xt, Xv, Xe = (torch.tensor(a) for a in A)
    yt = torch.tensor(y_tr.to_numpy(), dtype=torch.float32)
    m = nn.Sequential(nn.Linear(Xt.shape[1], 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3),
                      nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.3), nn.Linear(64, 1))
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight)))
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    best, state, bad = -1, None, 0
    for ep in range(100):
        m.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 512):
            idx = perm[i:i + 512]; opt.zero_grad(); crit(m(Xt[idx]).squeeze(-1), yt[idx]).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            a = average_precision_score(y_va, torch.sigmoid(m(Xv).squeeze(-1)).numpy())
        if a > best:
            best, bad = a, 0
            state = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 10:
                break
    m.load_state_dict(state); m.eval()
    with torch.no_grad():
        return torch.sigmoid(m(Xe).squeeze(-1)).numpy()


if CACHE.exists():
    P = dict(np.load(CACHE, allow_pickle=False))
    print("loaded", CACHE)
else:
    P = fit_all()
    np.savez(CACHE, **P)

y = y_te.to_numpy()
codes, uniq = pd.factorize(test[ID])
members = [np.where(codes == i)[0] for i in range(len(uniq))]
rng = np.random.default_rng(42)
names = list(P)
M = {k: [] for k in names}  # per model: rows of [auroc, auprc, acc, sens, spec]


def metrics(yy, pp):
    pred = pp >= 0.5
    tp, tn = ((pred == 1) & (yy == 1)).sum(), ((pred == 0) & (yy == 0)).sum()
    return [roc_auc_score(yy, pp), average_precision_score(yy, pp), (tp + tn) / len(yy),
            tp / max((yy == 1).sum(), 1), tn / max((yy == 0).sum(), 1)]


for b in range(N_BOOT):
    pick = rng.integers(0, len(members), len(members))
    idx = np.concatenate([members[i] for i in pick])
    if len(np.unique(y[idx])) < 2:
        continue
    for k in names:
        M[k].append(metrics(y[idx], P[k][idx]))
M = {k: np.array(v) for k, v in M.items()}
cols = ["auroc", "auprc", "accuracy", "sensitivity", "specificity"]
rows = []
for k in names:
    pt = metrics(y, P[k]); r = {"model": k, "n_test": len(y), "n_patients": len(uniq), "n_boot": len(M[k])}
    for j, c in enumerate(cols):
        lo, hi = np.percentile(M[k][:, j], [2.5, 97.5])
        r.update({c: pt[j], c + "_lo": lo, c + "_hi": hi})
    rows.append(r)
ci = pd.DataFrame(rows); ci.to_csv(OUT / "model_comparison_patient_ci.csv", index=False)
ref = "CatBoost (engineered features)"
d = []
for k in names:
    if k == ref:
        continue
    r = {"model": k, "versus": ref}
    for j, c in enumerate(cols[:2] + cols[3:4]):
        jj = cols.index(c); diff = M[k][:, jj] - M[ref][:, jj]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        r.update({c + "_diff": metrics(y, P[k])[jj] - metrics(y, P[ref])[jj], c + "_diff_lo": lo, c + "_diff_hi": hi})
    r["auroc_reliably_different"] = bool(r["auroc_diff_lo"] > 0 or r["auroc_diff_hi"] < 0)
    d.append(r)
pd.DataFrame(d).to_csv(OUT / "model_paired_differences.csv", index=False)
pd.set_option("display.width", 250)
print(ci[["model", "auroc", "auroc_lo", "auroc_hi", "auprc", "accuracy", "sensitivity", "sensitivity_lo", "sensitivity_hi"]].round(3).to_string(index=False))
print(pd.DataFrame(d)[["model", "auroc_diff", "auroc_diff_lo", "auroc_diff_hi", "auroc_reliably_different"]].round(3).to_string(index=False))
