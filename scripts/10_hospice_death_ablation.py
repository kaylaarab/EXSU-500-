"""
Step 10: Does excluding death/hospice discharges matter, and is excluding hospice justified?

Three cohorts, one fixed patient-level split:
  all          no exclusion (101,766 encounters)
  no_death     exclude expired only (codes 11, 19, 20, 21)
  no_death_hospice  exclude expired + hospice (11, 13, 14, 19, 20, 21)   <- our main cohort

The split is made ONCE on the full cohort and each cohort is then filtered
within train/val/test, so all three share the same patients and the test sets
are nested. Every model is scored on the common test set (our main cohort), so
the comparison is like-for-like; it is also scored on its own test set to show
how much an easy-negative group inflates the headline number.

Model: CatBoost, fixed params from 05_catboost.py (depth 4, lr 0.1). No tuning
on test. Patient-level bootstrap 95% CIs.
"""
import importlib.util
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, roc_auc_score

spec = importlib.util.spec_from_file_location("prep", "scripts/01_preprocess.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)

RAW, TARGET = "data/diabetic_data.csv", "readmit_30d"
DEATH, HOSPICE = [11, 19, 20, 21], [13, 14]
rng = np.random.default_rng(42)

df = prep.load_and_clean(RAW, dead_codes=[])           # keep everything, split once
code = df["discharge_disposition_id"].str.split("_").str[-1].astype(int)
tr, va, te = prep.patient_level_split(df)
code_of = lambda d: code.loc[d.index]

COHORTS = {
    "all": lambda d: d,
    "no_death": lambda d: d[~code_of(d).isin(DEATH)],
    "no_death_hospice": lambda d: d[~code_of(d).isin(DEATH + HOSPICE)],
}
common = COHORTS["no_death_hospice"](te)
print(f"common test set: {len(common):,} encounters\n")


def fit(train, val):
    drop = [TARGET, "patient_nbr"]
    X = [d.drop(columns=drop) for d in (train, val)]
    cats = X[0].select_dtypes(exclude="number").columns.tolist()
    pools = [Pool(x, d[TARGET], cat_features=cats) for x, d in zip(X, (train, val))]
    m = CatBoostClassifier(depth=4, learning_rate=0.1, iterations=1000, loss_function="Logloss",
                           eval_metric="AUC", random_seed=42, verbose=False,
                           scale_pos_weight=(train[TARGET] == 0).sum() / (train[TARGET] == 1).sum(),
                           early_stopping_rounds=50)
    m.fit(pools[0], eval_set=pools[1])
    return m, cats


def predict(m, cats, d):
    return m.predict_proba(Pool(d.drop(columns=[TARGET, "patient_nbr"]), cat_features=cats))[:, 1]


def boot_ci(y, p, g, n=1000):
    idx = pd.Series(np.arange(len(y))).groupby(g).apply(lambda s: s.to_numpy())
    arrs = list(idx.values)
    out = []
    for _ in range(n):
        pick = rng.integers(0, len(arrs), len(arrs))
        r = np.concatenate([arrs[i] for i in pick])
        if y[r].min() != y[r].max():
            out.append(roc_auc_score(y[r], p[r]))
    return np.percentile(out, [2.5, 97.5])


rows, preds = [], {}
for name, f in COHORTS.items():
    t, v, own = f(tr), f(va), f(te)
    m, cats = fit(t, v)
    p_common = predict(m, cats, common)
    p_own = predict(m, cats, own)
    preds[name] = p_common
    yc, yo = common[TARGET].to_numpy(), own[TARGET].to_numpy()
    lo, hi = boot_ci(yc, p_common, common["patient_nbr"].to_numpy())
    rows.append(dict(cohort=name, n_train=len(t), n_test_own=len(own),
                     pos_rate_own=yo.mean(), auroc_own_test=roc_auc_score(yo, p_own),
                     auprc_own_test=average_precision_score(yo, p_own),
                     auroc_common_test=roc_auc_score(yc, p_common), auroc_common_lo=lo,
                     auroc_common_hi=hi, auprc_common_test=average_precision_score(yc, p_common)))
    r = rows[-1]
    print(f"{name:18s} train={r['n_train']:6d}  own test: n={r['n_test_own']:6d} pos={r['pos_rate_own']:.3f} "
          f"AUROC={r['auroc_own_test']:.3f} | common test AUROC={r['auroc_common_test']:.3f} "
          f"[{lo:.3f}, {hi:.3f}] AUPRC={r['auprc_common_test']:.3f}")
    if name == "all":
        # how well does the all-cohort model rank the excluded group itself?
        ex = te[code_of(te).isin(DEATH + HOSPICE)]
        pe = predict(m, cats, ex)
        for label, codes in [("death only", DEATH), ("hospice only", HOSPICE)]:
            sub = ex[code_of(ex).isin(codes)]
            ps = pe[ex.index.get_indexer(sub.index)]
            ys = sub[TARGET].to_numpy()
            auc = roc_auc_score(ys, ps) if 0 < ys.sum() < len(ys) else float("nan")
            print(f"   excluded group ({label}): n={len(sub)} readmitted={int(ys.sum())} "
                  f"({ys.mean():.3f}) mean predicted risk={ps.mean():.3f} AUROC within group={auc:.3f}")
            rows[-1][f"excluded_{label.split()[0]}_n"] = len(sub)
            rows[-1][f"excluded_{label.split()[0]}_pos"] = int(ys.sum())

# paired differences on the common test set
yc, g = common[TARGET].to_numpy(), common["patient_nbr"].to_numpy()
idx = pd.Series(np.arange(len(yc))).groupby(g).apply(lambda s: s.to_numpy())
arrs = list(idx.values)
for a, b in [("no_death", "no_death_hospice"), ("all", "no_death_hospice")]:
    d = []
    for _ in range(1000):
        r = np.concatenate([arrs[i] for i in rng.integers(0, len(arrs), len(arrs))])
        if yc[r].min() != yc[r].max():
            d.append(roc_auc_score(yc[r], preds[a][r]) - roc_auc_score(yc[r], preds[b][r]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"paired AUROC on common test, {a} minus {b}: "
          f"{roc_auc_score(yc, preds[a]) - roc_auc_score(yc, preds[b]):+.4f} 95% CI [{lo:+.4f}, {hi:+.4f}]")
pd.DataFrame(rows).to_csv("5_results/ablations/hospice_death_ablation.csv", index=False)
print("\nSaved 5_results/ablations/hospice_death_ablation.csv")
