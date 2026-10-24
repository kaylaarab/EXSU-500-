"""
Step 7: Sensitivity analyses on the cleaned cohort (CatBoost, fixed params).

  A. Reference: current pipeline (payer_code and medical_specialty dropped).
  B. Keep medical_specialty and payer_code as categories ("Unknown" = missing).
  C. First-encounter-per-patient evaluation:
       C1. trained on all encounters, tested on first encounters only
       C2. trained AND tested on first encounters only
  D. Patient-level (cluster) bootstrap 95% CIs, plus the paired A-vs-B
     difference. Resampling patients, not encounters, respects the fact
     that one patient's rows are correlated.

Splits are identical to the main pipeline (same patients, same seed).
Test set is scored once per model; hyperparameters are fixed from
05_catboost.py (depth=4, lr=0.1), so nothing is tuned on test.
"""
import importlib.util
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, average_precision_score

spec = importlib.util.spec_from_file_location("prep", "scripts/01_preprocess.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)

RAW = "data/diabetic_data.csv"
TARGET = "readmit_30d"
N_BOOT = 1000
rng = np.random.default_rng(42)


def build(keep_specialty_payer):
    # keep encounter_id so we can find each patient's first encounter
    drop = ["weight"] + ([] if keep_specialty_payer else ["payer_code", "medical_specialty"])
    df = prep.load_and_clean(RAW, drop_cols=drop)
    df["encounter_id"] = pd.read_csv(RAW, usecols=["encounter_id", "discharge_disposition_id"],
                                     keep_default_na=False).loc[df.index, "encounter_id"]
    first_id = df.groupby("patient_nbr")["encounter_id"].transform("min")
    df["is_first"] = (df["encounter_id"] == first_id).astype(int)
    tr, va, te = prep.patient_level_split(df)
    return tr, va, te


def fit_predict(tr, va, te):
    drop = [TARGET, "patient_nbr", "encounter_id", "is_first"]
    X = [d.drop(columns=drop) for d in (tr, va, te)]
    cats = X[0].select_dtypes(exclude="number").columns.tolist()
    for d in X:
        for c in cats:
            d[c] = d[c].astype(str)
    pools = [Pool(x, d[TARGET], cat_features=cats) for x, d in zip(X, (tr, va, te))]
    m = CatBoostClassifier(depth=4, learning_rate=0.1, iterations=1000, loss_function="Logloss",
                           eval_metric="AUC",
                           scale_pos_weight=(tr[TARGET] == 0).sum() / (tr[TARGET] == 1).sum(),
                           random_seed=42, verbose=False, early_stopping_rounds=50)
    m.fit(pools[0], eval_set=pools[1])
    return m.predict_proba(pools[2])[:, 1]


def patient_boot(y, p, groups, n=N_BOOT, extra=None):
    """Resample patients with replacement; return AUROC (and AUPRC) samples."""
    df = pd.DataFrame({"y": y, "p": p, "g": groups})
    if extra is not None:
        df["q"] = extra
    by = {g: d for g, d in df.groupby("g")}
    keys = np.array(list(by))
    out = []
    for _ in range(n):
        pick = rng.choice(keys, len(keys))
        d = pd.concat([by[k] for k in pick])
        if d.y.nunique() < 2:
            continue
        row = [roc_auc_score(d.y, d.p), average_precision_score(d.y, d.p)]
        if extra is not None:
            row.append(roc_auc_score(d.y, d.q))
            row.append(row[0] - row[2])
        out.append(row)
    return np.array(out)


def ci(a):
    lo, hi = np.percentile(a, [2.5, 97.5])
    return f"[{lo:.3f}, {hi:.3f}]"


def report(name, y, p, g):
    b = patient_boot(y.to_numpy(), p, g.to_numpy())
    print(f"{name:55s} n={len(y):6d}  AUROC={roc_auc_score(y, p):.3f} {ci(b[:, 0])}  "
          f"AUPRC={average_precision_score(y, p):.3f} {ci(b[:, 1])}")
    return dict(analysis=name, n_test=len(y), pos_rate=y.mean(),
                auroc=roc_auc_score(y, p), auroc_lo=np.percentile(b[:, 0], 2.5),
                auroc_hi=np.percentile(b[:, 0], 97.5), auprc=average_precision_score(y, p),
                auprc_lo=np.percentile(b[:, 1], 2.5), auprc_hi=np.percentile(b[:, 1], 97.5))


rows = []
trA, vaA, teA = build(False)
pA = fit_predict(trA, vaA, teA)
rows.append(report("A. Reference (specialty/payer dropped)", teA[TARGET], pA, teA["patient_nbr"]))

trB, vaB, teB = build(True)
pB = fit_predict(trB, vaB, teB)
assert (teA["patient_nbr"].values == teB["patient_nbr"].values).all(), "splits differ"
rows.append(report("B. Specialty + payer kept as categories", teB[TARGET], pB, teB["patient_nbr"]))

# D. paired difference B - A, patient-level bootstrap
d = patient_boot(teB[TARGET].to_numpy(), pB, teB["patient_nbr"].to_numpy(), extra=pA)
print(f"\nPaired AUROC difference B-A: {roc_auc_score(teB[TARGET], pB) - roc_auc_score(teA[TARGET], pA):+.4f} "
      f"95% CI {ci(d[:, 3])}  (CI containing 0 => no reliable difference)\n")

# C1. train on all, test on first encounters only
m = teA["is_first"] == 1
rows.append(report("C1. Reference model, first-encounter test rows only",
                   teA.loc[m, TARGET], pA[m.values], teA.loc[m, "patient_nbr"]))

# C2. train + test on first encounters only
f = lambda d: d[d["is_first"] == 1]
trC, vaC, teC = f(trA), f(vaA), f(teA)
pC = fit_predict(trC, vaC, teC)
rows.append(report("C2. Trained and tested on first encounters only",
                   teC[TARGET], pC, teC["patient_nbr"]))

# Encounter-level vs patient-level bootstrap width on the reference model
enc = []
y = teA[TARGET].to_numpy()
for _ in range(N_BOOT):
    i = rng.integers(0, len(y), len(y))
    if len(np.unique(y[i])) > 1:
        enc.append(roc_auc_score(y[i], pA[i]))
print(f"\nReference AUROC CI width: encounter-level {np.ptp(np.percentile(enc, [2.5, 97.5])):.4f} "
      f"vs patient-level {np.ptp(np.percentile(patient_boot(y, pA, teA['patient_nbr'].to_numpy())[:, 0], [2.5, 97.5])):.4f}")

pd.DataFrame(rows).to_csv("5_results/ablations/sensitivity_analysis.csv", index=False)
print("\nSaved 5_results/ablations/sensitivity_analysis.csv")
