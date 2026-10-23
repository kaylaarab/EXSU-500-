"""
Step 11: How much does it inflate AUROC to apply SMOTE BEFORE cross-validation?

Motivation: arXiv:2002.11215 (EmbPred30) reports 0.974 AUROC on this dataset using
SMOTE-balanced data and 6-fold CV, without stating that SMOTE was confined to
training folds (see literature/ for the review). We cannot reproduce that
deep network, but we CAN measure the leakage mechanism on our own cohort with a
fixed gradient-boosting model. Only the evaluation protocol changes between rows.

  A0  row-level 6-fold CV, real data only         (patient leakage only)
  A1  SMOTE-NC on ALL data, then row-level 6-fold CV; scored on every held-out row
  A2  same as A1 but scored on REAL held-out rows only
  C   correct: patient-level split, SMOTE-NC on TRAIN only, scored on real test rows
  D   correct: patient-level split, class weights, scored on real test rows

If A1/A2 are far above C/D, that gap is evaluation leakage, not model skill.
Hyperparameters are fixed; nothing is tuned on any evaluation set.
"""
import importlib.util
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder

spec = importlib.util.spec_from_file_location("prep", "scripts/01_preprocess.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)

RAW, TARGET = "data/diabetic_data.csv", "readmit_30d"

df = prep.load_and_clean(RAW)
cats = df.drop(columns=[TARGET, "patient_nbr"]).select_dtypes(exclude="number").columns.tolist()
nums = [c for c in df.columns if c not in cats + [TARGET, "patient_nbr"]]
order = cats + nums
X = df[order].copy()
X[cats] = OrdinalEncoder().fit_transform(X[cats])          # integer codes, no label information
X = X.astype(float).to_numpy()
y = df[TARGET].to_numpy()
cat_idx = list(range(len(cats)))
mask = np.array([True] * len(cats) + [False] * len(nums))
n_orig = len(y)
print(f"cohort: {n_orig:,} encounters, {y.mean():.3f} positive\n")


def model():
    return HGB(max_depth=6, learning_rate=0.1, max_iter=300, categorical_features=mask, random_state=0)


def smote(Xs, ys):
    return SMOTENC(categorical_features=cat_idx, sampling_strategy=1.0, random_state=42).fit_resample(Xs, ys)


def cv_scores(Xa, ya, real_flag, train_is_balanced=False):
    """Row-level 6-fold CV. Returns (all-rows AUROC, real-rows AUROC) averaged over folds."""
    all_, real = [], []
    for tr, te in StratifiedKFold(6, shuffle=True, random_state=0).split(Xa, ya):
        p = model().fit(Xa[tr], ya[tr]).predict_proba(Xa[te])[:, 1]
        all_.append(roc_auc_score(ya[te], p))
        r = real_flag[te]
        real.append(roc_auc_score(ya[te][r], p[r]))
    return np.mean(all_), np.std(all_), np.mean(real), np.std(real)


rows = []


def record(name, auroc, sd, auprc=None, note=""):
    rows.append(dict(protocol=name, auroc=auroc, auroc_sd=sd, auprc=auprc, note=note))
    print(f"{name:62s} AUROC={auroc:.3f}" + (f" ±{sd:.3f}" if sd == sd else "")
          + (f"  AUPRC={auprc:.3f}" if auprc is not None else "") + f"  {note}")


# A0: real data, row-level CV (no SMOTE)
a_all, a_sd, _, _ = cv_scores(X, y, np.ones(n_orig, bool))
record("A0 row-level 6-fold CV, real data, no SMOTE", a_all, a_sd, note="patient leakage only")

# A1 / A2: SMOTE on everything first, then row-level CV
Xs, ys = smote(X, y)
is_real = np.arange(len(ys)) < n_orig                    # SMOTE appends synthetic rows after originals
print(f"SMOTE-NC on all data: {n_orig:,} -> {len(ys):,} rows; "
      f"{(~is_real).sum():,} synthetic ({(~is_real).sum() / (ys == 1).sum():.0%} of positives)")
a1, a1sd, a2, a2sd = cv_scores(Xs, ys, is_real)
record("A1 SMOTE before 6-fold CV, scored on ALL held-out rows", a1, a1sd, note="'EmbPred30-style' protocol")
record("A2 SMOTE before 6-fold CV, scored on REAL held-out rows", a2, a2sd, note="")

# C / D: correct protocol on a patient-level split
tr, va, te = prep.patient_level_split(df)
pos = df.index.get_indexer
itr, ite = pos(tr.index), pos(te.index)
Xtr, ytr, Xte, yte = X[itr], y[itr], X[ite], y[ite]
Xtr_s, ytr_s = smote(Xtr, ytr)
p = model().fit(Xtr_s, ytr_s).predict_proba(Xte)[:, 1]
record("C patient split, SMOTE on TRAIN only, real test rows", roc_auc_score(yte, p), float("nan"),
       average_precision_score(yte, p), "correct")
w = np.where(ytr == 1, (ytr == 0).sum() / (ytr == 1).sum(), 1.0)
p = HGB(max_depth=6, learning_rate=0.1, max_iter=300, categorical_features=mask,
        random_state=0).fit(Xtr, ytr, sample_weight=w).predict_proba(Xte)[:, 1]
record("D patient split, class weights, real test rows", roc_auc_score(yte, p), float("nan"),
       average_precision_score(yte, p), "correct, our method")

pd.DataFrame(rows).to_csv("results/smote_leakage_demo.csv", index=False)
print("\nSaved results/smote_leakage_demo.csv")
