"""
Step 11: How much does applying SMOTE BEFORE cross-validation inflate AUROC?

Motivation: EmbPred30 (arXiv:2002.11215) reports 0.974 AUROC / 95.2% accuracy on this
dataset. Reading the paper (checked 2026-10-23): it keeps ONE encounter per patient and
excludes death/hospice, so repeated-patient leakage is not the explanation. Its
preprocessing list ends with a SMOTE step ("Data Synthesis", 1:1 ratio) BEFORE
the 6-fold cross-validation, and the text never says SMOTE was confined to
training folds. We cannot reproduce their deep network; instead we test the
mechanism on the same kind of cohort with a fixed gradient-boosting model.
Only the evaluation protocol changes between rows. All use the same 6 folds.

  A  SMOTE-NC on all data, then 6-fold CV; scored on ALL held-out rows (real + synthetic)
  B  same folds as A; scored on the REAL held-out rows only
  C  SMOTE-NC inside each training fold only; scored on real held-out rows  (correct)
  D  class weights, no SMOTE; scored on real held-out rows                 (our method)

If A/B are far above C/D, the gap is evaluation leakage, not model skill.
Cohort: first encounter per patient (as in EmbPred30), death/hospice excluded,
one row per patient so row-level folds are also patient-level folds.
Hyperparameters fixed; nothing is tuned on any evaluation fold.
"""
import importlib.util
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score


def thr(yt, p):
    """accuracy, sensitivity, specificity at threshold 0.5 (accuracy is always shown WITH sensitivity)"""
    yp = p >= 0.5
    return (np.mean(yp == yt), np.mean(yp[yt == 1]), np.mean(~yp[yt == 0]))

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder

spec = importlib.util.spec_from_file_location("prep", "scripts/01_preprocess.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)

RAW, TARGET = "data/diabetic_data.csv", "readmit_30d"

df = prep.load_and_clean(RAW)
enc_id = pd.read_csv(RAW, usecols=["encounter_id"], low_memory=False).loc[df.index, "encounter_id"]
df = df[enc_id.groupby(df["patient_nbr"]).transform("min") == enc_id]       # first encounter only
cats = df.drop(columns=[TARGET, "patient_nbr"]).select_dtypes(exclude="number").columns.tolist()
nums = [c for c in df.columns if c not in cats + [TARGET, "patient_nbr"]]
X = df[cats + nums].copy()
X[cats] = OrdinalEncoder().fit_transform(X[cats])         # integer codes; uses no label information
X = X.astype(float).to_numpy()
y = df[TARGET].to_numpy()
cat_idx = list(range(len(cats)))
mask = np.array([True] * len(cats) + [False] * len(nums))
n = len(y)
print(f"cohort: {n:,} patients (first encounter), {y.mean():.3f} positive\n")


def model():
    return HGB(max_depth=6, learning_rate=0.1, max_iter=300, categorical_features=mask, random_state=0)


def smote(Xs, ys):
    return SMOTENC(categorical_features=cat_idx, sampling_strategy=1.0, random_state=42).fit_resample(Xs, ys)


folds = list(StratifiedKFold(6, shuffle=True, random_state=0).split(X, y))

# A / B: SMOTE first, then CV over the balanced dataset (the suspected protocol)
Xs, ys = smote(X, y)
is_real = np.arange(len(ys)) < n                          # SMOTE appends synthetic rows after the originals
print(f"SMOTE-NC on all data: {n:,} -> {len(ys):,} rows; {(~is_real).sum():,} synthetic "
      f"({(~is_real).sum() / (ys == 1).sum():.0%} of the positive class)\n")
A, B, Am, Bm = [], [], [], []
for tr, te in StratifiedKFold(6, shuffle=True, random_state=0).split(Xs, ys):
    p = model().fit(Xs[tr], ys[tr]).predict_proba(Xs[te])[:, 1]
    A.append(roc_auc_score(ys[te], p))
    Am.append(thr(ys[te], p))
    r = is_real[te]
    B.append(roc_auc_score(ys[te][r], p[r])); Bm.append(thr(ys[te][r], p[r]))

# C / D: correct protocols on the original folds (real held-out rows)
C, Cp, D, Dp, Cm, Dm = [], [], [], [], [], []
for tr, te in folds:
    Xt, yt = smote(X[tr], y[tr])
    p = model().fit(Xt, yt).predict_proba(X[te])[:, 1]
    C.append(roc_auc_score(y[te], p)); Cp.append(average_precision_score(y[te], p)); Cm.append(thr(y[te], p))
    w = np.where(y[tr] == 1, (y[tr] == 0).sum() / (y[tr] == 1).sum(), 1.0)
    p = model().fit(X[tr], y[tr], sample_weight=w).predict_proba(X[te])[:, 1]
    D.append(roc_auc_score(y[te], p)); Dp.append(average_precision_score(y[te], p)); Dm.append(thr(y[te], p))

rows = [
    ("A SMOTE before CV, scored on all held-out rows (real + synthetic)", A, None, Am),
    ("B SMOTE before CV, scored on real held-out rows only", B, None, Bm),
    ("C SMOTE inside training folds only, real held-out rows (correct)", C, Cp, Cm),
    ("D class weights, no SMOTE, real held-out rows (our method)", D, Dp, Dm),
]
out = []
for name, a, ap, m in rows:
    acc, sens, spec = np.mean(m, 0)
    print(f"{name:68s} AUROC={np.mean(a):.3f} ±{np.std(a):.3f}" + (f"  AUPRC={np.mean(ap):.3f}" if ap else "")
          + f"  Acc={acc:.3f}  Sens={sens:.3f}  Spec={spec:.3f}")
    out.append(dict(protocol=name, auroc=np.mean(a), auroc_sd=np.std(a), auprc=np.mean(ap) if ap else None,
                    accuracy=acc, sensitivity=sens, specificity=spec))
print(f"\nalways-'no' accuracy on the real cohort: {1 - y.mean():.3f}")
pd.DataFrame(out).to_csv("results/smote_leakage_demo.csv", index=False)
print("\nSaved results/smote_leakage_demo.csv")
