"""
Step 14: Can the "readmitted later (>30 days)" information help predict 30-day readmission?

All methods are scored on the SAME target (readmitted within 30 days), on the same
patient-level test set, with the same fixed model (HistGradientBoosting, class-weighted).

  A  binary <30d vs rest                       (our current model)
  B  three-class model (<30 / >30 / never), score = P(<30)
  C  two-stage: P(any readmission) x P(<30 | readmitted)
  D  stacking: out-of-fold P(any readmission) added as a feature to the binary model
     (out-of-fold, grouped by patient, so no label leakage into the feature)

Threshold-free: AUROC and AUPRC (paired patient-level bootstrap vs A).
Operating point: a threshold chosen on VALIDATION (maximising sensitivity+specificity)
is applied to the test set for every method, so accuracy and sensitivity are comparable.
Always-"no" accuracy is printed for reference.

Two scenarios: all encounters (default) and first encounter per patient only (FIRST_ONLY=1).
"""
import importlib.util
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold

spec = importlib.util.spec_from_file_location("prep", "scripts/01_preprocess.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)
RAW = "data/diabetic_data.csv"

FIRST_ONLY = bool(os.environ.get("FIRST_ONLY"))
df = prep.load_and_clean(RAW)
if FIRST_ONLY:                      # scenario 2: one row per patient (the earliest encounter)
    enc = pd.read_csv(RAW, usecols=["encounter_id"], low_memory=False).loc[df.index, "encounter_id"]
    df = df[enc.groupby(df["patient_nbr"]).transform("min") == enc]
    print(f"scenario: FIRST ENCOUNTER ONLY ({len(df):,} patients)")
else:
    print(f"scenario: ALL ENCOUNTERS ({len(df):,})")
lab = pd.read_csv(RAW, usecols=["readmitted"], keep_default_na=False).loc[df.index, "readmitted"]
tr, va, te = prep.patient_level_split(df)


def feats(d):
    X = d.drop(columns=["readmit_30d", "patient_nbr"])
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype(str)
    return X


allx = pd.concat([feats(d) for d in (tr, va, te)], keys=[0, 1, 2])
for c in allx.columns:
    if not pd.api.types.is_numeric_dtype(allx[c]):
        allx[c] = allx[c].astype("category")
X = [allx.loc[i].reset_index(drop=True) for i in range(3)]
L = [lab.loc[d.index].reset_index(drop=True) for d in (tr, va, te)]
Y30 = [(l == "<30").astype(int).to_numpy() for l in L]
YANY = [(l != "NO").astype(int).to_numpy() for l in L]
GROUPS = [d["patient_nbr"].reset_index(drop=True).to_numpy() for d in (tr, va, te)]


def hgb(Xa, ya):
    return HGB(max_depth=6, learning_rate=0.05, max_iter=300, categorical_features="from_dtype",
               class_weight="balanced", random_state=0).fit(Xa, ya)


# A: binary baseline
mA = hgb(X[0], Y30[0])
S = {"A binary <30d (current)": [mA.predict_proba(X[i])[:, 1] for i in (1, 2)]}

# B: three-class
cls = L[0].map({"<30": 0, ">30": 1, "NO": 2}).to_numpy()
mB = hgb(X[0], cls)
S["B three-class, P(<30)"] = [mB.predict_proba(X[i])[:, list(mB.classes_).index(0)] for i in (1, 2)]

# C: two-stage
m1 = hgb(X[0], YANY[0])
back = (L[0] != "NO").to_numpy()
m2 = hgb(X[0][back], Y30[0][back])
S["C two-stage P(any) x P(<30|any)"] = [m1.predict_proba(X[i])[:, 1] * m2.predict_proba(X[i])[:, 1] for i in (1, 2)]

# D: stacking with out-of-fold P(any)
oof = np.zeros(len(X[0]))
for a, b in GroupKFold(5).split(X[0], YANY[0], GROUPS[0]):
    oof[b] = hgb(X[0].iloc[a], YANY[0][a]).predict_proba(X[0].iloc[b])[:, 1]
Xs = [x.assign(p_any=p) for x, p in zip(X, (oof, m1.predict_proba(X[1])[:, 1], m1.predict_proba(X[2])[:, 1]))]
mD = hgb(Xs[0], Y30[0])
S["D stacking: + out-of-fold P(any)"] = [mD.predict_proba(Xs[i])[:, 1] for i in (1, 2)]

yv, yt, g = Y30[1], Y30[2], GROUPS[2]
print(f"always-'no' accuracy on test: {1 - yt.mean():.3f}; test positive rate {yt.mean():.3f}\n")
idx = pd.Series(np.arange(len(yt))).groupby(g).apply(lambda s: s.to_numpy()).tolist()
rng = np.random.default_rng(1)
resamples = [np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))]) for _ in range(1000)]
base = S["A binary <30d (current)"][1]
rows = []
for name, (pv, pt) in S.items():
    fpr, tpr, th = roc_curve(yv, pv)
    t = th[np.argmax(tpr - fpr)]                                # threshold chosen on validation only
    yp = pt >= t
    d = [roc_auc_score(yt[r], pt[r]) - roc_auc_score(yt[r], base[r]) for r in resamples if yt[r].min() != yt[r].max()]
    lo, hi = np.percentile(d, [2.5, 97.5])
    rows.append(dict(method=name, auroc=roc_auc_score(yt, pt), auprc=average_precision_score(yt, pt),
                     auroc_diff_vs_A=roc_auc_score(yt, pt) - roc_auc_score(yt, base), diff_lo=lo, diff_hi=hi,
                     threshold=t, accuracy=np.mean(yp == yt), sensitivity=np.mean(yp[yt == 1]),
                     specificity=np.mean(~yp[yt == 0])))
    r = rows[-1]
    print(f"{name:34s} AUROC={r['auroc']:.3f} AUPRC={r['auprc']:.3f} | vs A: {r['auroc_diff_vs_A']:+.4f} "
          f"[{lo:+.4f}, {hi:+.4f}] | Acc={r['accuracy']:.3f} Sens={r['sensitivity']:.3f} Spec={r['specificity']:.3f}")
out = "results/combined_targets_first_encounter.csv" if FIRST_ONLY else "results/combined_targets.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print(f"Saved {out}")
