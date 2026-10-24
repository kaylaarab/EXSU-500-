"""
Step 13: Does a different definition of "readmitted" give better results?

  T1  <30 days vs everything else            (our target; clinically actionable window)
  T2  any readmission (<30 or >30) vs none
  T3  <30 days vs never readmitted           (the >30 group is dropped)
Same cleaned cohort, same patient-level split, same fixed model (HistGradientBoosting,
class-weighted), test set scored once. Prevalence differs a lot between targets, so
AUROC is the only metric that is comparable across rows; accuracy and AUPRC are judged
against their own baselines (printed). Sensitivity is always shown with accuracy.
"""
import importlib.util
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import average_precision_score, roc_auc_score

spec = importlib.util.spec_from_file_location("prep", "scripts/01_preprocess.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)
RAW = "data/diabetic_data.csv"

df = prep.load_and_clean(RAW)
lab = pd.read_csv(RAW, usecols=["readmitted"], keep_default_na=False).loc[df.index, "readmitted"]
tr, va, te = prep.patient_level_split(df)


def prepare(d):
    X = d.drop(columns=["readmit_30d", "patient_nbr"])
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype(str)
    return X


Xall = pd.concat([prepare(d) for d in (tr, va, te)], keys=[0, 1, 2])
for c in Xall.columns:
    if not pd.api.types.is_numeric_dtype(Xall[c]):
        Xall[c] = Xall[c].astype("category")
Xs = [Xall.loc[i] for i in range(3)]
lab_s = [lab.loc[d.index] for d in (tr, va, te)]

TARGETS = {
    "T1 <30d vs rest (ours)": (lambda l: (l == "<30").astype(int), lambda l: np.ones(len(l), bool)),
    "T2 any readmission vs none": (lambda l: (l != "NO").astype(int), lambda l: np.ones(len(l), bool)),
    "T3 <30d vs never (>30 dropped)": (lambda l: (l == "<30").astype(int), lambda l: (l != ">30").to_numpy()),
}
rows = []
for name, (f, keep) in TARGETS.items():
    m_tr, m_te = keep(lab_s[0]), keep(lab_s[2])
    ytr, yte = f(lab_s[0])[m_tr].to_numpy(), f(lab_s[2])[m_te].to_numpy()
    model = HGB(max_depth=6, learning_rate=0.05, max_iter=300, categorical_features="from_dtype",
                class_weight="balanced", random_state=0).fit(Xs[0][m_tr], ytr)
    p = model.predict_proba(Xs[2][m_te])[:, 1]
    yp = p >= 0.5
    r = dict(target=name, n_test=len(yte), prevalence=yte.mean(), auroc=roc_auc_score(yte, p),
             auprc=average_precision_score(yte, p), auprc_baseline=yte.mean(),
             accuracy=np.mean(yp == yte), always_no_accuracy=1 - yte.mean(),
             sensitivity=np.mean(yp[yte == 1]), specificity=np.mean(~yp[yte == 0]))
    rows.append(r)
    print(f"{name:34s} n={r['n_test']:6d} prev={r['prevalence']:.3f} AUROC={r['auroc']:.3f} "
          f"AUPRC={r['auprc']:.3f} (baseline {r['auprc_baseline']:.3f}) Acc={r['accuracy']:.3f} "
          f"(always-no {r['always_no_accuracy']:.3f}) Sens={r['sensitivity']:.3f} Spec={r['specificity']:.3f}")
pd.DataFrame(rows).to_csv("5_results/ablations/target_definitions.csv", index=False)
print("Saved 5_results/ablations/target_definitions.csv")
