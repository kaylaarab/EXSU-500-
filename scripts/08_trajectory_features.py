"""
Step 8: Do within-dataset patient-history features help? (HistGradientBoosting)

The raw file links a patient's stays via patient_nbr. Ordering stays by
encounter_id (assumed chronological -- there are no dates, so this is an
assumption we cannot verify) gives features that are legitimately known at
discharge of the CURRENT stay:

  seq_num           how many earlier stays of this patient are in the data
  prev_label        outcome label of the previous stay: '<30' / '>30' / none.
                    '<30' means the current stay is itself a 30-day readmission.
  prev_disposition  where the patient went after the previous stay
  prev_los          length of the previous stay

Features deliberately NOT used because they peek at the future:
  n_encounters, is_last_encounter, has_later_encounter
  (69% of '<30' stays have a later record by construction, so these leak the label).

Same patient-level splits as the main pipeline; test scored once; patient-level
bootstrap for the paired difference.
"""
import importlib.util
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import roc_auc_score, average_precision_score

spec = importlib.util.spec_from_file_location("prep", "scripts/01_preprocess.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)

RAW, TARGET = "data/diabetic_data.csv", "readmit_30d"
rng = np.random.default_rng(42)

raw = pd.read_csv(RAW, keep_default_na=False, na_values=["?"], low_memory=False,
                  usecols=["encounter_id", "readmitted", "discharge_disposition_id", "time_in_hospital"])
df = prep.load_and_clean(RAW)
df = df.join(raw[["encounter_id", "readmitted"]].rename(columns={"readmitted": "label_raw"}))
# cleaned ID columns were turned into strings; keep raw copies for history
df["_disp"] = raw.loc[df.index, "discharge_disposition_id"].astype(str)
df["_los"] = raw.loc[df.index, "time_in_hospital"]

df = df.sort_values(["patient_nbr", "encounter_id"])
g = df.groupby("patient_nbr")
df["seq_num"] = g.cumcount()
df["prev_label"] = g["label_raw"].shift(1).fillna("none")
df["prev_disposition"] = g["_disp"].shift(1).fillna("none")
df["prev_los"] = g["_los"].shift(1).fillna(0)
df = df.sort_index()

tr, va, te = prep.patient_level_split(df)
HIST = ["seq_num", "prev_label", "prev_disposition", "prev_los"]
DROP = [TARGET, "patient_nbr", "encounter_id", "label_raw", "_disp", "_los"]


def make(d, use):
    X = d.drop(columns=DROP + [c for c in HIST if c not in use])
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype(str)
    return X


def run(use):
    Xs = [make(d, use) for d in (tr, va, te)]
    allx = pd.concat(Xs, keys=[0, 1, 2])
    for c in allx.columns:
        if not pd.api.types.is_numeric_dtype(allx[c]):
            allx[c] = allx[c].astype("category")
    m = HGB(max_depth=4, learning_rate=0.05, max_iter=400, categorical_features="from_dtype",
            class_weight="balanced", random_state=0)
    m.fit(allx.loc[0], tr[TARGET])  # fixed iterations, no tuning on test
    return m.predict_proba(allx.loc[2])[:, 1]


def boot(y, p1, p0, groups, n=1000):
    d = pd.DataFrame({"y": y, "a": p1, "b": p0, "g": groups})
    by = {k: v for k, v in d.groupby("g")}
    keys = np.array(list(by))
    out = []
    for _ in range(n):
        s = pd.concat([by[k] for k in rng.choice(keys, len(keys))])
        if s.y.nunique() > 1:
            out.append((roc_auc_score(s.y, s.a), roc_auc_score(s.y, s.b)))
    out = np.array(out)
    diff = out[:, 0] - out[:, 1]
    return np.percentile(out[:, 0], [2.5, 97.5]), np.percentile(diff, [2.5, 97.5])


y = te[TARGET].to_numpy()
base = run([])
print(f"{'model':40s} AUROC  AUPRC   (patient-level 95% CI on AUROC) | paired diff vs base")
res = []
for name, use in [("Base (no history)", []),
                  ("+ seq_num", ["seq_num"]),
                  ("+ seq_num + prev_label", ["seq_num", "prev_label"]),
                  ("+ all history features", HIST)]:
    p = base if not use else run(use)
    ci, dci = boot(y, p, base, te["patient_nbr"].to_numpy())
    print(f"{name:40s} {roc_auc_score(y, p):.3f}  {average_precision_score(y, p):.3f}   "
          f"[{ci[0]:.3f}, {ci[1]:.3f}] | diff 95% CI [{dci[0]:+.3f}, {dci[1]:+.3f}]")
    res.append(dict(model=name, auroc=roc_auc_score(y, p), auprc=average_precision_score(y, p),
                    auroc_lo=ci[0], auroc_hi=ci[1], diff_lo=dci[0], diff_hi=dci[1]))
pd.DataFrame(res).to_csv("results/trajectory_features.csv", index=False)
print("\nSaved results/trajectory_features.csv")
