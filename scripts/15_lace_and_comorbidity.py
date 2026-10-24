"""
Step 15: Clinical-score benchmark (LACE) and comorbidity features.

Part 1  LACE index as a stand-alone predictor (no training): the standard clinical
        readmission score, computed from columns we have. Our models should beat it.
Part 2  Do LACE / Charlson comorbidity / 3-digit diagnosis features improve CatBoost?
        M0 baseline | M1 + LACE | M2 + Charlson score and condition flags |
        M3 + 3-digit primary diagnosis | M4 all three

LACE (van Walraven 2010): Length of stay (0-7), Acuity of admission (3 if emergent),
Comorbidity (Charlson-based 0-5), ED visits in prior period (0-4); total 0-19, >=10 = high risk.
Approximations in this dataset: acuity = admission type Emergency or Urgent; ED visits
use the prior-YEAR count (LACE uses 6 months); the Charlson score uses only the three
diagnosis fields on the stay, so it UNDER-counts comorbidity. Documented as limitations.

Same patient-level splits as all other scripts; test set scored once; fixed CatBoost
parameters (depth 4, lr 0.1, early stopping on validation); thresholds chosen on
validation; paired patient-level bootstrap against M0. Accuracy is always shown with sensitivity.
"""
import importlib.util
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

spec = importlib.util.spec_from_file_location("prep", "scripts/01_preprocess.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)
RAW, TARGET = "data/diabetic_data.csv", "readmit_30d"

df = prep.load_and_clean(RAW)
codes = pd.read_csv(RAW, usecols=["diag_1", "diag_2", "diag_3"], keep_default_na=False, na_values=["?"],
                    low_memory=False).loc[df.index]

# ---- Charlson comorbidity index from ICD-9 (Deyo-style ranges, approximate) --------------
CHARLSON = {  # name: (weight, [(lo, hi), ...]) numeric ICD-9 ranges, inclusive
    "mi": (1, [(410, 410.99), (412, 412.99)]),
    "chf": (1, [(428, 428.99), (398.91, 398.91), (402.01, 402.01), (402.11, 402.11), (402.91, 402.91),
                (404.01, 404.01), (404.03, 404.03), (404.11, 404.11), (404.13, 404.13), (404.91, 404.91),
                (404.93, 404.93), (425.4, 425.99)]),
    "pvd": (1, [(441, 441.99), (443.1, 443.99), (447.1, 447.1), (785.4, 785.4)]),
    "cerebrovascular": (1, [(430, 438.99)]),
    "dementia": (1, [(290, 290.99), (294.1, 294.19), (331.2, 331.29)]),
    "copd": (1, [(490, 505.99), (506.4, 506.4), (416.8, 416.9)]),
    "rheumatic": (1, [(710, 710.19), (710.4, 710.4), (714, 714.29), (714.8, 714.8), (725, 725.99)]),
    "ulcer": (1, [(531, 534.99)]),
    "mild_liver": (1, [(571, 571.99), (573.3, 573.4), (573.8, 573.9)]),
    "diabetes_complicated": (2, [(250.4, 250.79)]),            # diabetes with end-organ damage
    "hemiplegia": (2, [(342, 342.99), (343, 343.99), (344.0, 344.69), (344.9, 344.9)]),
    "renal": (2, [(403.01, 403.01), (403.11, 403.11), (403.91, 403.91), (404.02, 404.03), (404.12, 404.13),
                  (404.92, 404.93), (582, 583.79), (585, 586.99), (588.0, 588.0)]),
    "malignancy": (2, [(140, 172.99), (174, 195.99), (200, 208.99)]),
    "severe_liver": (3, [(456.0, 456.21), (572.2, 572.89)]),
    "metastatic": (6, [(196, 199.99)]),
    "hiv": (6, [(42, 44.99)]),
}


def to_num(c):
    try:
        return float(c)
    except (TypeError, ValueError):
        return None


nums = codes.map(to_num)
flags = pd.DataFrame(index=df.index)
for name, (w, ranges) in CHARLSON.items():
    hit = np.zeros(len(df), bool)
    for col in nums.columns:
        v = nums[col].to_numpy(dtype=float, na_value=np.nan)
        for lo, hi in ranges:
            hit |= (v >= lo) & (v <= hi)
    flags[f"cm_{name}"] = hit.astype(int)
weights = {f"cm_{k}": w for k, (w, _) in CHARLSON.items()}
charlson = sum(flags[c] * w for c, w in weights.items()) + 1       # +1: every patient has uncomplicated diabetes
df["charlson_score"] = charlson.to_numpy()
df["n_comorbid_flags"] = flags.sum(axis=1).to_numpy()

# ---- LACE ----------------------------------------------------------------------------------
los = df["time_in_hospital"]
L = np.select([los < 1, los == 1, los == 2, los == 3, los <= 6, los <= 13], [0, 1, 2, 3, 4, 5], 7)
A = df["admission_type_id"].isin(["admission_type_1", "admission_type_2"]).astype(int) * 3
cs = df["charlson_score"]
C = np.select([cs == 0, cs == 1, cs == 2, cs == 3], [0, 1, 2, 3], 5)
ed = df["number_emergency"]
E = np.minimum(ed, 4)
df["lace_L"], df["lace_A"], df["lace_C"], df["lace_E"] = L, A.to_numpy(), C, E.to_numpy()
df["lace"] = df[["lace_L", "lace_A", "lace_C", "lace_E"]].sum(axis=1)

# ---- 3-digit primary diagnosis (top 40, rest "other") ----------------------------------------
d3 = codes["diag_1"].astype(str).str.split(".").str[0].where(codes["diag_1"].notna(), "missing")
top = d3.value_counts().index[:40]
df["diag1_3digit"] = d3.where(d3.isin(top), "other").to_numpy()

tr, va, te = prep.patient_level_split(df)
yv, yt = va[TARGET].to_numpy(), te[TARGET].to_numpy()
groups = te["patient_nbr"].to_numpy()
idx = pd.Series(np.arange(len(yt))).groupby(groups).apply(lambda s: s.to_numpy()).tolist()
rng = np.random.default_rng(1)
resamples = [np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))]) for _ in range(1000)]


def op(pv, pt):
    fpr, tpr, th = roc_curve(yv, pv)
    t = th[np.argmax(tpr - fpr)]
    yp = pt >= t
    return t, np.mean(yp == yt), np.mean(yp[yt == 1]), np.mean(~yp[yt == 0])


def ci(p):
    a = [roc_auc_score(yt[r], p[r]) for r in resamples if yt[r].min() != yt[r].max()]
    return np.percentile(a, [2.5, 97.5])


rows = []
print(f"test n={len(yt):,}, positive rate {yt.mean():.3f}, always-'no' accuracy {1 - yt.mean():.3f}\n")

# ---- Part 1: LACE alone ----------------------------------------------------------------------
lv, lt = va["lace"].to_numpy(), te["lace"].to_numpy()
t, acc, sens, spec = op(lv, lt)
lo, hi = ci(lt)
rows.append(dict(model="LACE score alone (no training)", auroc=roc_auc_score(yt, lt), lo=lo, hi=hi,
                 auprc=average_precision_score(yt, lt), accuracy=acc, sensitivity=sens, specificity=spec,
                 threshold=t, diff=np.nan, diff_lo=np.nan, diff_hi=np.nan))
hi_risk = lt >= 10
print(f"LACE alone: AUROC={rows[-1]['auroc']:.3f} [{lo:.3f},{hi:.3f}] AUPRC={rows[-1]['auprc']:.3f} | validation threshold {t:.0f}: "
      f"Acc={acc:.3f} Sens={sens:.3f} Spec={spec:.3f} | conventional >=10: share flagged {hi_risk.mean():.3f}, "
      f"Acc={np.mean(hi_risk == yt):.3f} Sens={np.mean(hi_risk[yt == 1]):.3f} Spec={np.mean(~hi_risk[yt == 0]):.3f}\n")

# ---- Part 2: CatBoost with added features -----------------------------------------------------
FLAGS = [f"cm_{k}" for k in CHARLSON]
for k in FLAGS:
    df[k] = flags[k].to_numpy()
tr, va, te = prep.patient_level_split(df)               # same split (same patients, same seed)
BASE_DROP = [TARGET, "patient_nbr", "lace", "lace_L", "lace_A", "lace_C", "lace_E", "charlson_score",
             "n_comorbid_flags", "diag1_3digit"] + FLAGS
SETS = {
    "M0 baseline (current features)": [],
    "M1 + LACE score and its 4 components": ["lace", "lace_L", "lace_A", "lace_C", "lace_E"],
    "M2 + Charlson score and condition flags": ["charlson_score", "n_comorbid_flags"] + FLAGS,
    "M3 + 3-digit primary diagnosis": ["diag1_3digit"],
    "M4 all of M1 + M2 + M3": ["lace", "lace_L", "lace_A", "lace_C", "lace_E", "charlson_score", "n_comorbid_flags", "diag1_3digit"] + FLAGS,
}
preds = {}
for name, extra in SETS.items():
    X = [d.drop(columns=[c for c in BASE_DROP if c not in extra]) for d in (tr, va, te)]
    cats = X[0].select_dtypes(exclude="number").columns.tolist()
    for d in X:
        for c in cats:
            d[c] = d[c].astype(str)
    pools = [Pool(x, d[TARGET], cat_features=cats) for x, d in zip(X, (tr, va, te))]
    m = CatBoostClassifier(depth=4, learning_rate=0.1, iterations=1000, loss_function="Logloss", eval_metric="AUC",
                           scale_pos_weight=(tr[TARGET] == 0).sum() / (tr[TARGET] == 1).sum(), random_seed=42,
                           verbose=False, early_stopping_rounds=50).fit(pools[0], eval_set=pools[1])
    preds[name] = (m.predict_proba(pools[1])[:, 1], m.predict_proba(pools[2])[:, 1])
base = preds["M0 baseline (current features)"][1]
for name, (pv, pt) in preds.items():
    t, acc, sens, spec = op(pv, pt)
    lo, hi = ci(pt)
    d = [roc_auc_score(yt[r], pt[r]) - roc_auc_score(yt[r], base[r]) for r in resamples if yt[r].min() != yt[r].max()]
    dlo, dhi = np.percentile(d, [2.5, 97.5])
    rows.append(dict(model=name, auroc=roc_auc_score(yt, pt), lo=lo, hi=hi, auprc=average_precision_score(yt, pt),
                     accuracy=acc, sensitivity=sens, specificity=spec, threshold=t,
                     diff=roc_auc_score(yt, pt) - roc_auc_score(yt, base), diff_lo=dlo, diff_hi=dhi))
    r = rows[-1]
    print(f"{name:42s} AUROC={r['auroc']:.3f} [{lo:.3f},{hi:.3f}] AUPRC={r['auprc']:.3f} | vs M0 {r['diff']:+.4f} "
          f"[{dlo:+.4f},{dhi:+.4f}] | Acc={acc:.3f} Sens={sens:.3f} Spec={spec:.3f}")
pd.DataFrame(rows).to_csv("5_results/ablations/lace_comorbidity.csv", index=False)
print("\nSaved 5_results/ablations/lace_comorbidity.csv")
print("\nCharlson prevalence in cohort (share with flag):")
print(df[FLAGS].mean().round(3).sort_values(ascending=False).head(8).to_string())
print(f"mean Charlson score {df.charlson_score.mean():.2f}; mean LACE {df.lace.mean():.2f}")
