"""
Step 9: Ablation -- SMOTE-NC oversampling vs class weighting.

Many published papers on this dataset oversample with SMOTE; we use class
weights. This compares them on identical splits, for LogReg and XGBoost.

Leakage control: the patient-level split is made FIRST. SMOTE-NC is fitted on
the training split only; validation and test stay real, untouched patients.
Hyperparameters are fixed (XGBoost from 02_train_evaluate.py), so nothing is
tuned on test. The test set is scored once per arm.

Arms (same model, different way of handling the 11% positive rate):
  weights   class_weight='balanced' / scale_pos_weight   (our main approach)
  smote_1.0 SMOTE-NC to 1:1, no weighting
  smote_0.5 SMOTE-NC to 1:2 (minority = half of majority), no weighting
Synthetic rows are for the ablation only and are never used for evaluation.
"""
import importlib.util
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

spec = importlib.util.spec_from_file_location("prep", "scripts/01_preprocess.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)

RAW, TARGET = "data/diabetic_data.csv", "readmit_30d"
rng = np.random.default_rng(42)

df = prep.load_and_clean(RAW)
tr, va, te = prep.patient_level_split(df)
drop = [TARGET, "patient_nbr"]
Xtr, Xte = tr.drop(columns=drop), te.drop(columns=drop)
ytr, yte = tr[TARGET].to_numpy(), te[TARGET].to_numpy()
groups = te["patient_nbr"].to_numpy()

cat_cols = Xtr.select_dtypes(exclude="number").columns.tolist()
num_cols = [c for c in Xtr.columns if c not in cat_cols]
order = cat_cols + num_cols                       # categoricals first -> indices 0..k-1
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)


def encode(X, fit=False):
    out = X[order].copy()
    out[cat_cols] = (enc.fit_transform if fit else enc.transform)(out[cat_cols])
    return out


Etr, Ete = encode(Xtr, fit=True), encode(Xte)
cat_idx = list(range(len(cat_cols)))
pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_idx),
    ("num", StandardScaler(), list(range(len(cat_cols), len(order)))),
])


def resample(ratio):
    if ratio is None:
        return Etr.to_numpy(), ytr
    sm = SMOTENC(categorical_features=cat_idx, sampling_strategy=ratio, random_state=42, k_neighbors=5)
    X, y = sm.fit_resample(Etr.to_numpy(), ytr)
    print(f"  SMOTE-NC {ratio}: {len(ytr):,} -> {len(y):,} training rows "
          f"({(y == 1).sum():,} positive, {(y == 1).sum() - (ytr == 1).sum():,} synthetic)")
    return X, y


def make_model(kind, weighted, y):
    spw = (y == 0).sum() / (y == 1).sum()
    if kind == "LogReg":
        return LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced" if weighted else None)
    return XGBClassifier(max_depth=3, n_estimators=400, learning_rate=0.1, subsample=0.8,
                         colsample_bytree=0.8, eval_metric="aucpr", random_state=42, n_jobs=-1,
                         scale_pos_weight=spw if weighted else 1.0)


def patient_boot_diff(p1, p0, n=1000):
    d = pd.DataFrame({"y": yte, "a": p1, "b": p0, "g": groups})
    by = {k: v for k, v in d.groupby("g")}
    keys = np.array(list(by))
    diffs = []
    for _ in range(n):
        s = pd.concat([by[k] for k in rng.choice(keys, len(keys))])
        if s.y.nunique() > 1:
            diffs.append(roc_auc_score(s.y, s.a) - roc_auc_score(s.y, s.b))
    return np.percentile(diffs, [2.5, 97.5])


rows, preds = [], {}
arms = [("weights", None, True), ("smote_1.0", 1.0, False), ("smote_0.5", 0.5, False)]
for ratio_name, ratio, weighted in arms:
    X, y = resample(ratio)
    Xt = pre.fit_transform(pd.DataFrame(X, columns=order).astype({c: float for c in order}))
    Xe = pre.transform(Ete.astype(float))
    for kind in ("LogReg", "XGBoost"):
        m = make_model(kind, weighted, y).fit(Xt, y)
        p = m.predict_proba(Xe)[:, 1]
        preds[(kind, ratio_name)] = p
        tn, fp, fn, tp = confusion_matrix(yte, (p >= 0.5).astype(int)).ravel()
        rows.append(dict(model=kind, arm=ratio_name, auroc=roc_auc_score(yte, p),
                         auprc=average_precision_score(yte, p),
                         accuracy=(tp + tn) / (tp + tn + fp + fn), sensitivity=tp / (tp + fn), specificity=tn / (tn + fp),
                         mean_pred=p.mean()))
        print(f"{kind:8s} {ratio_name:10s} AUROC={rows[-1]['auroc']:.3f} AUPRC={rows[-1]['auprc']:.3f} "
              f"Acc={rows[-1]['accuracy']:.3f} Sens={rows[-1]['sensitivity']:.3f} Spec={rows[-1]['specificity']:.3f} mean_pred={p.mean():.3f}")

for r in rows:
    if r["arm"] == "weights":
        r["diff_vs_weights_lo"] = r["diff_vs_weights_hi"] = 0.0
    else:
        lo, hi = patient_boot_diff(preds[(r["model"], r["arm"])], preds[(r["model"], "weights")])
        r["diff_vs_weights_lo"], r["diff_vs_weights_hi"] = lo, hi
        print(f"{r['model']:8s} {r['arm']:10s} AUROC diff vs weights: "
              f"{r['auroc'] - [x for x in rows if x['model'] == r['model'] and x['arm'] == 'weights'][0]['auroc']:+.4f} "
              f"95% CI [{lo:+.4f}, {hi:+.4f}]")
pd.DataFrame(rows).to_csv("5_results/ablations/smote_ablation.csv", index=False)
print("\nSaved 5_results/ablations/smote_ablation.csv")
