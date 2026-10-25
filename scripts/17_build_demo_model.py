"""
Step 17: Train and save the model behind the demo interface (7_interface/).

Final model = CatBoost with the settings chosen on validation in script 05 (depth 4, lr 0.1,
early stopping on validation). A Platt calibration (logistic regression on the raw score) is
fitted on the VALIDATION set so the interface can show a probability that means something
(class weighting distorts raw scores). Risk bands come from validation quantiles. The test
set is used once, only to report honest numbers shown in the interface.

Outputs (small, committed): 7_interface/model/catboost.cbm and 7_interface/model/config.json,
plus copies of the results tables the interface displays.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

TARGET = "readmit_30d"
OUT = Path("7_interface")
tr, va, te = [pd.read_csv(f"data/{s}.csv", keep_default_na=False) for s in ("train", "val", "test")]
feats = [c for c in tr.columns if c not in (TARGET, "patient_nbr")]
cats = tr[feats].select_dtypes(exclude="number").columns.tolist()
nums = [c for c in feats if c not in cats]
X = {k: d[feats].copy() for k, d in (("tr", tr), ("va", va), ("te", te))}
y = {k: d[TARGET].to_numpy() for k, d in (("tr", tr), ("va", va), ("te", te))}
pools = {k: Pool(X[k], y[k], cat_features=cats) for k in X}

m = CatBoostClassifier(depth=4, learning_rate=0.1, iterations=1000, loss_function="Logloss", eval_metric="AUC",
                       scale_pos_weight=(y["tr"] == 0).sum() / (y["tr"] == 1).sum(), random_seed=42, verbose=False,
                       early_stopping_rounds=50).fit(pools["tr"], eval_set=pools["va"])
m.save_model(str(OUT / "model/catboost.cbm"))

raw = {k: m.predict_proba(pools[k])[:, 1] for k in ("va", "te")}
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
cal = LogisticRegression(C=1e6).fit(logit(raw["va"]).reshape(-1, 1), y["va"])
prob = {k: cal.predict_proba(logit(raw[k]).reshape(-1, 1))[:, 1] for k in raw}

lo_cut, hi_cut = np.quantile(prob["va"], [0.40, 0.80])           # low: bottom 40%, high: top 20% (validation)
fpr, tpr, th = roc_curve(y["va"], prob["va"]); thr = float(th[np.argmax(tpr - fpr)])
band = lambda p: np.where(p < lo_cut, "low", np.where(p < hi_cut, "moderate", "high"))
bands = {}
for b in ("low", "moderate", "high"):
    mask = band(prob["te"]) == b
    bands[b] = dict(share=float(mask.mean()), n=int(mask.sum()), observed_rate=float(y["te"][mask].mean()))
yp = prob["te"] >= thr
metrics = dict(auroc=float(roc_auc_score(y["te"], prob["te"])), auprc=float(average_precision_score(y["te"], prob["te"])),
               brier=float(brier_score_loss(y["te"], prob["te"])), prevalence=float(y["te"].mean()),
               threshold=thr, accuracy=float(np.mean(yp == y["te"])), sensitivity=float(np.mean(yp[y["te"] == 1])),
               specificity=float(np.mean(~yp[y["te"] == 0])), always_no_accuracy=float(1 - y["te"].mean()), n_test=int(len(y["te"])))

def counts(col, top=None):
    vc = X["tr"][col].value_counts()
    return list(vc.index[:top]) if top else sorted(vc.index)

cfg = dict(features=feats, categorical=cats, numeric=nums,
           choices={c: counts(c, 25 if c in ("medical_specialty", "payer_code") else None) for c in cats},
           defaults={**{c: X["tr"][c].mode()[0] for c in cats}, **{c: float(X["tr"][c].median()) for c in nums}},
           ranges={c: [float(X["tr"][c].min()), float(X["tr"][c].max())] for c in nums},
           calibration=dict(coef=float(cal.coef_[0][0]), intercept=float(cal.intercept_[0])),
           band_cutoffs=dict(low=float(lo_cut), high=float(hi_cut)), bands=bands, metrics=metrics,
           med_columns=[c for c in feats if c in ("metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone")])
(OUT / "model/config.json").write_text(json.dumps(cfg, indent=1))
for src in ("5_results/main_models/model_comparison.csv", "5_results/model_variations_log.csv",
            "5_results/evaluation/calibration_and_decision_curve.png", "5_results/evaluation/subgroup_performance.csv"):
    if Path(src).exists(): shutil.copy(src, OUT / "assets" / Path(src).name)
print(json.dumps({"metrics": metrics, "bands": bands, "cutoffs": cfg["band_cutoffs"]}, indent=1))
print("model size KB:", round((OUT / "model/catboost.cbm").stat().st_size / 1024))
