"""
Step 18: Calibration, decision curves and subgroup audit for the final model.

Model = the CatBoost + Platt calibration saved for the interface (7_interface/model/, built by
script 17: trained on the training split, calibrated on validation). Everything here is computed
on the TEST split, once. Patient-level bootstrap (500 resamples) for all intervals.

  1. Calibration: reliability curve (raw class-weighted score vs calibrated probability), Brier,
     expected calibration error, calibration slope and intercept.
  2. Decision curve: net benefit across thresholds vs "flag everyone" and "flag no one".
  3. Subgroups (race, sex, age): n, positives, prevalence, AUROC, and at the operating
     threshold chosen on validation: sensitivity, specificity, precision, flag rate, accuracy;
     calibration-in-the-large (mean predicted vs observed). Groups with fewer than 30 positives
     are marked "too few to conclude". Sensitivity gaps vs the largest group = equal-opportunity
     difference.
Accuracy is always shown with sensitivity.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

TARGET, OUT, N_BOOT = "readmit_30d", Path("5_results/evaluation"), 500
cfg = json.loads(Path("7_interface/model/config.json").read_text())
model = CatBoostClassifier(); model.load_model("7_interface/model/catboost.cbm")
cal = cfg["calibration"]; feats, cats = cfg["features"], cfg["categorical"]
va, te = [pd.read_csv(f"data/{s}.csv", keep_default_na=False) for s in ("val", "test")]
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
score = lambda d: model.predict_proba(Pool(d[feats], cat_features=cats))[:, 1]
raw_te = score(te); p_te = 1 / (1 + np.exp(-(cal["coef"] * logit(raw_te) + cal["intercept"])))
y, groups = te[TARGET].to_numpy(), te["patient_nbr"].to_numpy()
thr = cfg["metrics"]["threshold"]
flag = p_te >= thr
print(f"test n={len(y):,}, prevalence {y.mean():.3f}, threshold {thr:.3f}, AUROC {roc_auc_score(y, p_te):.3f}")

idx = pd.Series(np.arange(len(y))).groupby(groups).apply(lambda s: s.to_numpy()).tolist()
rng = np.random.default_rng(7)
resamples = [np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))]) for _ in range(N_BOOT)]

# ---------------------------------------------------------------- 1. calibration
def bins(p, n=10):
    q = pd.qcut(p, n, duplicates="drop")
    d = pd.DataFrame({"p": p, "y": y, "b": q}).groupby("b", observed=True).agg(mean_pred=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
    d["lo"], d["hi"] = [[(lambda k, m: (k / m + 1.96 ** 2 / (2 * m) - 1.96 * np.sqrt(k / m * (1 - k / m) / m + 1.96 ** 2 / (4 * m ** 2))) / (1 + 1.96 ** 2 / m))(r.observed * r.n, r.n) for r in d.itertuples()],
                       [(lambda k, m: (k / m + 1.96 ** 2 / (2 * m) + 1.96 * np.sqrt(k / m * (1 - k / m) / m + 1.96 ** 2 / (4 * m ** 2))) / (1 + 1.96 ** 2 / m))(r.observed * r.n, r.n) for r in d.itertuples()]]
    return d.reset_index(drop=True)
bc, br = bins(p_te), bins(raw_te)
ece = float((bc.n * (bc.mean_pred - bc.observed).abs()).sum() / bc.n.sum())
slope = LogisticRegression(C=1e6).fit(logit(p_te).reshape(-1, 1), y)
cal_stats = dict(brier_calibrated=brier_score_loss(y, p_te), brier_raw=brier_score_loss(y, raw_te),
                 brier_always_prevalence=brier_score_loss(y, np.full(len(y), y.mean())), ece_calibrated=ece,
                 calibration_slope=float(slope.coef_[0][0]), calibration_intercept=float(slope.intercept_[0]))
print({k: round(v, 4) for k, v in cal_stats.items()})
bc.assign(kind="calibrated").to_csv(OUT / "calibration_bins.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
ax[0].plot([0, 0.8], [0, 0.8], "k--", lw=1, label="perfect")
ax[0].errorbar(bc.mean_pred, bc.observed, yerr=[bc.observed - bc.lo, bc.hi - bc.observed], fmt="o-", color="#1f77b4", label="calibrated (used in the interface)")
ax[0].plot(br.mean_pred, br.observed, "s-", color="#d62728", alpha=.7, label="raw class-weighted score")
ax[0].set_xlim(0, 0.8); ax[0].set_ylim(0, 0.8); ax[0].set_xlabel("Predicted risk"); ax[0].set_ylabel("Observed readmission rate"); ax[0].legend(fontsize=8)
ax[0].set_title(f"Reliability on {len(y):,} test stays\nBrier {cal_stats['brier_calibrated']:.3f}, ECE {ece:.3f}")
# ---------------------------------------------------------------- 2. decision curve
ts = np.arange(0.02, 0.41, 0.01); N = len(y)
nb = []
for t in ts:
    f = p_te >= t; tp, fp = (f & (y == 1)).sum(), (f & (y == 0)).sum()
    nb.append(dict(threshold=t, model=tp / N - fp / N * t / (1 - t), flag_all=y.mean() - (1 - y.mean()) * t / (1 - t), flag_none=0.0, share_flagged=f.mean()))
nb = pd.DataFrame(nb); nb.to_csv(OUT / "decision_curve.csv", index=False)
ax[1].plot(nb.threshold, nb.model, label="model", color="#1f77b4"); ax[1].plot(nb.threshold, nb.flag_all, label="flag everyone", color="#7f7f7f")
ax[1].axhline(0, color="k", lw=1, label="flag no one"); ax[1].set_ylim(-0.03, 0.12)
ax[1].set_xlabel("Risk threshold for acting"); ax[1].set_ylabel("Net benefit"); ax[1].legend(fontsize=8); ax[1].set_title("Decision curve (test set)")
plt.tight_layout(); plt.savefig(OUT / "calibration_and_decision_curve.png", dpi=150); plt.close()
better = nb[nb.model > np.maximum(nb.flag_all, 0)]
print(f"decision curve: model beats both defaults for thresholds {better.threshold.min():.2f}-{better.threshold.max():.2f}" if len(better) else "model never beats defaults")

# ---------------------------------------------------------------- 3. subgroups
def age_group(a):
    lo = int(a[1:a.index("-")]); return "under 50" if lo < 50 else ("50-69" if lo < 70 else ("70-79" if lo < 80 else "80+"))
te["age_group"] = te["age"].map(age_group)
spec = [("race", "race"), ("gender", "sex"), ("age_group", "age")]
rows = []
def metrics(mask):
    yy, ff, pp = y[mask], flag[mask], p_te[mask]
    if yy.sum() == 0 or yy.sum() == len(yy): return None
    return dict(auroc=roc_auc_score(yy, pp), sens=ff[yy == 1].mean(), spec=(~ff[yy == 0]).mean(), acc=(ff == yy).mean(),
                prec=yy[ff].mean() if ff.any() else np.nan, flag=ff.mean())
for col, label in spec:
    for g in te[col].unique():
        if g == "Unknown/Invalid": continue
        mask = (te[col] == g).to_numpy()
        base = metrics(mask)
        if base is None: continue
        bs = {"auroc": [], "sens": []}
        for r in resamples:
            # rows of the resample that fall in the group (with multiplicity)
            sel = r[mask[r]]
            yy, ff, pp = y[sel], flag[sel], p_te[sel]
            if len(yy) and 0 < yy.sum() < len(yy):
                bs["auroc"].append(roc_auc_score(yy, pp)); bs["sens"].append(ff[yy == 1].mean())
        rows.append(dict(attribute=label, group=g, n=int(mask.sum()), positives=int(y[mask].sum()), prevalence=y[mask].mean(),
                         mean_predicted=p_te[mask].mean(), **base,
                         auroc_lo=np.percentile(bs["auroc"], 2.5), auroc_hi=np.percentile(bs["auroc"], 97.5),
                         sens_lo=np.percentile(bs["sens"], 2.5), sens_hi=np.percentile(bs["sens"], 97.5),
                         note="too few to conclude (under 30 readmissions)" if y[mask].sum() < 30 else ""))
sg = pd.DataFrame(rows)
sg["sens_gap_vs_largest_group"] = np.nan
for label in sg.attribute.unique():
    s = sg[sg.attribute == label]; ref = s.loc[s.n.idxmax()]
    sg.loc[s.index, "sens_gap_vs_largest_group"] = s.sens - ref.sens
sg.round(4).to_csv(OUT / "subgroup_performance.csv", index=False)
pd.DataFrame([cal_stats]).round(5).to_csv(OUT / "calibration_summary.csv", index=False)
show = sg[["attribute", "group", "n", "positives", "prevalence", "auroc", "auroc_lo", "auroc_hi", "acc", "sens", "spec", "prec", "mean_predicted", "sens_gap_vs_largest_group", "note"]]
print(show.round(3).to_string(index=False))
