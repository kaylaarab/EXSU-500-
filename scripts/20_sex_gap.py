"""
Step 20: Investigating the sensitivity gap by sex (women 0.62 vs men 0.52 at the operating threshold).

Questions
  1. Do men and women differ in who is readmitted (case mix), or in how well the model separates them?
  2. Is the gap a threshold artefact? Compare the single global threshold with (a) equal flag rates and
     (b) sex-specific thresholds chosen on VALIDATION to equalise sensitivity; evaluate on TEST.
  3. Would sex-specific models do better than the pooled model?
  4. Does the pooled model use sex directly (SHAP)?
Everything uses the final CatBoost + calibration saved for the interface (scripts 17/18); thresholds
and sex-specific cut-offs come from validation data only; test is scored once; patient-level
bootstrap (500 resamples). Accuracy is always shown with sensitivity.
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
from sklearn.metrics import roc_auc_score

TARGET, OUT = "readmit_30d", Path("5_results/evaluation")
cfg = json.loads(Path("7_interface/model/config.json").read_text()); feats, cats = cfg["features"], cfg["categorical"]
cal = cfg["calibration"]; thr = cfg["metrics"]["threshold"]
model = CatBoostClassifier(); model.load_model("7_interface/model/catboost.cbm")
tr, va, te = [pd.read_csv(f"data/{s}.csv", keep_default_na=False) for s in ("train", "val", "test")]
tr, va, te = [d[d.gender.isin(["Female", "Male"])].reset_index(drop=True) for d in (tr, va, te)]
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
score = lambda d: 1 / (1 + np.exp(-(cal["coef"] * logit(model.predict_proba(Pool(d[feats], cat_features=cats))[:, 1]) + cal["intercept"])))
p_va, p_te = score(va), score(te)
y_va, y_te = va[TARGET].to_numpy(), te[TARGET].to_numpy()
F_va, F_te = (va.gender == "Female").to_numpy(), (te.gender == "Female").to_numpy()
print(f"test n={len(y_te):,}  women {F_te.sum():,}  men {(~F_te).sum():,}")

# ---------------------------------------------------------------- 1. case mix
AGE = {a: int(a[1:a.index("-")]) + 5 for a in te.age.unique()}
def profile(d, tag):
    return {f"{tag} n": len(d), f"{tag} age (mean)": d.age.map(AGE).mean(), f"{tag} days in hospital": d.time_in_hospital.mean(),
            f"{tag} prior inpatient visits": d.number_inpatient.mean(), f"{tag} prior emergency visits": d.number_emergency.mean(),
            f"{tag} medications": d.num_medications.mean(), f"{tag} diagnoses": d.number_diagnoses.mean(),
            f"{tag} discharged home %": (d.discharge_disposition_id == "discharge_disposition_1").mean() * 100,
            f"{tag} circulatory primary dx %": (d.diag_1_group == "Circulatory").mean() * 100,
            f"{tag} on insulin %": (d.insulin != "No").mean() * 100}
allte = pd.concat([va, te])
prof = pd.DataFrame({"Women (all)": profile(allte[allte.gender == "Female"], ""), "Men (all)": profile(allte[allte.gender == "Male"], ""),
                     "Women readmitted": profile(allte[(allte.gender == "Female") & (allte[TARGET] == 1)], ""),
                     "Men readmitted": profile(allte[(allte.gender == "Male") & (allte[TARGET] == 1)], "")}).round(2)
prof.index = [i.strip() for i in prof.index]
prof.to_csv(OUT / "sex_gap_case_mix.csv"); print("\n", prof.to_string())

# ---------------------------------------------------------------- 2. separation and calibration by sex
sep = []
for name, m in (("Women", F_te), ("Men", ~F_te)):
    c = LogisticRegression(C=1e6).fit(logit(p_te[m]).reshape(-1, 1), y_te[m])
    sep.append(dict(group=name, auroc=roc_auc_score(y_te[m], p_te[m]), mean_risk_readmitted=p_te[m][y_te[m] == 1].mean(),
                    mean_risk_not_readmitted=p_te[m][y_te[m] == 0].mean(), observed_rate=y_te[m].mean(), mean_predicted=p_te[m].mean(),
                    calibration_slope=c.coef_[0][0], calibration_intercept=c.intercept_[0]))
sep = pd.DataFrame(sep); sep.round(4).to_csv(OUT / "sex_gap_separation.csv", index=False); print("\n", sep.round(3).to_string(index=False))

# ---------------------------------------------------------------- 3. threshold scenarios
flag_va = p_va >= thr
sens_target = flag_va[y_va == 1].mean()
share_target = flag_va.mean()
def cut_for(p, y, mask, kind):
    pm, ym = p[mask], y[mask]
    if kind == "flag_rate": return np.quantile(pm, 1 - share_target)
    return np.quantile(pm[ym == 1], 1 - sens_target)                  # equalise sensitivity
scen = {"S0 one global threshold": (np.where(F_te, thr, thr)),
        "S1 equal flag rate (each sex flags the same share)": np.where(F_te, cut_for(p_va, y_va, F_va, "flag_rate"), cut_for(p_va, y_va, ~F_va, "flag_rate")),
        "S2 equal sensitivity (sex-specific thresholds)": np.where(F_te, cut_for(p_va, y_va, F_va, "sens"), cut_for(p_va, y_va, ~F_va, "sens"))}
idx = pd.Series(np.arange(len(y_te))).groupby(te.patient_nbr.to_numpy()).apply(lambda s: s.to_numpy()).tolist()
rng = np.random.default_rng(3); res = [np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))]) for _ in range(500)]
def stats(flag, y, F):
    out = {}
    for nm, m in (("W", F), ("M", ~F)):
        yy, ff = y[m], flag[m]
        out[f"sens_{nm}"] = ff[yy == 1].mean(); out[f"spec_{nm}"] = (~ff[yy == 0]).mean(); out[f"flag_{nm}"] = ff.mean()
        out[f"acc_{nm}"] = (ff == yy).mean(); out[f"prec_{nm}"] = yy[ff].mean() if ff.any() else np.nan
    out["sens_all"] = flag[y == 1].mean(); out["acc_all"] = (flag == y).mean(); out["flag_all"] = flag.mean()
    out["gap"] = out["sens_W"] - out["sens_M"]; return out
rows = []
for name, cut in scen.items():
    flag = p_te >= cut; base = stats(flag, y_te, F_te)
    bs = pd.DataFrame([stats(flag[r], y_te[r], F_te[r]) for r in res])
    rows.append(dict(scenario=name, **{k: v for k, v in base.items()}, gap_lo=bs.gap.quantile(.025), gap_hi=bs.gap.quantile(.975),
                     cut_women=float(np.unique(cut[F_te])[0]), cut_men=float(np.unique(cut[~F_te])[0])))
sc = pd.DataFrame(rows); sc.round(4).to_csv(OUT / "sex_gap_thresholds.csv", index=False)
print("\n", sc[["scenario", "sens_W", "sens_M", "gap", "gap_lo", "gap_hi", "acc_all", "sens_all", "flag_all", "spec_W", "spec_M", "prec_W", "prec_M", "cut_women", "cut_men"]].round(3).to_string(index=False))

# ---------------------------------------------------------------- 4. sex-specific models vs pooled
def fit_sex(m_tr, m_va):
    dt, dv = tr[m_tr], va[m_va]; y = dt[TARGET].to_numpy()
    return CatBoostClassifier(depth=4, learning_rate=0.1, iterations=1000, loss_function="Logloss", eval_metric="AUC",
                              scale_pos_weight=(y == 0).sum() / (y == 1).sum(), random_seed=42, verbose=False,
                              early_stopping_rounds=50).fit(Pool(dt[feats], y, cat_features=cats), eval_set=Pool(dv[feats], dv[TARGET], cat_features=cats))
spec_p = np.zeros(len(te))
for nm, mt, mv, mte in (("W", tr.gender == "Female", F_va, F_te), ("M", tr.gender == "Male", ~F_va, ~F_te)):
    spec_p[mte] = fit_sex(mt.to_numpy(), mv).predict_proba(Pool(te[mte][feats], cat_features=cats))[:, 1]
cmp_rows = []
for nm, m in (("Women", F_te), ("Men", ~F_te)):
    a_pool, a_spec = roc_auc_score(y_te[m], p_te[m]), roc_auc_score(y_te[m], spec_p[m])
    d = [roc_auc_score(y_te[r][m[r]], spec_p[r][m[r]]) - roc_auc_score(y_te[r][m[r]], p_te[r][m[r]]) for r in res if 0 < y_te[r][m[r]].sum() < m[r].sum()]
    cmp_rows.append(dict(group=nm, auroc_pooled=a_pool, auroc_sex_specific=a_spec, diff=a_spec - a_pool, diff_lo=np.percentile(d, 2.5), diff_hi=np.percentile(d, 97.5)))
cmp_ = pd.DataFrame(cmp_rows); cmp_.round(4).to_csv(OUT / "sex_gap_sex_specific_models.csv", index=False); print("\n", cmp_.round(4).to_string(index=False))

# ---------------------------------------------------------------- 5. does the model use sex?
sub = te.sample(3000, random_state=0)
sv = model.get_feature_importance(Pool(sub[feats], cat_features=cats), type="ShapValues")[:, :-1]
imp = pd.Series(np.abs(sv).mean(0), index=feats).sort_values(ascending=False)
print(f"\nSHAP: gender mean |SHAP| = {imp['gender']:.4f}, rank {list(imp.index).index('gender') + 1} of {len(feats)}; top 5: {', '.join(imp.index[:5])}")
pd.DataFrame({"feature": imp.index, "mean_abs_shap": imp.values}).head(15).to_csv(OUT / "sex_gap_shap_top.csv", index=False)

fig, ax = plt.subplots(figsize=(7.2, 3.8)); xs = np.arange(len(sc)); w = 0.35
ax.bar(xs - w / 2, sc.sens_W, w, label="Women", color="#9467bd"); ax.bar(xs + w / 2, sc.sens_M, w, label="Men", color="#17becf")
ax.set_xticks(xs); ax.set_xticklabels(["one global\nthreshold", "equal\nflag rate", "equal sensitivity\n(sex-specific thresholds)"]); ax.set_ylabel("Sensitivity on test")
for i, r in sc.iterrows(): ax.text(i, 0.8, f"gap {r.gap * 100:+.0f} pts\n(95% CI {r.gap_lo * 100:+.0f} to {r.gap_hi * 100:+.0f})\naccuracy {r.acc_all:.2f}", ha="center", fontsize=7)
ax.set_ylim(0, 0.95); ax.legend(loc="lower right", fontsize=8); ax.set_title("Sex gap in sensitivity under three operating rules"); plt.tight_layout()
plt.savefig(OUT / "sex_gap_sensitivity.png", dpi=150)
