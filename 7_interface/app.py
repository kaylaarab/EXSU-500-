"""
Demo interface: 30-day readmission risk for diabetic inpatients (EXSU 500, Team 8).

COURSE PROTOTYPE for research and educational purposes. NOT a medical device. NOT for clinical use.
Run locally:  streamlit run 7_interface/app.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier, Pool

HERE = Path(__file__).parent
st.set_page_config(page_title="Readmission risk prototype", page_icon="🩺", layout="wide")

DISCLAIMER = ("**Course prototype for research and educational purposes. This is not a medical device and must not be "
              "used for clinical decisions.** It was trained on 1999-2008 US hospital records and has not been validated "
              "on any current patient population.")


@st.cache_resource
def load():
    cfg = json.loads((HERE / "model/config.json").read_text())
    model = CatBoostClassifier()
    model.load_model(str(HERE / "model/catboost.cbm"))
    return cfg, model


cfg, model = load()
M, B = cfg["metrics"], cfg["bands"]

# Friendly labels for coded fields (from the dataset's IDS_mapping.csv)
ADM_TYPE = {1: "Emergency", 2: "Urgent", 3: "Elective", 4: "Newborn", 5: "Not available", 6: "Not recorded", 7: "Trauma centre", 8: "Not mapped"}
DISCH = {1: "Home", 2: "Another short-term hospital", 3: "Skilled nursing facility", 4: "Intermediate care facility",
         5: "Other inpatient care institution", 6: "Home with home health service", 7: "Left against medical advice",
         18: "Not recorded", 22: "Rehabilitation facility", 23: "Long-term care hospital", 25: "Not mapped", 28: "Psychiatric hospital"}
EXCLUDED_DISCHARGE = {"Died or hospice care": "Patients who died or went to hospice were excluded from training, because they cannot be readmitted. The model gives no score for them."}
ADM_SRC = {1: "Physician referral", 2: "Clinic referral", 3: "HMO referral", 4: "Transfer from a hospital", 5: "Transfer from a skilled nursing facility",
           6: "Transfer from another health facility", 7: "Emergency room", 9: "Not available", 17: "Not recorded", 20: "Not mapped"}


def code_options(col, prefix, labels):
    out = {}
    for v in cfg["choices"][col]:
        n = int(v.rsplit("_", 1)[1])
        out[labels.get(n, f"Code {n}")] = v
    return out


tab_calc, tab_about = st.tabs(["Risk calculator", "About, results and limits"])
st.warning(DISCLAIMER)

with tab_calc:
    st.title("30-day readmission risk (prototype)")
    st.caption("Enter details known at discharge. Fields you leave at their default use the most common value in the training data.")
    with st.form("inputs"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Patient")
            age = st.selectbox("Age band", cfg["choices"]["age"], index=cfg["choices"]["age"].index("[70-80)"))
            gender = st.selectbox("Sex", [g for g in cfg["choices"]["gender"] if g != "Unknown/Invalid"])
            race = st.selectbox("Race (as recorded in the dataset)", cfg["choices"]["race"], index=cfg["choices"]["race"].index(cfg["defaults"]["race"]))
            payer = st.selectbox("Payer", cfg["choices"]["payer_code"], index=cfg["choices"]["payer_code"].index(cfg["defaults"]["payer_code"]))
            specialty = st.selectbox("Admitting specialty", cfg["choices"]["medical_specialty"], index=cfg["choices"]["medical_specialty"].index(cfg["defaults"]["medical_specialty"]))
        with c2:
            st.subheader("This hospital stay")
            tih = st.number_input("Days in hospital", 1, 14, 4)
            adm_t = code_options("admission_type_id", "admission_type", ADM_TYPE)
            adm_type = st.selectbox("Admission type", list(adm_t), index=list(adm_t).index("Emergency"))
            adm_s = code_options("admission_source_id", "admission_source", ADM_SRC)
            adm_src = st.selectbox("Admission source", list(adm_s), index=list(adm_s).index("Emergency room"))
            dis = code_options("discharge_disposition_id", "discharge_disposition", DISCH)
            dis_opts = list(dis) + list(EXCLUDED_DISCHARGE)
            disch = st.selectbox("Discharged to", dis_opts, index=dis_opts.index("Home"))
            nlab = st.number_input("Lab procedures", 0, 132, 43)
            nproc = st.number_input("Other procedures", 0, 6, 1)
            nmed = st.number_input("Distinct medications given", 1, 81, 15)
            ndx = st.number_input("Number of diagnoses recorded", 1, 16, 8)
        with c3:
            st.subheader("History and diabetes care")
            nin = st.number_input("Inpatient visits in the past year", 0, 21, 0)
            ned = st.number_input("Emergency visits in the past year", 0, 76, 0)
            nout = st.number_input("Outpatient visits in the past year", 0, 42, 0)
            a1c = st.selectbox("HbA1c result", ["None", "Norm", ">7", ">8"], format_func=lambda v: {"None": "Not measured"}.get(v, v))
            glu = st.selectbox("Serum glucose result", ["None", "Norm", ">200", ">300"], format_func=lambda v: {"None": "Not measured"}.get(v, v))
            insulin = st.selectbox("Insulin", ["No", "Steady", "Up", "Down"], index=1)
            metformin = st.selectbox("Metformin", ["No", "Steady", "Up", "Down"])
            on_dm_med = st.selectbox("On any diabetes medication", ["Yes", "No"])
            changed = st.selectbox("Medication changed during stay", ["No", "Ch"], format_func=lambda v: "Yes" if v == "Ch" else "No")
            dgroups = cfg["choices"]["diag_1_group"]
            d1 = st.selectbox("Primary diagnosis group", dgroups, index=dgroups.index("Circulatory"))
            d2 = st.selectbox("Secondary diagnosis group", dgroups, index=dgroups.index(cfg["defaults"]["diag_2_group"]))
            d3 = st.selectbox("Third diagnosis group", dgroups, index=dgroups.index(cfg["defaults"]["diag_3_group"]))
        go = st.form_submit_button("Estimate risk", type="primary")

    if go:
        try:
            if disch in EXCLUDED_DISCHARGE:
                st.info(EXCLUDED_DISCHARGE[disch])
            else:
                row = dict(cfg["defaults"])
                row.update(age=age, gender=gender, race=race, payer_code=payer, medical_specialty=specialty,
                           time_in_hospital=tih, admission_type_id=adm_t[adm_type], admission_source_id=adm_s[adm_src],
                           discharge_disposition_id=dis[disch], num_lab_procedures=nlab, num_procedures=nproc,
                           num_medications=nmed, number_diagnoses=ndx, number_inpatient=nin, number_emergency=ned,
                           number_outpatient=nout, A1Cresult=a1c, max_glu_serum=glu, insulin=insulin, metformin=metformin,
                           diabetesMed=on_dm_med, change=changed, diag_1_group=d1, diag_2_group=d2, diag_3_group=d3)
                for c in cfg["med_columns"]:
                    if c not in ("insulin", "metformin"):
                        row[c] = "No"                                   # other drugs: "No" (over 99% of training rows)
                row["total_prior_visits"] = nin + ned + nout
                row["num_med_changes"] = sum(row[c] in ("Up", "Down") for c in cfg["med_columns"])
                row["a1c_measured_and_changed"] = int(a1c != "None" and changed == "Ch")
                X = pd.DataFrame([row])[cfg["features"]]
                pool = Pool(X, cat_features=cfg["categorical"])
                raw = float(model.predict_proba(pool)[0, 1])
                z = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / (1 - np.clip(raw, 1e-6, 1 - 1e-6)))
                p = float(1 / (1 + np.exp(-(cfg["calibration"]["coef"] * z + cfg["calibration"]["intercept"]))))
                band = "low" if p < cfg["band_cutoffs"]["low"] else ("moderate" if p < cfg["band_cutoffs"]["high"] else "high")
                col = {"low": "green", "moderate": "orange", "high": "red"}[band]
                r1, r2 = st.columns([1, 2])
                with r1:
                    st.metric("Estimated 30-day readmission risk", f"{p * 100:.1f}%", delta=f"{(p - M['prevalence']) * 100:+.1f} points vs average ({M['prevalence'] * 100:.1f}%)", delta_color="inverse")
                    st.markdown(f"Risk band: **:{col}[{band.upper()}]**")
                    st.caption(f"Among {B[band]['n']:,} unseen test patients in this band, **{B[band]['observed_rate'] * 100:.1f}%** were actually readmitted within 30 days. "
                               f"Low band: {B['low']['observed_rate'] * 100:.1f}%, moderate: {B['moderate']['observed_rate'] * 100:.1f}%, high: {B['high']['observed_rate'] * 100:.1f}%.")
                    st.caption("The estimate is uncertain: the model ranks patients only moderately well (AUROC about 0.68), so treat it as a rough indication of relative risk.")
                with r2:
                    st.subheader("What pushed the estimate up or down")
                    sv = model.get_feature_importance(pool, type="ShapValues")[0][:-1]
                    contrib = pd.DataFrame({"feature": cfg["features"], "effect": sv, "value": [str(X.iloc[0][f]) for f in cfg["features"]]})
                    top = contrib.reindex(contrib.effect.abs().sort_values(ascending=False).index).head(8).iloc[::-1]
                    top["label"] = top.feature + " = " + top.value
                    st.bar_chart(top.set_index("label")["effect"], horizontal=True)
                    st.caption("Positive bars raise the risk estimate, negative bars lower it (SHAP values, in log-odds). These show what the model uses, not what causes readmission.")
        except Exception as e:                                          # never crash on odd input
            st.error(f"Could not compute a score for these inputs ({type(e).__name__}). Please check the values and try again.")

with tab_about:
    st.header("About this prototype")
    st.markdown(
        "Part of the EXSU 500 (Fall 2026) team project, Team 8. The model predicts whether a diabetic inpatient will be readmitted "
        "to hospital within 30 days of discharge, using information known at discharge.\n\n"
        "**Data:** UCI *Diabetes 130-US Hospitals* (1999-2008, 101,766 stays, CC BY 4.0). We used 99,343 stays after excluding "
        "patients who died or went to hospice. Splits are by patient, so no patient appears in both training and test sets.\n\n"
        "**Model:** CatBoost gradient boosting with a calibration step so the percentage is meaningful; risk bands come from validation data.")
    a, b, c, d = st.columns(4)
    a.metric("AUROC (test)", f"{M['auroc']:.3f}")
    b.metric("AUPRC (test)", f"{M['auprc']:.3f}", help=f"A model that guesses at random scores {M['prevalence']:.3f}")
    c.metric("Sensitivity", f"{M['sensitivity']:.2f}", help="Share of truly readmitted patients flagged at the operating threshold")
    d.metric("Accuracy", f"{M['accuracy']:.2f}", help=f"Always answering 'not readmitted' would score {M['always_no_accuracy']:.2f} accuracy with 0 sensitivity")
    st.caption(f"Operating threshold {M['threshold'] * 100:.1f}% risk, chosen on validation data; specificity {M['specificity']:.2f}. Test set: {M['n_test']:,} stays. "
               "Accuracy is shown beside sensitivity because the readmission rate is only 11%.")
    f = HERE / "assets/model_comparison.csv"
    if f.exists():
        st.subheader("All models compared (test set)")
        st.dataframe(pd.read_csv(f).round(3), hide_index=True, width="stretch")
    f = HERE / "assets/model_variations_log.csv"
    if f.exists():
        with st.expander("Every model variation we tried (experiment log)"):
            st.dataframe(pd.read_csv(f)[["id", "stage", "script", "model", "auroc", "auprc", "accuracy", "sensitivity", "specificity", "takeaway"]].round(3),
                         hide_index=True, width="stretch")
    st.subheader("Limits you should know")
    st.markdown(
        "- **Not clinical software.** Research and teaching only.\n"
        "- **Old, narrow data:** 130 US hospitals, 1999-2008, inpatient diabetes stays; practice and coding have changed.\n"
        "- **Moderate accuracy:** about 1 in 6 patients flagged at the operating point is truly readmitted, and about 4 in 10 readmissions are missed. "
        "The data lacks lab values at discharge, social factors, follow-up plans and clinical notes.\n"
        "- **Population skew:** about three quarters of patients are recorded as White and 45% are over 70, so estimates for small groups are unreliable. "
        "Performance by race, age and sex has not yet been audited.\n"
        "- **Not scored:** patients who died or went to hospice.\n"
        "- **Unlisted drugs are assumed 'No'**, and fields you leave unchanged use the most common training value.")
    st.caption("Source code, experiments and documentation: see the project repository.")
st.divider()
st.caption(DISCLAIMER)
