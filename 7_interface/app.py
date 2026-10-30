"""
Demo interface: 30-day readmission risk for diabetic inpatients (EXSU 500, Team 8).

COURSE PROTOTYPE for research and educational purposes. NOT a medical device. NOT for clinical use.
Run locally:  streamlit run 7_interface/app.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import altair as alt
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


NICE = {"number_inpatient": "Inpatient visits last year", "number_emergency": "Emergency visits last year", "number_outpatient": "Outpatient visits last year",
        "total_prior_visits": "Total visits last year", "discharge_disposition_id": "Discharged to", "admission_type_id": "Admission type",
        "admission_source_id": "Admission source", "time_in_hospital": "Days in hospital", "number_diagnoses": "Number of diagnoses",
        "num_medications": "Medications given", "num_lab_procedures": "Lab procedures", "num_procedures": "Other procedures", "age": "Age band",
        "gender": "Sex", "race": "Race", "payer_code": "Payer", "medical_specialty": "Admitting specialty", "A1Cresult": "HbA1c result",
        "max_glu_serum": "Glucose result", "insulin": "Insulin", "metformin": "Metformin", "change": "Medication changed", "diabetesMed": "On diabetes drug",
        "diag_1_group": "Primary diagnosis", "diag_2_group": "Secondary diagnosis", "diag_3_group": "Third diagnosis",
        "num_med_changes": "Number of drug changes", "a1c_measured_and_changed": "HbA1c measured and drugs changed"}


def code_options(col, prefix, labels):
    out = {}
    for v in cfg["choices"][col]:
        n = int(v.rsplit("_", 1)[1])
        out[labels.get(n, f"Code {n}")] = v
    return out


PRESETS = {
    "Example: lower-risk patient": dict(k_age="[50-60)", k_gender="Female", k_tih=2, k_nlab=30, k_nproc=0, k_nmed=8, k_ndx=5, k_nin=0, k_ned=0,
                                         k_nout=0, k_a1c="None", k_glu="None", k_insulin="No", k_metformin="Steady", k_changed="No",
                                         k_disch="Home", k_adm_type="Elective", k_adm_src="Physician referral", k_d1="Other"),
    "Example: higher-risk patient": dict(k_age="[70-80)", k_gender="Male", k_tih=9, k_nlab=60, k_nproc=2, k_nmed=24, k_ndx=9, k_nin=3, k_ned=2,
                                          k_nout=1, k_a1c=">8", k_glu="None", k_insulin="Up", k_metformin="No", k_changed="Ch",
                                          k_disch="Skilled nursing facility", k_adm_type="Emergency", k_adm_src="Emergency room", k_d1="Circulatory"),
}


INIT = dict(k_age="[70-80)", k_gender="Female", k_race=cfg["defaults"]["race"], k_payer=cfg["defaults"]["payer_code"],
            k_specialty=cfg["defaults"]["medical_specialty"], k_tih=4, k_adm_type="Emergency", k_adm_src="Emergency room", k_disch="Home",
            k_nlab=43, k_nproc=1, k_nmed=15, k_ndx=8, k_nin=0, k_ned=0, k_nout=0, k_a1c="None", k_glu="None", k_insulin="Steady",
            k_metformin="No", k_on_dm_med="Yes", k_changed="No", k_d1="Circulatory", k_d2=cfg["defaults"]["diag_2_group"],
            k_d3=cfg["defaults"]["diag_3_group"])
for _k, _v in INIT.items():
    st.session_state.setdefault(_k, _v)

# Force one silent rerun before any widget is built, on a session's very
# first script run. Without this, a freshly connected browser can render a
# widget's default state before the backend's first response (carrying the
# correct session_state values set just above) has arrived, so an
# interaction that happens in that window can submit a stale default
# instead of the intended one (logged 2026-10-26: the Race box showed the
# wrong value right after a fresh page load). Rerunning here means the
# first content the browser ever receives already has every widget's
# correct value baked into its initial spec, closing that window.
if not st.session_state.get("_booted"):
    st.session_state["_booted"] = True
    st.rerun()


def load_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v


st.warning(DISCLAIMER)                                  # shown above the tabs, so it is visible on every page
tab_calc, tab_about = st.tabs(["Risk calculator", "About, results and limits"])

with tab_calc:
    st.title("30-day readmission risk (prototype)")
    st.caption("Enter details known at discharge. Fields you leave at their default use the most common value in the training data.")
    pc = st.columns(len(PRESETS) + 2)
    for i, name in enumerate(PRESETS):
        pc[i].button(name, on_click=load_preset, args=(name,))
    st.caption("The example buttons only fill the form with made-up values so you can see how the tool behaves. Press *Estimate risk* afterwards.")
    with st.form("inputs"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Patient")
            age = st.selectbox("Age band", cfg["choices"]["age"], key="k_age")
            gender = st.selectbox("Sex", [g for g in cfg["choices"]["gender"] if g != "Unknown/Invalid"], key="k_gender")
            race = st.selectbox("Race (as recorded in the dataset)", cfg["choices"]["race"], key="k_race")
            payer = st.selectbox("Payer", cfg["choices"]["payer_code"], key="k_payer")
            specialty = st.selectbox("Admitting specialty", cfg["choices"]["medical_specialty"], key="k_specialty")
        with c2:
            st.subheader("This hospital stay")
            tih = st.number_input("Days in hospital", min_value=1, max_value=14, key="k_tih")
            adm_t = code_options("admission_type_id", "admission_type", ADM_TYPE)
            adm_type = st.selectbox("Admission type", list(adm_t), key="k_adm_type")
            adm_s = code_options("admission_source_id", "admission_source", ADM_SRC)
            adm_src = st.selectbox("Admission source", list(adm_s), key="k_adm_src")
            dis = code_options("discharge_disposition_id", "discharge_disposition", DISCH)
            dis_opts = list(dis) + list(EXCLUDED_DISCHARGE)
            disch = st.selectbox("Discharged to", dis_opts, key="k_disch")
            nlab = st.number_input("Lab procedures", min_value=0, max_value=132, key="k_nlab")
            nproc = st.number_input("Other procedures", min_value=0, max_value=6, key="k_nproc")
            nmed = st.number_input("Distinct medications given", min_value=1, max_value=81, key="k_nmed")
            ndx = st.number_input("Number of diagnoses recorded", min_value=1, max_value=16, key="k_ndx")
        with c3:
            st.subheader("History and diabetes care")
            nin = st.number_input("Inpatient visits in the past year", min_value=0, max_value=21, key="k_nin")
            ned = st.number_input("Emergency visits in the past year", min_value=0, max_value=76, key="k_ned")
            nout = st.number_input("Outpatient visits in the past year", min_value=0, max_value=42, key="k_nout")
            a1c = st.selectbox("HbA1c result", ["None", "Norm", ">7", ">8"], format_func=lambda v: {"None": "Not measured"}.get(v, v), key="k_a1c")
            glu = st.selectbox("Serum glucose result", ["None", "Norm", ">200", ">300"], format_func=lambda v: {"None": "Not measured"}.get(v, v), key="k_glu")
            insulin = st.selectbox("Insulin", ["No", "Steady", "Up", "Down"], key="k_insulin")
            metformin = st.selectbox("Metformin", ["No", "Steady", "Up", "Down"], key="k_metformin")
            on_dm_med = st.selectbox("On any diabetes medication", ["Yes", "No"], key="k_on_dm_med")
            changed = st.selectbox("Medication changed during stay", ["No", "Ch"], format_func=lambda v: "Yes" if v == "Ch" else "No", key="k_changed")
            dgroups = cfg["choices"]["diag_1_group"]
            d1 = st.selectbox("Primary diagnosis group", dgroups, key="k_d1")
            d2 = st.selectbox("Secondary diagnosis group", dgroups, key="k_d2")
            d3 = st.selectbox("Third diagnosis group", dgroups, key="k_d3")
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
                    top = contrib.reindex(contrib.effect.abs().sort_values(ascending=False).index).head(8).reset_index(drop=True)
                    top["label"] = top.feature.map(lambda f: NICE.get(f, f)) + ": " + top.value.str.replace("_", " ").str.slice(0, 28)
                    chart = alt.Chart(top).mark_bar().encode(
                        x=alt.X("effect:Q", title="Effect on risk (log-odds)"),
                        y=alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelLimit=420)),
                        color=alt.condition(alt.datum.effect > 0, alt.value("#d62728"), alt.value("#1f77b4")), tooltip=["label", "effect"])
                    st.altair_chart(chart, width="stretch")
                    st.caption("Red bars raise the risk estimate, blue bars lower it (SHAP values, in log-odds). These show what the model uses, not what causes readmission.")
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
    mv = cfg.get("model_version")
    if mv:
        st.caption(f"Model trained {mv.get('trained', 'date unknown')}. {mv.get('note', '')}")
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
    img = HERE / "assets/calibration_and_decision_curve.png"
    if img.exists():
        st.subheader("Is the percentage trustworthy? Calibration and decision curve")
        st.image(str(img))
        st.caption("Left: predicted risk against observed readmission rate on unseen test patients; the blue calibrated estimate follows the diagonal "
                   "(Brier 0.094, expected calibration error 0.007), while the raw model score (red) must not be read as a probability. "
                   "Right: net benefit of flagging patients at different risk thresholds compared with flagging everyone or no one.")
    f = HERE / "assets/subgroup_performance.csv"
    if f.exists():
        st.subheader("Does it work equally well for everyone? Subgroup audit")
        sg = pd.read_csv(f).fillna({"note": ""})
        st.dataframe(sg[["attribute", "group", "n", "positives", "auroc", "auroc_lo", "auroc_hi", "accuracy" if "accuracy" in sg else "acc", "sens", "spec", "note"]]
                     .rename(columns={"acc": "accuracy", "sens": "sensitivity", "spec": "specificity"}).round(3), hide_index=True, width="stretch")
        st.caption("At the operating threshold (11.6%). The model catches about 62% of women's readmissions but 52% of men's, and discriminates less well for patients 70 and over. "
                   "Race groups other than Caucasian and African American have too few readmissions to judge. First audit only; no intersections tested.")
    st.subheader("Limits you should know")
    st.markdown(
        "- **Not clinical software.** Research and teaching only.\n"
        "- **Old, narrow data:** 130 US hospitals, 1999-2008, inpatient diabetes stays; practice and coding have changed.\n"
        "- **Moderate accuracy:** about 1 in 6 patients flagged at the operating point is truly readmitted, and about 4 in 10 readmissions are missed. "
        "The data lacks lab values at discharge, social factors, follow-up plans and clinical notes.\n"
        "- **Population skew:** about three quarters of patients are recorded as White and 45% are over 70, so estimates for small groups are unreliable. "
        "A first audit by race, sex and age is shown in this tab; it found a sensitivity gap by sex.\n"
        "- **Not scored:** patients who died or went to hospice.\n"
        "- **Unlisted drugs are assumed 'No'**, and fields you leave unchanged use the most common training value.")
    st.caption("Source code, experiments and documentation: see the project repository.")
st.divider()
st.caption(DISCLAIMER)
