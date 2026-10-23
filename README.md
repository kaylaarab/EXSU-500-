# Predicting 30-Day Hospital Readmission in Diabetic Patients

**EXSU 500 — Fundamentals of Artificial Intelligence in Medicine, Fall 2026**
**Team 8 — Predictive analytics & clinical decision support**

Team members: Siham Khaldoun, Larissa Chiu, Magalie Achim, Kayla Arab

## Problem statement

Diabetic patients are at elevated risk of early hospital readmission, which is
both a recognized indicator of care quality and a significant driver of
healthcare cost. We predict whether a diabetic inpatient encounter will result
in a readmission within 30 days of discharge, using information available at
discharge: demographics, admission/discharge details, diagnoses, procedures,
lab results, and medication changes. A reliable risk flag at discharge could
support triage of patients for closer follow-up or discharge planning.

This is **not** intended as a deployable clinical tool. It is a course
exercise in building and honestly evaluating a clinical risk-prediction
pipeline.

## Task type

**Binary classification.** The source label `readmitted` has three levels
(`<30`, `>30`, `NO`); we collapse it to a binary target: readmitted within 30
days (`1`) vs. not (`0`, covering both `>30` and `NO`). This choice is made
because `<30` is the clinically actionable readmission window, and because
collapsing to binary keeps the evaluation protocol (AUROC/AUPRC, thresholded
sensitivity/specificity) simple and defensible. The resulting positive rate is
~11.2%, so the task is meaningfully imbalanced.

## Dataset

**Diabetes 130-US Hospitals for Years 1999-2008**

- Source: UCI Machine Learning Repository, [archive.ics.uci.edu/dataset/296](https://archive.ics.uci.edu/dataset/296)
- Original reference: Strack et al. (2014), "Impact of HbA1c Measurement on
  Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient
  Records," *BioMed Research International*.
- Licence: CC BY 4.0 (UCI ML Repository)
- Access tier: Open — no registration, no agreement required.
- Size: 101,766 inpatient encounters, 50 columns, covering 130 US hospitals
  over 1999–2008.
- **Grouping structure:** encounters are not independent — 101,766 encounters
  belong to only 71,518 unique patients (`patient_nbr`), with up to 40
  encounters for a single patient. All train/validation/test splits are made
  at the **patient level**, not the encounter level, to avoid leakage.
- **Class balance:** `NO` = 54,864 / `>30` = 35,545 / `<30` = 11,357 (~11.2%
  positive after binarization).
- **Known missingness:** `weight` is 96.9% missing (excluded as a feature);
  `medical_specialty` 49.1% missing and `payer_code` 39.6% missing (kept as
  an explicit "Unknown" category in the sensitivity analysis; dropped in the main pipeline); `race`, `diag_1/2/3`
  under 2.5% missing.
- `IDS_mapping.csv` (included with the download) decodes the numeric
  `admission_type_id`, `discharge_disposition_id`, and `admission_source_id`
  columns into human-readable categories.

## How to obtain the data

Data is **not** committed to this repository. To reproduce:

```bash
# Download directly from UCI (no account required)
curl -L -o data/diabetic_data.zip \
  https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip
unzip data/diabetic_data.zip -d data/
```

This produces `data/diabetic_data.csv` (~19 MB) and `data/IDS_mapping.csv`.

## Evaluation protocol

- **Cohort:** 99,343 encounters from 69,990 patients after excluding 2,423
  death/hospice discharges (those patients cannot be readmitted).
- **Splits** (made before any modelling, seed 42): train 69,519 encounters /
  48,993 patients, validation 14,911 / 10,498, held-out test 14,913 / 10,499.
  Split at the **patient** level (`patient_nbr`): no patient's encounters appear
  in more than one split (asserted in `scripts/01_preprocess.py`). Positive rate
  is 11.4% / 11.7% / 11.1%.
- **Tune on validation, report on test.** The test set is scored once per model.
- **Metrics:** AUROC and AUPRC (primary, given class imbalance); sensitivity and
  specificity at a stated threshold of 0.5. Accuracy is not used.
- **Uncertainty:** 95% bootstrap confidence intervals on the test set. The main
  table uses encounter-level resampling; `scripts/07_sensitivity.py` adds
  patient-level (cluster) bootstrap intervals, which are slightly wider.
- **Interpretability:** SHAP feature importance (`results/shap_summary.png`).
  Top drivers: prior-year inpatient visits, then discharge destination.
- **Imbalance handling:** class weights. SMOTE-NC was tested as an ablation and was
  clearly worse on real test data (`scripts/09_smote_ablation.py`). No synthetic
  data is used for training or evaluation.
- Rationale for each decision and its trade-offs: `dataset_analysis_report.md`.

## Models and current results

1. **Baseline**: majority-class predictor.
2. **Classical ML**: regularised logistic regression (`class_weight=balanced`).
3. **Classical ML**: XGBoost, tuned on validation (depth, trees, learning rate).
4. **Classical ML**: CatBoost with native categorical handling and engineered features.
5. **Deep learning**: MLP on one-hot features.
6. **Ensemble**: validation-tuned blend of XGBoost and CatBoost.

| Model | AUROC | AUPRC | Sensitivity | Specificity |
|---|---|---|---|---|
| Baseline (majority-class) | 0.500 | 0.111 | 0.000 | 1.000 |
| Logistic Regression | 0.663 | 0.212 | 0.528 | 0.689 |
| XGBoost | 0.665 | 0.222 | 0.540 | 0.695 |
| MLP | 0.659 | 0.219 | 0.539 | 0.686 |
| CatBoost | 0.669 | 0.221 | 0.568 | 0.670 |
| Ensemble | 0.669 | 0.224 | 0.550 | 0.683 |

95% CI on the best models is roughly 0.653-0.686; the models are statistically
indistinguishable. Keeping `medical_specialty` and `payer_code` adds about +0.008
AUROC (paired 95% CI +0.005 to +0.011), restricting to each patient's first
encounter gives about 0.652, and patient-history features add about +0.006 (borderline).
These are first complete results on the cleaned data, not final manuscript numbers.

## Repository structure

```
data/                    raw CSV + generated splits (git-ignored; never committed)
scripts/
  00_profile.py          profile the raw data (missingness, repeat patients, subgroups)
  01_preprocess.py       clean, exclude death/hospice, patient-level split
  02_train_evaluate.py   baseline, LogReg, XGBoost + bootstrap CI
  03_interpretability.py SHAP
  04_deep_learning.py    MLP
  05_catboost.py         CatBoost + engineered features
  06_ensemble.py         blended ensemble
  07_sensitivity.py      specialty/payer, first-encounter, patient-level CIs
  08_trajectory_features.py  patient-history features
  09_smote_ablation.py   SMOTE-NC vs class weights
results/                 all tables and figures
dataset_analysis_report.md   dataset, decisions, trade-offs, bias, results
PROJECT_STATUS.md        plain-language progress report for the team
LITERATURE_BENCHMARKS*.md  literature log and comparison with other work
```

Run order: `00` (optional) then `01` to `06`; `07` to `09` are analyses. Run all
from the repo root after downloading the data per above and installing the pinned
`requirements.txt`.

*(Still to add: model card, open-source licence, CRediT contribution statement,
calibration and subgroup analysis, per course requirements.)*

## Use of Generative AI

*(statement to be added per course requirements — document any AI assistance
used for code, debugging, or writing)*
