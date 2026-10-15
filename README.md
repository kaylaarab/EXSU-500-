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
  `medical_specialty` 49.1% missing and `payer_code` 39.6% missing (encoded as
  an explicit "unknown" category rather than imputed); `race`, `diag_1/2/3`
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

- Splits: train (71,520 encounters / 50,062 patients) / validation (15,027 /
  10,728) / held-out test (15,219 / 10,728), split at the **patient level**
  (`patient_nbr`) so no patient's encounters appear in more than one split
  (verified programmatically in `scripts/01_preprocess.py`). Positive rate is
  consistent across splits (~11%).
- Metrics: AUROC and AUPRC (primary, given class imbalance), sensitivity and
  specificity at threshold 0.5. Accuracy is not used as a headline metric.
- Uncertainty: 95% bootstrap confidence interval (1,000 resamples) on the test
  set for the best model's AUROC. (Seed-based variability was tried first but
  is not meaningful for this XGBoost configuration — see note in
  `scripts/02_train_evaluate.py`.)
- Interpretability/calibration artifact: not yet built (planned: SHAP feature
  importance for the XGBoost model).

## Candidate models and current results

1. **Baseline** — majority-class predictor (always predicts "no readmission").
2. **Classical ML** — regularized logistic regression (`class_weight=balanced`).
3. **Classical ML** — gradient-boosted trees (XGBoost), tuned over max_depth
   and n_estimators on the validation set.
4. **Deep learning** — not yet added; to be decided as a team. On tabular data
   of this size a well-tuned classical model may legitimately be competitive,
   and the spec explicitly allows stopping at two strong classical models if
   justified in the manuscript.

| Model | AUROC | AUPRC | Sensitivity | Specificity |
|---|---|---|---|---|
| Baseline (majority-class) | 0.500 | 0.110 | 0.000 | 1.000 |
| Logistic Regression | 0.637 | 0.202 | 0.533 | 0.660 |
| XGBoost (tuned on val) | 0.677 | 0.226 | 0.603 | 0.661 |

XGBoost test AUROC: **0.677 (95% bootstrap CI: 0.662–0.690)**.

These are first-pass results (default-ish hyperparameters, small grid, no
feature engineering beyond basic cleaning) — a starting point to iterate on,
not final numbers for the manuscript.

## Repository structure

```
data/           # raw CSVs (gitignored) + generated train/val/test splits
scripts/
  01_preprocess.py       # load, clean, patient-level split
  02_train_evaluate.py   # train baseline/LogReg/XGBoost, evaluate, bootstrap CI
results/
  model_comparison.csv   # metrics table
```

Run order: `python scripts/01_preprocess.py` then `python scripts/02_train_evaluate.py`
(from the repo root, after downloading the data per above and installing
`requirements.txt`).

*(Still to add: model card, open-source licence, CRediT contribution
statement — per course requirements.)*

## Use of Generative AI

*(statement to be added per course requirements — document any AI assistance
used for code, debugging, or writing)*
