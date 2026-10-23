# Dataset analysis report

**Project:** Predicting 30-day readmission in diabetic inpatients (EXSU 500, Team 8)
**Dataset:** Diabetes 130-US Hospitals, 1999–2008 (UCI, CC BY 4.0)
**Basis:** profile of the real file (`scripts/00_profile.py`, tables in `results/data_profile/`) and re-runs of the pipeline on the cleaned data.

---

## 1. What the data is

| | |
|---|---|
| Rows | 101,766 inpatient encounters |
| Patients | 71,518 (up to 40 encounters each) |
| Columns | 50 (demographics, admission/discharge, utilisation, diagnoses, labs, 23 drugs) |
| Target | `readmitted`: `<30` days 11,357 (11.2%), `>30` 35,545, `NO` 54,864 |
| Our label | 1 = readmitted within 30 days, 0 = everything else |
| After cleaning | 99,343 encounters, 11.4% positive |

---

## 2. Limitations and risks of the raw data

| # | Limitation | Evidence | Why it matters |
|---|---|---|---|
| 1 | **Repeat patients dominate** | 23.5% of patients have >1 encounter, but they make up **46.2% of rows**. Positive rate is 11.2% on all rows vs **8.8% on first encounters only** | Same patient's rows are correlated; leaks identity across splits if split by row; metrics describe encounters, not patients |
| 2 | **Death/hospice discharges** | 2,423 rows (2.4%). Positive rate **1.8%** vs 11.4% for others | These patients cannot be readmitted. Model learns "died → no readmission" and looks better than it is |
| 3 | **Large missing/unmeasured fields** | `weight` 96.9%, `max_glu_serum` 94.8%, `A1Cresult` 83.3%, `medical_specialty` 49.1%, `payer_code` 39.6%, `race` 2.2% | Little clinical lab/vital information; what remains reflects *who got tested*, not just *who is sick* |
| 4 | **Skewed population** | White 76.5%, Black 19.3%, Hispanic 2.0%, Other 1.5%, Asian 0.6%. 45% of rows are age 70+; only 6% under 40 | Small groups give unreliable estimates; findings mostly apply to older White patients |
| 5 | **Old and narrow** | 1999–2008, 130 US hospitals, ICD-9 codes, no SGLT2/GLP-1 drugs, no hospital ID | Practice, coding and treatments have changed; unknown hospital effects |
| 6 | **Label is imperfect** | Readmission is only observed within the data; a readmission elsewhere, or a death after discharge, counts as 0. `<30` vs `>30` is a hard cut at day 30 | Some true readmissions are mislabelled negative (label noise) |
| 7 | **High-cardinality / sparse features** | ~750 distinct ICD-9 codes per diagnosis slot; 15 of 23 drug columns are ≥99% "No"; age 0–10 has 161 rows | Sparse one-hot encoding; noisy rare categories |
| 8 | **Code bugs in our own first pass** | Pandas read the text `"None"` (lab not ordered) as missing, so `a1c_measured_and_changed` was identical to `change == "Ch"` (46%, should be ~9.5%). Admission/discharge ID codes were treated as numbers | Feature did not mean what the report says; ID codes imply a false ordering |
| 9 | **Weak signal** | Positive rate barely differs by race (9.6–11.3%), gender (11.1–11.3%) or age band beyond extremes | Ceiling on achievable accuracy; published work on this data sits at ~0.64–0.68 AUROC |

---

## 3. What we did, and the risk each action creates

| Action | What it fixes | New risk it creates |
|---|---|---|
| **Patient-level split** (70/15/15 by `patient_nbr`, no overlap, checked in code) | Row-level leakage (limitation 1) | Test set still weights frequent patients heavily; reported scores are per encounter, not per patient |
| **Excluded 2,423 death/hospice discharges** | Trivial "died → not readmitted" shortcut (2) | Removes disproportionately the oldest and sickest (7.3% of age 90+, 4.4% of age 80–90, vs 0.1–1% of under-50s). Model has **not been validated for end-of-life patients**; at deployment they must be filtered by rule |
| **Dropped `weight`, `payer_code`, `medical_specialty`** | Mostly-missing columns (3) | Drops insurance (a socioeconomic proxy) and specialty (possibly predictive). No socioeconomic information remains, although the literature links it to readmission. Avoids payer-based proxy bias, but also hides any disparity linked to it |
| **"Unknown" category for remaining missing values** (race, diagnoses) | Keeps rows rather than deleting them | Treats missing as informative; "Unknown race" patients may differ systematically; cannot audit them in a race analysis |
| **Kept "None" labs as their own category** | Fixed the pandas bug (8) | Not-measured reflects hospital practice and patient severity. Model may learn *care-pattern* signals, which can differ across hospitals and time |
| **Grouped ICD-9 into 9 categories** | Sparsity and noise (7) | Loses detail, e.g. diabetic complications vs control; coarse groups hide sub-populations |
| **Engineered features** (total prior visits, medication changes, A1C×change) | Denser signals from sparse columns | `total_prior_visits` partly encodes earlier encounters of the same patient; strong, clinically sensible, but a utilisation effect that can disadvantage people with limited access to care (fewer visits recorded ≠ healthier) |
| **ID codes as categories** (admission type, discharge disposition, source) | False numeric ordering (8) | More levels, some very rare (e.g. 5 patients) → rare levels get unstable weights |
| **Class weighting (`balanced`, `scale_pos_weight`)** | 11% positive rate | Predicted probabilities no longer calibrated; the 0.5 threshold is arbitrary |
| **Hyperparameters and blend weights tuned on validation; test used once** | Test-set leakage | Validation scores are optimistic; only test numbers are reported |

---

## 4. Bias and representation

- **Not yet measured:** we have **not** computed model performance by race, age or sex, so we cannot yet say whether errors differ across groups.
- **Subgroups are too small for reliable testing.** In the test split (15% of the data), Asian patients contribute roughly 10 readmissions and Hispanic patients roughly 35. Group-level AUROC would have very wide confidence intervals.
- **What the numbers do show:** readmission rates are similar across race and gender, so there is little label-level disparity to learn. The main representation risk is **coverage** (older, White, US-hospital patients), not a skewed label.
- **Proxy risks:** `discharge_disposition` (where the patient goes) and prior visits encode social circumstances (home support, access to care). A model using them may flag patients for reasons that are really about resources.

---

## 5. Is there enough information to predict readmission?

**Partly, but not enough for individual decisions.**

- Current results (see Section 6) are about 0.66–0.67 AUROC. A coin flip is 0.50.
- At the 0.5 threshold, sensitivity is about 0.55 and specificity about 0.68. With an 11% base rate that means **roughly 1 in 6 flagged patients is actually readmitted (precision ≈ 18%), and about 45% of true readmissions are missed**.
- What the data lacks: lab values and vitals at discharge, discharge plan and follow-up appointments, medication adherence, social determinants, clinical notes, and hospital identifiers.
- The strongest signals are prior-year inpatient visits and discharge destination, which are plausible but are markers of past utilisation, not causes.

**Conclusion:** the dataset supports a *population-level risk ranking* and a useful course exercise. It does **not** support clinical use.

---

## 6. Model results after the mitigations

| Model | AUROC | AUPRC | Sensitivity | Specificity |
|---|---|---|---|---|
| Baseline (always "no readmission") | 0.500 | 0.111 | 0.000 | 1.000 |
| Logistic regression | 0.663 | 0.212 | 0.528 | 0.689 |
| XGBoost (tuned on validation) | 0.665 | 0.222 | 0.540 | 0.695 |
| MLP (deep learning) | 0.659 | 0.219 | 0.539 | 0.686 |
| CatBoost (engineered features) | 0.669 | 0.221 | 0.568 | 0.670 |
| Ensemble (XGBoost 0.7 + CatBoost 0.3) | 0.669 | 0.224 | 0.550 | 0.683 |

95% bootstrap CI (encounter-level, 1,000 resamples): XGBoost 0.651–0.679; CatBoost and ensemble 0.655–0.682. **The models are statistically indistinguishable**; differences of 0.004 are well inside the interval. Sensitivity/specificity are at a fixed 0.5 threshold.

All values are on the held-out test split (patient-level, 99,343 cleaned encounters). These are **lower** than the first-pass numbers (e.g. ensemble 0.679) because those used uncleaned data that still contained the easy death/hospice rows and the broken A1C feature. The cleaned numbers are the honest ones.

---

## 7. What is still open

1. Subgroup performance (race, age, sex) with confidence intervals.
2. Calibration curve and a validation-chosen threshold.
3. Sensitivity run on **first encounter per patient only**.
4. Patient-level (cluster) bootstrap for confidence intervals.
5. Ablations: effect of each engineered feature and of the dropped columns.
6. Remove `diabetic_data.csv` and the split CSVs from `main`; the course requires data not to be committed.
