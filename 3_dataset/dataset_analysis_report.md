# Dataset analysis, design decisions and trade-offs

**Project:** Predicting 30-day readmission in diabetic inpatients (EXSU 500, Team 8)
**Dataset:** Diabetes 130-US Hospitals, 1999–2008 (UCI #296, CC BY 4.0)
**Purpose of this file:** one place where the team can see *what the data is, what is wrong with it, what we decided and why, and what each decision costs*. Use it as the source for the Data, Methods and Limitations sections of the report.
**Last updated:** 2026-10-23 (after keeping payer/specialty in the main pipeline and the death/hospice ablation). All numbers come from scripts in `scripts/` and tables in `results/`; nothing here is hand-typed from memory.

---

## 1. The data at a glance

| | |
|---|---|
| Raw size | 101,766 inpatient encounters, 50 columns, 71,518 patients (1–40 encounters each) |
| Source | Health Facts database, 130 US hospitals, 1999–2008, de-identified, public |
| Target (raw) | `readmitted`: `<30` 11,357 (11.2%) · `>30` 35,545 · `NO` 54,864 |
| Our label | `readmit_30d` = 1 if `<30`, else 0 (binary) |
| Analysed cohort | **99,343 encounters, 69,990 patients, 11.4% positive** (after removing 2,423 death/hospice discharges) |
| Feature groups | demographics · admission/discharge codes · stay length · prior-year visit counts · 3 diagnosis codes · 2 labs · 23 drugs · change flags |
| Not in the data | dates, hospital ID, vitals, lab values, social factors, follow-up plans, notes, adherence |

---

## 2. What is wrong or missing, and does it matter?

| Issue | Evidence | Effect on the model |
|---|---|---|
| **Mostly-empty columns** | `weight` 96.9% missing; `max_glu_serum` 94.8% and `A1Cresult` 83.3% "not tested"; `medical_specialty` 49.1%; `payer_code` 39.6%; `race` 2.2%; diagnoses 0.02–1.4% | Little lab/clinical detail. "Not tested" reflects hospital practice and severity, not biology, so the model partly learns *care patterns*. |
| **Repeat patients** | 23.5% of patients have >1 encounter and make up 46.2% of rows (max 40). Positive rate 11.2% on all rows vs 8.8% on first encounters | Rows are correlated; a row-level split would leak (see §4). |
| **Death/hospice discharges** | 2,423 rows (2.4%); positive rate 1.8% vs 11.4% | These patients cannot be readmitted: a free, meaningless "signal" if kept. |
| **Skewed population** | White 76.5%, Black 19.3%, Hispanic 2.0%, Other 1.5%, Asian 0.6%; 45% aged 70+, 6% under 40 | Subgroup estimates for small groups are unreliable; findings mostly describe older White US inpatients. |
| **Old and narrow** | 1999–2008, ICD-9, no modern diabetes drugs, no hospital ID | Practice and coding have changed; hospital effects are invisible. |
| **Label is imperfect** | Readmission is only seen *within the data*. 30.7% of `<30` stays have no later record for that patient; 456 stays follow a "not readmitted" stay | Readmissions elsewhere, deaths after discharge and non-chronological ID order all become label noise. |
| **Sparse features** | ~750 ICD-9 codes per diagnosis slot; 15 of 23 drug columns are ≥99% "No"; age 0–10 has 161 rows | Noisy rare categories, large one-hot matrices. |
| **Weak signal** | Positive rate barely moves by race (9.6–11.3%), gender (11.1–11.3%) | Limits the achievable score (see §7). |

**Bottom line:** the core fields are complete, but the dataset lacks most of what actually drives readmission. The score ceiling (~0.67 AUROC, matching the literature) comes from missing *information*, not from missing *cells*.

---

## 3. Decision log: what we did, why, and the cost

| # | Decision | Rationale | Trade-off / new risk |
|---|---|---|---|
| D1 | **Exclude death/hospice discharges** (codes 11, 13, 14, 19, 20, 21) | Deaths cannot be readmitted (0 of 229 in our test set); hospice patients rarely are (9 of 118). Strack et al., the Cureus paper, EmbPred30 and the 2026 ceiling audit all exclude them, so it keeps our cohort comparable. **The ablation (Script 10) shows this is a validity and comparability choice, not a score lever:** on a common test set, a model trained *with* these rows scores the same (paired AUROC difference −0.0003, 95% CI −0.003 to +0.002). Scoring the easy rows only lifts the headline number by about 0.005 (0.684 vs 0.679). | Removes the oldest, sickest patients disproportionately (7.3% of age 90+, 4.4% of 80–90, <1% of under-50s). Model is **not validated for end-of-life patients**; deployment must filter them by rule. The hospice half is a judgment call (7.6% of hospice test rows were readmitted), but keeping it or not does not change the result. |
| D2 | **Read `"None"` as a value, not missing** (`keep_default_na=False`) | Pandas turned lab "None" (not ordered) into NaN, silently breaking the A1C feature (it equalled `change=="Ch"`, 46% instead of 9.5%). | None; it was a bug. |
| D3 | **Admission/discharge/source IDs as categories** | They are codes, not quantities (e.g. discharge 22 = rehab, 27.7% readmitted). Numeric treatment implies a false order. | More levels; some very rare (<10 rows). Gain: +0.004 AUROC in a quick test. |
| D4 | **Drop `weight`; keep `payer_code` and `medical_specialty` as categories ("Unknown" for missing). Adopted in the main pipeline on 2026-10-23.** | Weight is 97% empty. Specialty and payer help: **+0.008 AUROC, paired 95% CI +0.005 to +0.011** (Script 07). Strack et al. also kept specialty with a "missing" value, but dropped payer. | Payer is a socioeconomic proxy, so it may encode disparity; specialty missingness is itself informative. Open question: does the gain come from specialty alone? If so we can drop payer at no cost (§9). |
| D5 | **Group ICD-9 into 9 categories** (same scheme as Strack et al. 2014) | Sparse high-cardinality codes learn poorly. | Loses detail (e.g. diabetes complications vs control). |
| D6 | **Engineered features**: total prior visits, number of medication changes, A1C-measured-and-changed | Dense signals from sparse columns; used in follow-on work. | `total_prior_visits` is a utilisation measure and can disadvantage people with less access to care. |
| D7 | **Handle the 11% positive rate with class weights, not SMOTE** | See §6. | Probabilities are no longer calibrated; the 0.5 threshold is arbitrary. |
| D8 | **Fixed 0.5 threshold, reported with sensitivity/specificity** | Course requires a stated operating point. | Not clinically tuned; see §7. |
| D9 | **No synthetic data** | See §6. | None for the score; we forgo a method some papers use. |
| D10 | **Tune on validation, score test once** | Prevents test leakage. | Validation scores are optimistic; only test numbers are reported. |

---

## 4. Encounter level vs patient level (and why we split the way we do)

Two different questions are easy to confuse:

- **What is one row?** One *encounter* (hospital stay). We predict, per stay, "will this patient be readmitted within 30 days of *this* discharge?" That is the clinically actionable question.
- **What do we split on?** *Patients.* All stays of one patient go to the same set.

**How the split works** (`01_preprocess.py`, `GroupShuffleSplit` on `patient_nbr`, seed 42): 70% of patients to train, then the remaining 30% divided evenly into validation and test. A patient is never in two sets; the script asserts zero overlap, and we verified it.

| Set | Encounters | Patients | Positive rate |
|---|---|---|---|
| Train | 69,519 | 48,993 | 11.4% |
| Validation | 14,911 | 10,498 | 11.7% |
| Test | 14,913 | 10,499 | 11.1% |

**Why not split by encounter?** The same person's stays look alike. If one stay is in training and another in testing, the model can recognise the *person*, and the test score is inflated. This is the most common reason very high scores are reported on this dataset (e.g. the 0.974 AUROC claim in arXiv:2002.11215, which we treat as not credible).

**Trade-offs we accepted**
- Scores are *per encounter*. Frequent patients contribute more test rows, so the score describes "a random stay", not "a random patient".
- The split is random, not chronological (there are no dates), so we cannot test "train on older, test on newer".
- Not stratified by outcome; positive rates ended up close by chance.

**Sensitivity check, first encounter per patient only** (Script 07, patient-level bootstrap CIs; "reference" here is the earlier pipeline with payer/specialty dropped, so the main pipeline now corresponds to row B in Script 07, 0.677):

| Evaluation | n (test) | Positive rate | AUROC (95% CI) |
|---|---|---|---|
| All test encounters (reference) | 14,913 | 11.1% | 0.669 (0.653–0.686) |
| First encounters only, same model | 10,499 | 8.9% | 0.653 (0.634–0.671) |
| First encounters only, trained and tested | 10,499 | 8.9% | 0.652 (0.633–0.670) |

First stays have no history, so they are harder (≈0.652). **Report the all-encounter number as the headline and the first-encounter number as the "no history" case.**

**Confidence intervals** resample *patients*, not rows. This widens the AUROC interval from ±0.013 to ±0.016, so the earlier encounter-level intervals were slightly too optimistic.

---

## 5. Can the encounter history be used?

Yes, partly. Ordering a patient's stays by `encounter_id` (**assumed** chronological; no dates exist) gives real patterns:

| Pattern | Evidence |
|---|---|
| Readmission rises with stay number | 9.0% (1st) → 14.2% → 17.4% → 19.4% → 27.1% (5th+) |
| Readmission chains | After a stay that was itself a `<30` readmission, the next stay is readmitted 21.6% of the time (14.8% after a `>30` one) |
| First stays carry no history | prior-year inpatient visits: 0.18 on first stays vs 1.72 on later ones |

**Safe features** (known at discharge of the current stay): stay number, previous stay's label, previous discharge destination, previous length of stay.

**Unsafe features, never use:** total stays per patient, "is last stay", "has a later stay". 69% of readmitted stays have a later record by construction, so these leak the label.

| Model (HistGradientBoosting, same splits) | AUROC | Paired gain vs base (95% CI) |
|---|---|---|
| Base, no history | 0.675 | n/a |
| + stay number | 0.678 | 0.000 to +0.007 (not reliable) |
| + stay number + previous label | 0.681 | +0.002 to +0.010 |
| + all four history features | **0.682** | **+0.003 to +0.011** |

**Reading:** a small but now reliable gain of about 0.007 AUROC (the interval excludes zero). It supports a descriptive claim ("patients who keep returning keep being readmitted") but **not causation**: the data is observational, sicker patients return more, and the order is an assumption. Not yet in the main pipeline.

---

## 6. Why we do not use synthetic data (and what we tested)

**Rationale**
1. **It cannot add information.** A generator learns from the same 99,343 rows. Missing labs, social factors and follow-up plans stay missing, so the ceiling does not move.
2. **It cannot fix representation.** Generating more Asian or Hispanic patients from 641 and 2,037 real ones repeats the same small sample and creates false confidence.
3. **Leakage risk.** Synthetic rows must be made *after* the patient split and from training data only; one mistake inflates test scores.
4. **It breaks histories.** SMOTE interpolates between rows, which creates impossible patient trajectories (a stay 3 with no stay 2).
5. **Evaluation must be on real patients anyway**, so it cannot make results more trustworthy.

**Empirical check** (Script 09: SMOTE-NC fitted on the training split only, fixed hyperparameters, same test set; a control with plain random oversampling gave 0.663 for logistic regression, so the pipeline is sound):

| Model | Arm | AUROC | AUPRC | Accuracy | Sensitivity | Specificity | AUROC vs class weights (95% CI) |
|---|---|---|---|---|---|---|---|
| LogReg | class weights | 0.663 | 0.212 | 0.671 | 0.529 | 0.689 | n/a |
| LogReg | SMOTE-NC to 1:1 | 0.561 | 0.155 | 0.731 | 0.314 | 0.783 | −0.102 (−0.120 to −0.086) |
| LogReg | SMOTE-NC to 1:2 | 0.572 | 0.162 | 0.829 | 0.164 | 0.913 | −0.091 (−0.107 to −0.074) |
| XGBoost | class weights | 0.665 | 0.217 | 0.658 | 0.563 | 0.670 | n/a |
| XGBoost | SMOTE-NC to 1:1 | 0.625 | 0.194 | 0.889 | 0.024 | 0.997 | −0.040 (−0.051 to −0.029) |
| XGBoost | SMOTE-NC to 1:2 | 0.647 | 0.209 | 0.889 | 0.022 | 0.998 | −0.018 (−0.027 to −0.009) |

*(Ablation run with payer/specialty dropped; the conclusion is not expected to change.)*

**Reading:** class weighting beats SMOTE-NC clearly for both models. An independent benchmark that applied SMOTE correctly (training data only) also reached only 0.608 ROC-AUC (Gupta and Choudhary 2026). Trees trained on SMOTE-NC rows learn the synthetic distribution and then flag almost nothing on real data (sensitivity ≈ 0.02). **This is why accuracy must never be read alone:** those XGBoost rows show 88.9% accuracy, identical to the always-"no" baseline, while catching 2% of readmissions. The decision to use class weights is supported by data, not only by argument.

**Why high published scores should be doubted:** the 0.974 AUROC preprint (EmbPred30) applies SMOTE before cross-validation. Our own test of that mechanism is in [`../4_benchmarking/embpred30_review.md`](../4_benchmarking/embpred30_review.md).

**Where synthetic data *is* acceptable:** demo inputs for the optional interface and pipeline stress tests. Not for training or evaluation.

---

## 6b. Why the target is "readmitted within 30 days", and what other definitions give

Same cleaned cohort, same patient-level split, same fixed model (Script 13), test set scored once. Prevalence differs by target, so **only AUROC compares across rows**; accuracy and AUPRC must be read against their own baselines.

| Target | Test n | Positive rate | AUROC | AUPRC (baseline) | Accuracy (always-"no") | Sensitivity | Specificity |
|---|---|---|---|---|---|---|---|
| **T1: within 30 days vs everything else (ours)** | 14,913 | 11.1% | 0.675 | 0.225 (0.111) | 0.661 (0.889) | 0.580 | 0.671 |
| T2: any readmission vs none | 14,913 | 47.3% | 0.696 | 0.668 (0.473) | 0.642 (0.527) | 0.627 | 0.656 |
| T3: within 30 days vs never (later readmissions dropped) | 9,519 | 17.4% | 0.728 | 0.403 (0.174) | 0.707 (0.826) | 0.596 | 0.731 |

**Reading**
- Scores rise as the question gets easier, not because the model improves: T2 asks about a much more common event, and T3 removes the hard middle group (36% of test patients) that look like both groups.
- **T3 is not a usable task.** At prediction time we cannot know who will be readmitted late, so the test set would no longer match real use. Keep it only as a diagnostic.
- **T2 is a legitimate but different question** ("will this patient return at all?"). It is a useful second output, not a replacement.
- **We keep T1 as the headline** because 30-day readmission is the standard quality measure in US hospital policy (CMS Hospital Readmissions Reduction Program, which covers six conditions, not diabetes itself), and the dataset's own source paper (Strack et al.) defines it this way. The claim that an early return is more preventable by discharge planning is clinical reasoning, not a sourced finding; do not cite it as a fact.
- Whether the "later" information can help T1 is tested in Script 14 (three-class, two-stage, stacking). Results in `results/combined_targets.csv` once finished.

## 7. Is the information sufficient to predict readmission?

**Partly; enough for population-level ranking, not for individual decisions.**

Main results on the held-out test set (cleaned cohort, class-weighted, threshold 0.5):

| Model | AUROC | AUPRC | Accuracy | Sensitivity | Specificity |
|---|---|---|---|---|---|
| Baseline (always "no") | 0.500 | 0.111 | 0.889 | 0.000 | 1.000 |
| Logistic regression | 0.668 | 0.214 | 0.670 | 0.547 | 0.685 |
| XGBoost | 0.672 | 0.226 | 0.682 | 0.539 | 0.700 |
| MLP | 0.660 | 0.219 | 0.602 | 0.630 | 0.598 |
| CatBoost | **0.677** | 0.228 | 0.671 | 0.555 | 0.685 |
| Ensemble (XGBoost + CatBoost + LogReg blend) | 0.676 | 0.228 | 0.674 | 0.560 | 0.688 |

The models are statistically close (95% CIs ≈ ±0.015). In plain terms, at the 0.5 threshold with an 11% base rate:
- About **1 in 6 flagged patients is truly readmitted** (precision ≈ 18%).
- About **45% of true readmissions are missed**.
- Strongest drivers: prior-year inpatient visits, then discharge destination (SHAP). These are plausible markers of past utilisation, not causes.

Earlier numbers (ensemble 0.679) came from uncleaned data and the broken A1C feature. The figures above are in line with the literature: an independent 2026 audit on the identical cohort (99,343 encounters) reports 0.6692 for gradient boosting ([`../4_benchmarking/benchmarking_report.md`](../4_benchmarking/benchmarking_report.md)).

---

## 8. Bias and representation

- **Not yet measured:** performance by race, age and sex. This is the main fairness gap.
- **Group sizes limit testing:** the test split holds ≈10 Asian and ≈30 Hispanic readmissions, so subgroup AUROCs would have very wide CIs.
- **Labels are similar across groups**, so there is little label-level disparity to learn. The risk is **coverage** (older, White, US patients) and **proxies**: discharge destination, prior visits and payer encode social circumstances and access to care, so a flagged patient may reflect resources, not risk.
- **Exclusions have an age effect** (D1), and "Unknown race" patients cannot be audited.

---

## 9. Open items (in priority order)

1. ~~Adopt specialty + payer in the main pipeline~~ **Done 2026-10-23** (CatBoost 0.677).
2. **Specialty-only vs payer-only:** if specialty carries the gain, drop payer and remove the socioeconomic-proxy concern.
3. Subgroup performance by race, age and sex, with CIs and an explicit "too few to conclude" note.
4. Calibration curve and a validation-chosen threshold (replace 0.5).
5. Decide whether history features (§5, +0.007 AUROC) enter the headline model or stay a sensitivity result. Caveat: they rely on `encounter_id` being chronological, which we cannot verify.
6. Patient-level bootstrap CIs for every model in `model_comparison.csv`.
7. Ablations: engineered features, ICD grouping.
8. Entity embeddings and extra tuning (Script 12, running): expected gain small.
9. Learning curve to test the "ceiling is in the data" claim from the 2026 audit.
10. Midterm report (due **Mon 9 Nov**): ≥15 primary references, 3 IEEE pages, CRediT and Generative-AI statements.

## 10. Reproduce

```bash
# data: download from UCI (see README), place diabetic_data.csv in data/
pip install -r requirements.txt
python scripts/00_profile.py data/diabetic_data.csv    # profile + results/data_profile/
python scripts/01_preprocess.py                        # cleaned patient-level splits
python scripts/02_train_evaluate.py                    # baseline, LogReg, XGBoost
python scripts/03_interpretability.py                  # SHAP
python scripts/04_deep_learning.py                     # MLP
python scripts/05_catboost.py; python scripts/06_ensemble.py
python scripts/07_sensitivity.py                       # specialty/payer, first-encounter, patient CIs
python scripts/08_trajectory_features.py               # patient-history features
python scripts/09_smote_ablation.py                    # SMOTE vs class weights
python scripts/10_hospice_death_ablation.py            # does excluding death/hospice matter?
python scripts/11_smote_leakage_demo.py                # SMOTE-before-CV inflation
python scripts/12_embeddings_tuning.py                 # entity embeddings, extra tuning
python scripts/13_target_definitions.py                # <30d vs any readmission vs <30d-vs-never
python scripts/14_combined_targets.py                  # use the >30d information to help the 30-day model
```

The raw data and splits are git-ignored on purpose (course rule: do not commit data).
