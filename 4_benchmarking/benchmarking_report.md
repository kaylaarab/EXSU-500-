# Benchmarking report: what others did on this dataset, and how we compare

**Last updated:** 2026-10-23. Part of the team repo; start with the summary, use the tables as the source for the Related Work section.
**Companion files in this folder:** [`embpred30_review.md`](embpred30_review.md) (analysis of the 0.974 AUROC claim) · [`sources.csv`](sources.csv) (same table, spreadsheet form) · [`abstracts/abstracts.md`](abstracts/abstracts.md) · [`papers/`](papers/) (downloaded open-access copies) · [`team_literature_log.md`](team_literature_log.md) (Kayla's and Magalie's notes).

## 1. Summary

1. **Credible results on this dataset sit between about 0.60 and 0.68 AUROC.** Ours (0.660-0.677) are inside that band. The one outlier is EmbPred30 at 0.974.
2. **Our cohort matches the careful studies.** Death and hospice discharges removed gives 99,343 encounters from 69,990 patients with 11.4% positive; the 2026 ceiling audit uses the identical cohort and reports 0.6692 AUROC with gradient boosting, essentially our result.
3. **The 0.974 claim is not trustworthy.** We checked the paper: it keeps only the first encounter per patient and removes death/hospice, so repeated-patient leakage is not the cause. Its own preprocessing list applies SMOTE (to 1:1) before the 6-fold cross-validation and never says SMOTE was confined to the training folds. Details and our own test of the mechanism: [`embpred30_review.md`](embpred30_review.md).
4. **Studies that apply SMOTE correctly (training data only) do not get high scores.** Gupta and Choudhary (2026) report a best ROC-AUC of 0.6079; our SMOTE-NC ablation also scored below class weighting (0.56-0.65). Two independent tests point the same way.
5. **Three of our choices are verified against the original paper.** Strack et al. (2014) used first encounters only, removed hospice/death, dropped `payer_code`, and kept `medical_specialty` with a "missing" value.
6. **Published accuracy figures are mostly the do-nothing baseline.** About 11% of patients are readmitted, so always answering "no" scores 88.9% accuracy with 0% sensitivity. The closest benchmark (Cureus 2025) reports 0.89 accuracy but catches only 1.7% of readmitted patients with XGBoost (its own Table 2). Our models have lower accuracy (0.60-0.68) because they catch 54-63% of readmissions. See section 2b; always compare accuracy and sensitivity together.

## 2. Our results for reference

*Cleaned cohort (99,343 encounters), patient-level split, test set scored once, `payer_code` and `medical_specialty` kept. Source: `5_results/main_models/model_comparison.csv`; reasoning in [`../3_dataset/dataset_analysis_report.md`](../3_dataset/dataset_analysis_report.md).*

| Model | AUROC | AUPRC | Accuracy | Sensitivity | Specificity |
|---|---|---|---|---|---|
| Baseline (always "no readmission") | 0.500 | 0.111 | 0.889 | 0.000 | 1.000 |
| Logistic regression | 0.668 | 0.214 | 0.670 | 0.547 | 0.685 |
| XGBoost (tuned on validation) | 0.672 | 0.226 | 0.682 | 0.539 | 0.700 |
| MLP (deep learning) | 0.660 | 0.219 | 0.602 | 0.630 | 0.598 |
| CatBoost (engineered features) | **0.677** | 0.228 | 0.671 | 0.555 | 0.685 |
| Ensemble (CatBoost + XGBoost + LogReg) | 0.676 | 0.228 | 0.674 | 0.560 | 0.688 |

The models are statistically close (95% intervals about +/-0.015). The first-pass numbers we reported earlier (0.678-0.679) came from uncleaned data and a bug and are superseded.

## 2b. Accuracy and sensitivity side by side

Sensitivity here means the share of truly readmitted patients the model catches. "Always-no" accuracy is the accuracy of predicting nobody is readmitted (about 88.9% at 11.2% prevalence). A model whose accuracy is near that value and whose sensitivity is near zero has learned nothing useful at its threshold.

| Study and model | AUROC | Accuracy | Sensitivity (readmitted) | Always-no accuracy | Reading |
|---|---|---|---|---|---|
| **Ours: CatBoost** | 0.677 | 0.671 | **0.555** | 0.889 | Catches over half of readmissions; accuracy below baseline is the price |
| **Ours: MLP** | 0.660 | 0.602 | **0.630** | 0.889 | Most sensitive of ours |
| Cureus 2025: XGBoost | 0.667 | 0.89 | 0.017 | 0.888 | Accuracy equals baseline; catches 2% |
| Cureus 2025: logistic regression | 0.642 | 0.89 | 0.016 | 0.888 | Same |
| Cureus 2025: random forest | 0.630 | 0.89 | 0.002 | 0.888 | Catches almost nobody |
| Cureus 2025: DNN | 0.579 | 0.84 | 0.143 | 0.888 | Highest sensitivity of theirs, still low |
| Gupta and Choudhary 2026: Extra Trees | 0.608 | 0.865 | not stated | about 0.889 | Accuracy below baseline |
| Gupta and Choudhary 2026: AdaBoost | not stated | not stated | 0.334 | about 0.889 | Best recall of their eight models |
| GitHub project (LogReg + oversampling) | 0.649 | not reported | 0.51 | n/a | Threshold tuned to recall at least 0.5 |
| ICCS 2022 (MIMIC-IV, elderly) | 0.75 | not reported | 0.69 | n/a | Different dataset |
| EmbPred30 2020 | 0.974 | 0.952 | about 0.943 (read from a figure) | **0.50** (balanced data) | Evaluated on SMOTE-balanced data, so not comparable |

What this shows:
1. **At the same AUROC (about 0.66-0.67) our models are far more sensitive than the Cureus models**, because we use class weighting and report a threshold that actually flags patients; their reported thresholds flag almost no one. AUROC is the fair comparison across studies; sensitivity and accuracy depend on the threshold chosen.
2. **Accuracy is not a usable ranking metric here.** The three best-looking Cureus models by accuracy are the three that catch the fewest readmissions.
3. **EmbPred30's 95.2% accuracy is not comparable** to the natural-prevalence figures above: on a 1:1 balanced evaluation the do-nothing baseline is 50%, not 89%.
4. **Cureus is internally inconsistent** (Siham's earlier finding, now verified): Table 1 lists XGBoost "recall" 0.84, but Table 2 gives recall for the readmitted class as 0.017. Table 3 confirms it (2,244 false negatives out of about 2,280 readmitted test patients). Table 1 appears to average over both classes, which hides the failure. Cite their AUROC only.

## 3. How our choices compare with other work

| Practice | What others do | Where verified | Us |
|---|---|---|---|
| **One row per patient (first encounter)** | Strack et al. 2014 and EmbPred30 keep only the first encounter (Strack: 69,984 encounters). A GitHub project reports 69,973 patients, 9.0% positive (snippet only). | Full text of Strack and EmbPred30 | Main analysis keeps all encounters but splits by patient; first-encounter-only is a sensitivity run: AUROC 0.652, 8.9% positive |
| **Exclude death and hospice** | Strack, EmbPred30, Cureus, the CSoI report, the audit and a GitHub project all do | Full text | Same (codes 11, 13, 14, 19, 20, 21); see the ablation in `5_results/ablations/` |
| **Patient-level splitting** | Rarely stated. Cureus uses a stratified 80/20 split with no patient grouping described; the CSoI report uses 60/40 | Full text | Patient-level split, asserted in code |
| **Handling class imbalance** | Class weights (Cureus, us); SMOTE or oversampling (EmbPred30, Gupta and Choudhary, a GitHub project) | Full text | Class weights; SMOTE-NC tested and worse |
| **`payer_code` / `medical_specialty`** | Strack drops payer and keeps specialty with "missing" | Full text | We keep both; keeping them gave +0.008 AUROC. Payer is a socioeconomic proxy, so the fairness trade-off is documented |
| **Uncertainty** | Rare in the papers above | Full text | Bootstrap CIs, patient-level resampling in the sensitivity analysis |
| **Metrics** | Accuracy, F1 or recall alone in several (the ICCDA study reports F1 0.81); Cureus's accuracy equals the always-no baseline (section 2b) | Full text for Cureus, abstracts for the rest | AUROC and AUPRC plus sensitivity/specificity at a stated threshold |

## 4. Detailed notes on the main comparators

### Strack et al. (2014): the dataset's origin
- Association study (logistic regression), not a prediction benchmark: HbA1c measurement and treatment change versus 30-day readmission.
- The public UCI file (101,766 rows) is the extract *before* the authors' exclusions; they reduced it to 69,984 first encounters without hospice/death.
- Read as the source of our cohort rules and our feature groupings (ICD-9 groups).

### Emi-Johnson and Nkrumah (2025): the closest published benchmark
- UCI data, hospice/expired excluded, stratified 80/20 split, class weights, no SMOTE. AUROC XGBoost 0.667, LogReg 0.642, RF 0.630, DNN 0.579.
- Our logistic regression (0.668) and XGBoost (0.672) are within 0.03 of theirs; our MLP (0.660) is far above their DNN (0.579). Differences plausibly come from patient-level splitting, kept specialty/payer and categorical handling.
- **Cite carefully (Siham's earlier finding, verified against the PDF):** their Table 1 does not reconcile with Tables 2 and 3. Table 1 gives XGBoost recall 0.84; Table 2 gives recall for the readmitted class as 0.017 (38 of about 2,280 caught). Their accuracy of 0.89 equals the always-no baseline. Cite the AUROC values only and state this when discussing their accuracy or recall.
- **We also noticed:** the paper says adults aged 18 and over, but the public file includes age bands 0-10 and 10-20, and no age filter is described.

### Chowdhury et al. (2026): an independent "ceiling" check
- Uses the identical cohort (99,343 / 69,990 / 0.1139) and finds gradient boosting reaches 0.6692 AUROC, close to its estimated ceiling for the recorded variables (a gap of about 0.002).
- Supports our central message: the limit is the information in the dataset, not the model. Very recent arXiv preprint, so verify before leaning on it heavily. We did not verify its fold construction.

### Gupta and Choudhary (2026): benchmarking with SMOTE done correctly
- Eight classifiers on this dataset; SMOTE on training only, test left at natural prevalence. Best ROC-AUC 0.6079 (Extra Trees), best balanced accuracy 53.35% (LogReg). "No classifier is superior on all measures."
- Lower than ours, plausibly because oversampling hurts (as in our ablation) and because of other preprocessing differences.

### Accuracy-only studies (Siham's earlier notes, kept for completeness)
- SVM with attribute selection: about 64.5% accuracy; random forest about 86.4% on a different task framing. Accuracy is not comparable with AUROC and is misleading on an 11% positive label (always predicting "no" scores about 89%), so we do not compare against these.

### EmbPred30 (2020): the outlier
- See [`embpred30_review.md`](embpred30_review.md). Treat as a cautionary example, not a target.

## 5. Sources checked

Every source we looked at, whether or not it was used. "How we checked" says how much of each we actually read. Links are to the paper or abstract. Copies of the open-licence ones are in [`papers/`](papers/); the rest are link-only because their licence does not allow us to redistribute (see [`papers/README.md`](papers/README.md)).

| # | Source | Year | Dataset / cohort | AUROC / headline | Accuracy and sensitivity reported | Split / leakage notes | How we checked | Link | Local copy |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Strack et al.: Impact of HbA1c Measurement on Hospital Readmission Rates | 2014 | UCI source data; 74M encounters / 17M patients in Health Facts; ~70,000 analysed | Logistic regression of 30-day readmission on HbA1c measurement (an association study, not a prediction benchmark) | n/a (association study) | First encounter per patient only; hospice/death removed -> 69,984 encounters; payer_code dropped; medical_specialty kept with a 'missing' value | Full text read | https://pmc.ncbi.nlm.nih.gov/articles/PMC3996476/ | papers/2014_strack_et_al_full_text_PMC3996476.html (CC BY 3.0) |
| 2 | UCI ML Repository, dataset 296: Diabetes 130-US Hospitals 1999-2008 | 2014 | 101,766 encounters, 50 columns, 130 hospitals | nan | n/a | Suggests a standard split, or a 3-way hold-out when selecting models; says nothing about patient grouping | Page opened | https://archive.ics.uci.edu/dataset/296/diabetes-130-us-hospitals-for-years-1999-2008 | link only |
| 3 | Emi-Johnson and Nkrumah: Predicting 30-Day Hospital Readmission in Patients With Diabetes Using ML on EHR Data | 2025 | UCI 101,766 encounters; hospice/expired excluded | AUROC: XGBoost 0.667, LogReg 0.642, RF 0.630, DNN 0.579 | Accuracy 0.84-0.89 (always-"no" scores about 0.889). Sensitivity for readmitted patients (their Table 2, class 1 recall): LogReg 0.016, RF 0.002, XGBoost 0.017, DNN 0.143. Their Table 1 'recall' of 0.84-0.89 is dominated by the majority class | Stratified 80/20 split with no stated patient grouping; class weights, no SMOTE; says adults >=18 although the public file has age bands under 18 and no filter is described | Full text read | https://pmc.ncbi.nlm.nih.gov/articles/PMC12085305/ | papers/2025_emi-johnson_nkrumah_cureus_readmission_diabetes.pdf (open access) |
| 4 | Sarthak, Shukla, Tripathi: EmbPred30 - Assessing 30-days Readmission for Diabetic Patients using Categorical Embeddings | 2020 | UCI; first encounter per patient only; death/hospice removed | AUROC 0.974, accuracy 95.2% (6-fold CV) | Accuracy 95.2%; sensitivity about 94.3% and specificity about 96.0% (read from Figure 3 by Siham). All on SMOTE-balanced data, where always-"no" would score 50%, so not comparable with natural-prevalence results | SMOTE to 1:1 is listed as a preprocessing step before the 6-fold CV and the text does not say it was confined to training folds; features chosen by importance after 70 epochs of training. See embpred30_review.md | Full text read | https://arxiv.org/abs/2002.11215 | link only (arXiv non-exclusive licence; Siham has her own copy) |
| 5 | Shang et al.: The 30-days hospital readmission risk in diabetic patients: predictive modeling with machine learning classifiers | 2021 | Health Facts, >100,000 records; 100,244 after preprocessing | Random forest had the highest AUC of three classifiers (RF, Naive Bayes, decision-tree ensemble); AUC value not seen | Not seen (abstract has no metrics) | Not seen (PDF blocked) | Abstract only (via Crossref) | https://doi.org/10.1186/s12911-021-01423-y | link only (PDF not retrievable here) |
| 6 | Gupta and Choudhary: Benchmarking Machine Learning Classifiers for Hospital Readmission Prediction in Diabetes Care | 2026 | UCI 101,766 records | Best ROC-AUC 0.6079 (Extra Trees; accuracy 86.54%); best balanced accuracy 53.35% (LogReg); no classifier best on every metric | Accuracy 86.54% for the best-AUC model (below the 88.9% always-"no" baseline); best recall 33.42% (AdaBoost); best precision 21.58% (RF); best balanced accuracy 53.35% (LogReg). Sensitivity of the best-AUC model not stated | SMOTE applied to the training set only; test set kept at natural prevalence | Abstract and methods read | https://geneticsmr.com/index.php/gmr/article/download/2444/1484/6655 | link only (licence not stated) |
| 7 | Albu, Gao, Wynants: missForestPredict - Missing data imputation for prediction settings | 2024 | Uses the UCI readmission data as one example (101,766 records) | Not a readmission benchmark; compares imputation methods | n/a | nan | Skimmed | https://arxiv.org/abs/2407.03379 | papers/2024_albu_gao_wynants_missForestPredict_arXiv-2407.03379.pdf (CC BY 4.0) |
| 8 | Chowdhury et al.: The Ceiling Is in the Channel - Auditing Learner Gaps and Measurement Frontiers in Clinical Prediction | 2026 | UCI cohort: 99,343 encounters, 69,990 patients, prevalence 0.1139 after removing death and hospice discharges | Cross-fitted gradient boosting AUROC 0.6692, balanced accuracy 0.6223; argues the ceiling is set by the recorded variables, not the learner | Balanced accuracy 0.6223 at the optimal threshold; accuracy and sensitivity not reported | Fold construction not verified by us | Relevant sections read | https://arxiv.org/abs/2609.01909 | papers/2026_chowdhury_et_al_ceiling_audit_arXiv-2609.01909.pdf (CC BY 4.0) |
| 9 | Nazyrova, Chaussalet, Chahed: ML models for predicting 30-day readmission of elderly patients using custom target encoding | 2022 | MIMIC-IV, patients over 65 (NOT the UCI dataset) | AUC 0.75, TPR 0.69, TNR 0.67 | TPR 0.69, TNR 0.67 (different dataset, MIMIC-IV); accuracy not reported | Hospice discharges excluded; target-encoding method | Skimmed | https://www.iccs-meeting.org/archive/iccs2022/papers/133520120.pdf | link only |
| 10 | Pathiravasan, Madushani, LaRosa: Improved Predictive Models for Readmission of Patients with Diabetes (CSoI report) | 2018 | UCI 101,766; death and hospice discharges removed | Base GLM AUC 0.6034 | AUC only; accuracy and sensitivity not reported | 60/40 train/test split | Read | https://compbio.soihub.org/site/assets/files/6113/annual-report-2018-diabetic-probability-team.pdf | link only |
| 11 | Borst et al.: Diabetes U.S. Hospitals from 1999-2008 (University of Oregon course project) | 2026 | UCI | None (descriptive); notes that the 101,766 rows are hospitalizations, not patients | n/a | nan | Read | https://netphi.uoregon.edu/sites/default/files/2026-03/Diabetes%20Group%20Project.pdf | link only |
| 12 | Diabetic patient readmission predictive analysis: a comparative study of machine learning models of hospital readmissions | 2024 | 1999-2008 data, 130 US facilities | Naive Bayes best F1 0.81 (F1 on an imbalanced label) | F1 0.81 (best); accuracy and sensitivity not seen | Not seen | Abstract only | https://pure.solent.ac.uk/en/publications/diabetic-patient-readmission-predictive-analysis-a-comparative-st/ | link only |
| 13 | NYC Data Science Academy: Hospital readmission prediction | 2019 | UCI | XGBoost reported with the highest AUC of the models compared (value not extracted) | AUC, sensitivity and lift mentioned; values not extracted | Not checked | Skimmed | https://nycdatascience.com/blog/student-works/hospital-readmission-prediction/ | link only |
| 14 | chaitra-satheesh/hospital-readmission-prediction (GitHub README) | n.d. | UCI; death and hospice removed; weight dropped | LogReg + RandomOverSampler: ROC-AUC 0.649, average precision 0.209 (threshold 0.509); CV ROC-AUC 0.646 | Recall 0.51 for readmitted at threshold 0.509, precision 0.18; accuracy not reported | 80/20 split; whether oversampling sits inside the CV was not checked | README read | https://github.com/chaitra-satheesh/hospital-readmission-prediction | link only |
| 15 | Fairlearn: Diabetes 130-Hospitals dataset documentation | 2023 | UCI | n/a (fairness-oriented description of the data) | n/a | nan | Page opened | https://fairlearn.org/v0.9/user_guide/datasets/diabetes_hospital_data.html | link only |
| 16 | imbalanced-learn documentation: Common pitfalls (resampling before splitting) | n.d. | nan | Warns that resampling the whole dataset before splitting leaks information | n/a | nan | Read | https://imbalanced-learn.org/stable/common_pitfalls.html | link only |
| 17 | angelmanzur/Diabetes_130Hospitals (GitHub) | n.d. | UCI; first encounter kept: 69,973 patients, 6,277 (9.0%) readmitted | Not seen | Not seen | Not seen | Search-result snippet only (page returned 403) | https://github.com/angelmanzur/Diabetes_130Hospitals | link only |
| 18 | Predicting Hospital Readmission of Diabetic Patients Using Machine Learning (Academia.edu) | n.d. | UCI | Not seen | Not seen | Not seen | Search result only, not opened | https://www.academia.edu/61152855/Predicting_Hospital_Readmission_of_Diabetic_Patients_Using_Machine_Learning | link only |
| 19 | Deep Learning vs Traditional Models (AbdelHai, Temple University) | n.d. | Appeared in search results for this dataset | Not seen | Not seen | Not seen | Search result only; host not reachable from our environment | https://dabi.temple.edu/external/zoran/papers/Deep_Learning_vs_Traditional_Models_AbdelHai_Final.pdf | link only |
| 20 | Predictive modeling of 30 day readmission risk of... (VU eprints) | n.d. | Appeared in search results for this dataset | Not seen | Not seen | Not seen | Search result only; host not reachable | https://vu-9.eprints-hosting.org/49252/1/predictive_modeling_of_30_day_readmission_risk_of.6.pdf | link only |
| 21 | R package 'readmission': Hospital Readmission Data for Patients with Diabetes | n.d. | Packaged copy of the dataset | nan | n/a | nan | Search result only, not opened | https://rdrr.io/cran/readmission/man/readmission.html | link only |

## 6. Worth analysing further (from this benchmarking)

Ordered by value for the report. Items marked **(running)** or **(done)** are tracked in [`../6_project_progress/PROJECT_STATUS.md`](../6_project_progress/PROJECT_STATUS.md).

1. **Reproduce the leakage mechanism behind the 0.974 claim (running).** Script 11 compares SMOTE-before-CV with SMOTE-in-fold and class weights on a first-encounter cohort like EmbPred30's. A large gap would explain the outlier and make a strong limitations paragraph.
2. **Learning curve / ceiling check.** The 2026 audit argues the ceiling is set by the variables. We can test this ourselves: train on 10%-100% of the data and see whether AUROC saturates near 0.68. If it does, more data or bigger models will not help.
3. **Strack's original finding.** Replicate the HbA1c result (measured and changed versus readmission) with an explainable logistic model on the first-encounter cohort, as an interpretability section and a link to the paper's clinical argument.
4. **Payer versus specialty.** Strack kept specialty and dropped payer. Test each alone; if specialty carries the gain, drop payer and avoid the socioeconomic-proxy concern at no cost.
5. **Threshold analysis at matched sensitivity.** The GitHub project tunes its threshold to recall >= 0.5; compare our models at the same sensitivity so precision is comparable.
6. **Leak-free entity-embedding network.** EmbPred30's idea (learned embeddings for categorical columns) is sound; test it under a correct patient-level protocol to see whether it beats one-hot encoding.
7. **Subgroup performance (race, age, sex) with the Fairlearn toolkit.** Not done by any source above; a distinct contribution.
8. **Check what the near-constant drug columns add** (the CSoI report and a GitHub project dropped zero-variance columns such as `examide` and `citoglipton`).
9. **Stronger deep-learning baselines (checked 2026-10-23, not yet run).** Candidates: RealMLP (Holzmüller et al., NeurIPS 2024, [arXiv 2407.04491](https://arxiv.org/abs/2407.04491), installable via `pytabkit`), TabM and FT-Transformer (Gorishniy et al.; TabM and FT-Transformer figures not verified by our searches), TabPFN-2.5 ([arXiv 2511.08667](https://arxiv.org/pdf/2511.08667v2); needs pretrained weights and has dataset-size limits). Background: tree models beat neural nets on medium-sized tabular data in [Grinsztajn et al. 2022](https://ar5iv.labs.arxiv.org/html/2207.08815), and [McElfresh et al.](https://arxiv.org/pdf/2305.02997v2) argue the gap depends on the dataset. Expect at most a small gain here because the 2026 audit puts the ceiling in the data.
10. **Read the sources we could only see in search results** (rows 17-21) before citing them; each reference must be read by a team member.

## 7. Notes on rigour

- Statements marked "full text read" were checked against the paper itself by us on 2026-10-23. The EmbPred30 review also includes Siham's independent assessment of her own copy of the PDF.
- Earlier versions of this file contained three attribution errors that are now corrected: (a) EmbPred30 *does* de-duplicate patients (first encounter only) and *does* exclude death/hospice; (b) the 99,343-encounter cohort belongs to the 2026 audit, not to missForestPredict; (c) the "no classifier best on every metric" SMOTE benchmark is Gupta and Choudhary, not the NYC Data Science Academy blog.
- Several sources are student projects, blogs or very recent preprints. Use them as context, not as authorities.
