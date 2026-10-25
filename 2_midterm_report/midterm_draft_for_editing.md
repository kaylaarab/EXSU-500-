# DRAFT text for the midterm report (due Mon 9 Nov 2026)

**How to use this file.** It is plain-text draft material built from our computed results and from the sources we checked, for the team to edit into the IEEE Word or LaTeX template (Kayla's `midterm_report.docx` holds the template). It is **not** the submission. Three rules before anything is submitted:

1. **Every reference must be read by a team member.** The "Read status" column in the reference list says what has actually been read so far. Anything marked "abstract only" or "not read" needs a human reader before it is cited. The assistant's summaries are screening aids only.
2. **Three IEEE pages including references.** The body is about 1,600 words plus three small tables and the reference list; that should fit when set in two columns, but check the page count in the template and trim or add a figure accordingly.
3. **Check every number** against `5_results/` before submission. Numbers below are from the final pipeline (2026-10-25).

Required content (spec section 6): problem and motivation; synthesised review; quantitative dataset description; evaluation protocol with splits and metrics; candidate models; work plan with role assignments and risks; CRediT and Generative-AI statements (these two do not count toward the page limit; drafts in `6_project_progress/statements.md`).

---

## Title
Predicting 30-Day Hospital Readmission in Patients with Diabetes: A Patient-Level, Leakage-Aware Comparison of Machine-Learning Models

**Authors:** Kayla Arab, Siham Khaldoun, Magalie Achim, Larissa Chiu. McGill University, Faculty of Medicine and Health Sciences.

## Abstract (draft, about 150 words)
Hospital readmission within 30 days of discharge is a standard quality measure and a major cost, and patients with diabetes are at elevated risk [1], [2]. We predict 30-day readmission for diabetic inpatients from information available at discharge, using the public Diabetes 130-US Hospitals data (99,343 stays, 11.4% readmitted). We split by patient to prevent leakage, compare a baseline, logistic regression, gradient boosting, deep learning and an ensemble, and report AUROC and AUPRC with accuracy, sensitivity and specificity at a stated threshold. In preliminary experiments all models reach AUROC 0.66-0.68, consistent with published work, and a standard clinical score (LACE) reaches 0.58. We show that reported scores near 0.97 on this dataset arise from evaluation order, not model skill. We describe the plan to complete calibration, subgroup auditing and final reporting.

## I. Introduction and clinical significance
Readmission within 30 days of discharge signals possible gaps in discharge planning, follow-up and medication management, and it is costly [1], [2]. In the United States it is the basis of a Medicare payment program, although the program's penalised conditions do not include diabetes itself [3]. Patients with diabetes are particularly vulnerable: reported readmission rates are substantially higher than for patients without diabetes, driven by comorbidity burden, previous hospitalisations and the complexity of transitions from inpatient to outpatient care [1], [2], [4]. A tool that ranks patients by readmission risk at discharge could help target scarce follow-up resources, but only if it is evaluated honestly. Our contribution is a careful comparison on a public dataset, with attention to the evaluation pitfalls that most often invalidate medical machine-learning results.

**Task.** Binary classification: will a diabetic inpatient be readmitted within 30 days of discharge (1) or not (0), using information known at discharge. The 30-day window is the standard quality measure and the one used by the dataset's source study [5].

## II. Related work (synthesised)
**What others report.** On this dataset, credible studies converge on modest discrimination. The source study used one encounter per patient and found readmission associated with HbA1c measurement and care changes [5]. A 2025 comparison of logistic regression, random forest, XGBoost and a neural network reported AUROC 0.64, 0.63, 0.67 and 0.58 [6]; a 2026 stacking ensemble with nested cross-validation and calibration reported 0.66-0.69 [7]; a benchmark of eight classifiers with SMOTE applied to training data only reported a best ROC-AUC of 0.61 [8]. An independent 2026 audit on the same cohort found gradient boosting at 0.669 and argued that the ceiling is set by the recorded variables, not the learner [9]. In other cohorts, diabetes-specific models reach 0.64-0.68 [10] and deep sequence models reach 0.79 when millions of encounters with longitudinal laboratory data are available [11].

**What the standard clinical scores do.** The LACE index (length of stay, acuity, comorbidity, emergency visits) and the HOSPITAL score are widely validated, with AUROC of 0.73 and 0.69 respectively in one comparison of 291,886 encounters [13], and machine-learning models can improve on them [12]. We therefore include LACE as a benchmark.

**Why some reported numbers should not be trusted.** One preprint reports 0.974 AUROC and 95.2% accuracy on this dataset [14]. Its pre-processing keeps one encounter per patient, so repeated-patient leakage does not explain it, but it applies SMOTE to a 1:1 ratio before six-fold cross-validation and does not state that oversampling was confined to training folds. Oversampling the whole dataset before splitting is a known leakage pitfall. The closest published benchmark reports accuracy of about 0.89, equal to the majority-class rate, while its own class-wise table shows only 0.2-14% of readmitted patients detected [6]. When one of our experiments applies SMOTE before cross-validation, a standard gradient-boosting model reaches 0.96 AUROC; the same model on real held-out patients reaches 0.64.

**Fairness.** Readmission models can perform differently across groups; evaluations of LACE, HOSPITAL and CMS measures in over ten million discharges show that fairness metrics can reveal disparities in false-negative and false-positive rates but cannot explain them [16], and multistage bias audits show a fairness-calibration-discrimination trade-off [17]. We therefore plan an audit by race, sex and age.

**Gap.** Few studies on this dataset combine a patient-level split, calibration, uncertainty, subgroup performance and a clinical benchmark. This project does.

## III. Data and feasibility
**Dataset.** UCI *Diabetes 130-US Hospitals for Years 1999-2008* [5]: 101,766 inpatient encounters, 50 columns, 71,518 patients, 130 hospitals, CC BY 4.0, freely downloadable. Patients who died or went to hospice (2,423 stays, 2.4%) cannot be readmitted and were excluded, leaving **99,343 stays from 69,990 patients; 11.4% were readmitted within 30 days** (about 89% are not, so accuracy alone is misleading).

**Quality and limits.** Almost all core fields are complete. Mostly-missing columns: weight 96.9%, glucose result 94.8% and HbA1c 83.3% (mostly "not measured"), medical specialty 49.1%, payer 39.6%, race 2.2%. 23.5% of patients have more than one stay and account for 46% of rows (up to 40 stays), so rows are not independent. The population is 76.5% White, 19.3% Black, 2.0% Hispanic, 1.5% other, 0.6% Asian, and 45% are aged 70 or over. Data are from 1999-2008 and ICD-9 coded, and contain no dates, hospital identifiers, vital signs, laboratory values, social factors or notes. Positive rate differs little by sex or race (9.6-11.3%).

**Features.** Demographics, admission and discharge details, length of stay, counts of laboratory tests, procedures, medications and diagnoses, prior-year visits, HbA1c and glucose results, 23 drug fields, three ICD-9 diagnosis groups (nine categories) and three derived features (total prior visits, number of medication changes, HbA1c measured and treatment changed).

**Compute.** Everything runs on CPU in minutes to under an hour per experiment; no special hardware is needed.

## IV. Evaluation protocol
- **Splits, fixed before modelling and made at the patient level:** train 70% (69,519 stays, 48,993 patients), validation 15% (14,911; 10,498), test 15% (14,913; 10,499). No patient appears in more than one set (checked in code).
- **Tune on validation, report on test.** Hyperparameters, thresholds and calibration use validation data only; the test set is scored once.
- **Metrics.** Primary: AUROC and AUPRC (baseline 0.111). At a stated threshold: **accuracy, sensitivity and specificity, always reported together**, because the always-"not readmitted" model scores 88.9% accuracy with 0% sensitivity.
- **Uncertainty.** 95% bootstrap confidence intervals resampling patients, and paired intervals for model differences.
- **Calibration and interpretability.** Reliability curve and Brier score; SHAP feature attributions; decision-curve analysis.
- **Class imbalance.** Class weighting. Oversampling (SMOTE-NC) was tested and was clearly worse on real held-out patients.
- **Subgroups.** Performance by race, sex and age with confidence intervals.

## V. Candidate models and preliminary results
Baseline (majority class); regularised logistic regression; XGBoost and CatBoost gradient boosting; multilayer perceptron (also tuned, with entity embeddings, and with stronger tabular architectures); validation-weighted ensemble; LACE score as a clinical benchmark. The choice of families follows evidence that boosted trees remain strong on medium-sized tabular data, with deep models included to test that on our data [18].

**TABLE I. Preliminary test results** (14,913 stays; always-"no" accuracy 0.889)

| Model | AUROC | AUPRC | Accuracy | Sensitivity | Specificity |
|---|---|---|---|---|---|
| Logistic regression | 0.668 | 0.214 | 0.670 | 0.547 | 0.685 |
| XGBoost | 0.672 | 0.226 | 0.682 | 0.539 | 0.700 |
| MLP (tuned, 3 seeds) | 0.678 | 0.231 | 0.630 | 0.619 | 0.631 |
| CatBoost | 0.677 | 0.228 | 0.671 | 0.555 | 0.685 |
| LACE score alone | 0.575 | 0.139 | 0.642 | 0.435 | 0.668 |

**Findings so far.** (1) All models are within about 0.015 AUROC of each other; differences are inside the confidence intervals. (2) Machine learning adds about 0.10 AUROC over the LACE score. (3) Real gains came from data handling (keeping payer and specialty, +0.008; treating ID codes as categories; patient-history features, +0.007), not from model choice. (4) Calibrated probabilities are reliable (Brier 0.094, expected calibration error 0.007). (5) A sensitivity gap by sex (women 0.62, men 0.52) needs investigation.

## VI. Work plan, roles and risks
| Period | Work | Lead (to confirm) |
|---|---|---|
| to 9 Nov | midterm report, 15+ read references | writing lead; all |
| 10-24 Nov | finish model comparison; sex-gap investigation; learning curve | modelling lead |
| 25 Nov-3 Dec | external-style checks, interface polish, model card, statements | interface lead; data lead |
| 4-10 Dec | manuscript (6-8 pages), 7-minute video, peer evaluation | all |
| 11 Dec | submit; repository tidy | all |

Roles required by the specification (data, modelling, interface, writing lead) are **to be assigned by the team**.

**Risks.** (a) Information ceiling: performance cannot exceed about 0.68 AUROC with these variables, so the contribution is rigour, not accuracy. (b) Data age and one-country origin limit generalisability; no external validation set is available. (c) Small subgroups cannot be assessed reliably. (d) Time: contributions from all members and verified references are needed before the deadlines. (e) Misreading of metrics: mitigated by always reporting sensitivity with accuracy.

---

## References (draft; verify format against IEEE style and READ STATUS before citing)

| # | Reference | Read status (2026-10-25) |
|---|---|---|
| [1] | American Diabetes Association, "16. Diabetes Care in the Hospital: Standards of Care in Diabetes-2026," *Diabetes Care*, vol. 49, suppl. 1, pp. S339-S355, 2026. | Kayla's list; **human reader to confirm** |
| [2] | D. J. Rubin et al., "The DiaTOHC pilot study...," *J. Clin. Med.*, vol. 11, no. 6, 1471, 2022. | Kayla's list; **human reader to confirm** |
| [3] | Centers for Medicare & Medicaid Services, "Hospital Readmissions Reduction Program." | web page; the assistant saw it via search only; **confirm current condition list and penalty details** |
| [4] | J. Tan et al., "Derivation and validation of prediction models for prolonged length of stay and 30-day readmission in elderly patients with type 2 diabetes," *J. Diabetes Res.*, 2025. | Kayla's list; **human reader to confirm** |
| [5] | B. Strack et al., "Impact of HbA1c measurement on hospital readmission rates: analysis of 70,000 clinical database patient records," *BioMed Res. Int.*, 2014, doi:10.1155/2014/781670. | full text read (assistant); **human reader to confirm** |
| [6] | O. G. Emi-Johnson and K. J. Nkrumah, "Predicting 30-day hospital readmission in patients with diabetes using machine learning on EHR data," *Cureus*, vol. 17, no. 4, e82437, 2025. | full text read (assistant); **human reader to confirm** |
| [7] | S. S. Salim et al., "A machine learning approach for predicting 30-day hospital readmission in patients with diabetes," *Healthcare*, 2026. | abstract only |
| [8] | A. K. Gupta and A. Choudhary, "Benchmarking machine learning classifiers for hospital readmission prediction in diabetes care," *Genet. Mol. Res.*, vol. 25, no. 11s, 2026. | abstract and methods read (assistant) |
| [9] | S. S. Chowdhury et al., "The ceiling is in the channel: auditing learner gaps and measurement frontiers in clinical prediction," arXiv:2609.01909, 2026. | relevant sections read (assistant); recent preprint, verify |
| [10] | J. G. S. Soh et al., "Predicting and validating 30-day hospital readmission in adults with diabetes whose index admission is diabetes-related," *J. Clin. Endocrinol. Metab.*, 2022. | abstract only |
| [11] | A. A. Hai et al., "Deep learning vs traditional models for predicting hospital readmission among patients with diabetes," *AMIA Annu. Symp. Proc.*, 2022. | abstract only |
| [12] | S. Davis et al., "Effective hospital readmission prediction models using machine-learned features," *BMC Health Serv. Res.*, 2022. | abstract only |
| [13] | S. L. Jones et al., "Comparison of LACE and HOSPITAL readmission risk scores for CMS target and nontarget conditions," *Am. J. Med. Qual.*, 2022. | abstract only |
| [14] | Sarthak, S. Shukla, S. P. Tripathi, "EmbPred30: assessing 30-days readmission for diabetic patients using categorical embeddings," arXiv:2002.11215, 2020. | full text read (assistant); preprint |
| [16] | H. E. Wang et al., "Evaluating algorithmic bias in 30-day hospital readmission models," *J. Med. Internet Res.*, 2024. | abstract only |
| [17] | R. Mateedulsatit et al., "Detecting and mitigating AI bias in health care...," *JMIR AI*, 2026. | abstract only |
| [18] | L. Grinsztajn, E. Oyallon, G. Varoquaux, "Why do tree-based models still outperform deep learning on typical tabular data?" *NeurIPS*, 2022. | seen in search results only; **to read** |
| [19] | C. van Walraven et al., "Derivation and validation of an index to predict early death or unplanned readmission after discharge from hospital to the community," *CMAJ*, 2010 (the LACE index). | **not read; cite from the original paper** |

**Counts.** 18 entries (reference [15] was removed; renumber when finalising). The assistant read 5 in full or in part ([5], [6], [8], [9], [14]) and 7 as abstracts only ([7], [10]-[13], [16], [17]); the rest are Kayla's list, a web page, a search hit and a classic paper not yet read. **None has yet been read by a human team member.** The midterm needs at least 15 references, mostly primary research from the last five years, each read by a team member. Larissa's reading list (`4_benchmarking/LARISSA/`) and the 74 PubMed candidates (`4_benchmarking/pubmed_candidates.md`) are the pool to fill the gap; the summaries there are abstract-based.

## Statements (not counted in the page limit)
- CRediT contribution statement: draft in `6_project_progress/statements.md`; each member to confirm.
- Use of Generative AI statement: draft in `6_project_progress/statements.md`; each member to confirm.
