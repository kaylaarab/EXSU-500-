# Literature benchmarks

Shared file for tracking published work relevant to our project (30-day
readmission prediction on the Diabetes 130-US Hospitals dataset, or closely
related problems). Goal: build the ≥15-reference list for the midterm, and
have real numbers to compare our results against in the manuscript.

**Everyone:** add your own findings in the "Team research log" section below
using the same format, so we can merge everything before the midterm. Don't
worry about duplicates — better to over-collect now and dedupe later.

---

## Our model results, for reference

*Updated 2026-10-23. Cleaned cohort (99,343 encounters; death/hospice discharges
excluded), patient-level split, test set scored once. Details and rationale:
`docs/dataset_analysis_report.md`.*

| Model | AUROC | AUPRC | Sensitivity | Specificity |
|---|---|---|---|---|
| Baseline (majority-class) | 0.500 | 0.111 | 0.000 | 1.000 |
| Logistic Regression | 0.663 | 0.212 | 0.528 | 0.689 |
| XGBoost (tuned on val) | 0.665 | 0.222 | 0.540 | 0.695 |
| MLP (deep learning) | 0.659 | 0.219 | 0.539 | 0.686 |
| CatBoost (engineered features) | **0.669** | 0.221 | 0.568 | 0.670 |
| Ensemble (XGBoost 0.7 + CatBoost 0.3) | **0.669** | 0.224 | 0.550 | 0.683 |

95% bootstrap CI (encounter-level): XGBoost 0.651-0.679; CatBoost/ensemble
0.655-0.682. **The models are statistically indistinguishable.** The earlier
first-pass figures (XGBoost 0.678, ensemble 0.679) were inflated by
death/hospice rows (cannot be readmitted) and a broken A1C feature.

Additional checks (Script 07-09; patient-level bootstrap CIs):

| Check | AUROC | Note |
|---|---|---|
| Reference CatBoost | 0.669 (0.653-0.686) | |
| + `medical_specialty`, `payer_code` kept | 0.677 (0.661-0.692) | paired gain +0.008 (CI +0.005 to +0.011) |
| First encounters only (trained and tested) | 0.652 (0.633-0.670) | the "no history" case; closest to published first-encounter studies |
| + patient-history features | 0.676 vs 0.670 base | paired gain +0.001 to +0.010, borderline |
| SMOTE-NC instead of class weights | LogReg 0.561, XGBoost 0.625 | clearly worse; see `docs/dataset_analysis_report.md` s6 |

---|---|---|---|---|
| Baseline (majority-class) | 0.500 | 0.110 | 0.000 | 1.000 |
| Logistic Regression | 0.643 | 0.204 | 0.517 | 0.678 |
| XGBoost (tuned) | **0.678** | 0.229 | 0.579 | 0.678 |
| MLP (deep learning) | 0.655 | 0.208 | 0.614 | 0.617 |

(95% bootstrap CI on XGBoost test AUROC: 0.663–0.690)

---

## Directly comparable benchmark (same dataset, same task)

### Emi-Johnson & Nkrumah (2025)
- **Citation:** Emi-Johnson OG, Nkrumah KJ. "Predicting 30-Day Hospital
  Readmission in Patients With Diabetes Using Machine Learning on Electronic
  Health Record Data." *Cureus*. 2025;17(4):e82437.
  doi:10.7759/cureus.82437
- **Dataset:** Same as ours — UCI Diabetes 130-US Hospitals, 101,766
  encounters, 80/20 train/test split, ~11.2% positive rate.
- **Task:** Same as ours — binary 30-day readmission.
- **Reported AUROC:** Logistic regression 0.642, Random forest 0.630,
  XGBoost 0.667, DNN 0.579.
- **Comparison to us:** Our logistic regression (0.663) and XGBoost
  (0.665) sit within about 0.02 of theirs (0.642 / 0.667); our MLP (0.659) is
  well above their DNN (0.579). Differences are plausibly due to cohort
  rules (we drop death/hospice rows) and split design (we split by patient;
  their 80/20 split is not described as patient-level) rather than model
  quality. All values fall in the same 0.64-0.68 band.
- **Caveat worth citing carefully:** their published Table 1 (headline
  metrics) is internally inconsistent with their own Table 2/3
  (confusion-matrix-derived metrics) — e.g. their XGBoost precision/recall
  in Table 1 don't reconcile with the false-negative count in Table 3. Cite
  their AUROC figures (which look consistent across tables) but flag the
  discrepancy if citing their precision/recall/F1 numbers.

## Other relevant findings (not directly comparable — different split, label
definition, or preprocessing, so don't present these as apples-to-apples)

### arXiv:2002.11215 (2020)
- Claims AUROC 0.974 / accuracy 95.2% using an embedding-based deep neural
  network on this same dataset.
- **Treat with skepticism, not as a benchmark to match.** This is far above
  every other result found (including our own and the Cureus paper above),
  and papers investigating this dataset specifically warn that class
  imbalance and data leakage (e.g. row-level splits that leak patients
  across train/test) produce exactly this kind of inflated number. Worth
  citing in the related-work/limitations discussion as a cautionary example
  of what happens without a patient-level split and proper evaluation
  protocol — which is directly relevant to why we were careful about this
  in our own methods.

### Other studies (accuracy-only, not AUROC — not directly comparable)
- SVM + attribute selection: ~64.5% accuracy (whole-cohort readmission)
- Random forest: ~86.4% accuracy (different task framing)
- Accuracy numbers are not directly comparable to AUROC and are of limited
  value on an imbalanced label anyway (a model predicting "no readmission"
  always would also score high "accuracy") — flagged here for completeness,
  not for direct comparison in our results section.

---

Kayla and Magalie's literature findings are tracked separately in
`LITERATURE_BENCHMARKS_TEAM.md` (same folder) — merge in before the midterm reference list
is finalized.

---

## Cohort and method comparison with other work (added 2026-10-23)

How our choices line up with what other projects on this dataset do. Source
snippets come from web searches (abstracts, not full texts), so check the
original paper before citing a specific detail.

| Practice | What others report | Us |
|---|---|---|
| Cohort rules | An analysis reports [99,343 encounters from 69,990 patients, 30-day prevalence 0.1139](https://arxiv.org/pdf/2407.03379) | **Same cohort** (99,343 / 69,990 / 11.4%) |
| First encounter per patient | A study kept [69,973 patients, 9.0% readmitted](https://github.com/angelmanzur/Diabetes_130Hospitals); Strack et al. also used first encounters (from memory of the paper, not confirmed by search) | Sensitivity run: 8.9% positive, AUROC 0.652 |
| Patient-level splitting | Careful work uses group-aware CV on `patient_nbr` and patient-level bootstrap | Patient-level split; patient-level bootstrap in Script 07 |
| Missing specialty/payer | Often kept as a "Missing" category | Dropped in main pipeline; keeping them gives +0.008 AUROC |
| Typical scores | [Cureus 2025](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12085305/): XGBoost 0.667, LogReg 0.642, RF 0.630. An [audit](https://arxiv.org/pdf/2609.01909) reports a ceiling near 0.669 (verify before citing; very recent) | 0.659-0.669 |
| Implausibly high scores | [EmbPred30](https://arxiv.org/pdf/2002.11215): 0.974 AUROC, no sign of patient-level deduplication | Cited only as a cautionary example |
| Oversampling | Many use SMOTE; a [benchmark](https://nycdatascience.com/blog/wp-json/wp/v2/posts/52818) found no classifier best on all measures | SMOTE-NC tested and rejected (worse on real test data) |

**Take-away for the report:** our cohort and score match the credible
published range; the studies reporting far more almost certainly leak
patient information across the split.

### Sources to read before citing (each reference must be read by a team member)
- Strack et al. (2014), *BioMed Research International*, doi:10.1155/2014/781670 (origin of the dataset)
- Emi-Johnson and Nkrumah (2025), *Cureus* 17(4):e82437 (direct benchmark)
- Dataset page: [UCI #296](https://archive.ics.uci.edu/dataset/296/diabetes-130-us-hospitals-for-years-1999-2008)
