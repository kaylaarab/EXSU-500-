# Progress log

Newest first. One entry per significant piece of work, added when it is finished. The current summary, to-dos and key dates are in [`PROJECT_STATUS.md`](PROJECT_STATUS.md).

## 2026-10-23

- **Repo reorganised into numbered topic folders** (`1_course_materials`, `2_midterm_report`, `3_dataset`, `4_benchmarking`, `5_results`, `6_project_progress`, plus `scripts/`). Both repos mirror the same branch.
- **Benchmarking folder rebuilt.** New `benchmarking_report.md` with a 21-source table (links, how much we read, local copies), `sources.csv`, abstracts, and four open-licence papers downloaded. Reading the papers corrected three errors in our earlier notes: EmbPred30 *does* keep one encounter per patient and *does* exclude death/hospice; the 99,343-encounter cohort belongs to the 2026 ceiling audit; the SMOTE benchmark is Gupta and Choudhary 2026.
- **EmbPred30 (0.974 AUROC) review added** (`4_benchmarking/embpred30_review.md`): Siham's assessment, our check against the PDF text, and our own experiment (Script 11).
- **Death/hospice ablation (Script 10).** Training with or without those rows gives the same score on a common test set (paired AUROC difference −0.0003, CI −0.003 to +0.002). Exclusion is justified for validity and comparability, not for score; the 0.005 "inflation" only appears when scoring the easy rows. Dataset report updated.
- **Specialty and payer adopted in the main pipeline.** All models retrained: CatBoost 0.677, ensemble 0.676, XGBoost 0.672, LogReg 0.668, MLP 0.660 AUROC. SHAP refreshed.
- **SMOTE ablation finished (Script 09).** SMOTE-NC is reliably worse than class weights (−0.02 to −0.10 AUROC). A control with plain random oversampling matched class weights, so the pipeline is sound.
- **Sensitivity analyses finished (Scripts 07, 08).** Keeping specialty+payer: +0.008 AUROC. First-encounter-only: 0.652. Patient-history features: about +0.006 (borderline).
- **Dataset analysis report written** (`3_dataset/dataset_analysis_report.md`): data quality, 10-decision log with trade-offs, encounter vs patient level, why no synthetic data, bias, open items.
- **Fixed three problems in our first-pass pipeline:** the A1C feature bug ("None" read as missing), admission/discharge IDs treated as numbers, and unexcluded death/hospice rows. Raw data and split CSVs removed from the repo (course rule) and git-ignored.
- **Entity-embedding / tuning experiment (Script 12) written and queued.**

## Earlier (2026-10-15 to 2026-10-20)

- Baseline, logistic regression and XGBoost pipeline with patient-level split and bootstrap CI (Siham).
- SHAP interpretability, MLP, ICD-9 grouping, CatBoost, ensemble (Siham).
- First midterm report draft (Kayla, Oct 17-18); course PDFs added (Oct 20).
