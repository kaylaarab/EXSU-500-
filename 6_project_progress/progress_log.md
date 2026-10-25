# Progress log

Newest first. One entry per significant piece of work, added when it is finished. The current summary, to-dos and key dates are in [`PROJECT_STATUS.md`](PROJECT_STATUS.md).

## 2026-10-25

- **Stronger deep models finished (Script 16).** RealMLP 0.669 (reliably -0.008 vs CatBoost), TabM 0.674 (-0.003, not reliable; best AUPRC 0.235), FT-Transformer 0.675 (-0.002, not reliable); CatBoost 0.677. No deep model wins; the model family does not matter here. Training was shortened (TabM 8 ensemble members, FT-T 15 epochs). **Operational finding:** the cloud machine is suspended and rebooted when the session is idle, which kills background jobs; long jobs only progress while the session is active, so they are checkpointed and watched in active waits.

- **Reproducibility check passed.** A fresh clone of the GitHub repo, with only the UCI CSV added, rebuilds splits that are byte-identical to ours (same MD5 for train, validation and test). `requirements.txt` resolves. `8_final_deliverables/` added (7-minute presentation script draft, manuscript outline).

- **Required repo items drafted:** `MODEL_CARD.md`, `LICENSE` (MIT, proposed: team to confirm), `6_project_progress/statements.md` (CRediT and Generative-AI drafts; each member must complete their own lines). **Midterm draft text** for every required section in `2_midterm_report/midterm_draft_for_editing.md`, with a table showing which references have actually been read (none yet by a human) and a rubric checklist.
- **Interface upgraded:** example-patient buttons (lower-risk 5.1% / higher-risk 27.3%), calibration and decision-curve figure and subgroup audit table in the results tab; still private and undeployed by decision of the team.

- **Calibration, decision curves and subgroup audit (Script 18, `5_results/evaluation/`).** Brier 0.094 (raw class-weighted score 0.219), ECE 0.007, slope 0.96: well calibrated after Platt scaling. Model has higher net benefit than flag-everyone or flag-no-one at all thresholds 2-40%, modestly. Audit: sensitivity 0.62 for women vs 0.52 for men (non-overlapping CIs); no reliable race difference between the two large groups; AUROC lower in patients 70+ (0.64) than under 50 (0.72); small race groups cannot be assessed.

- **Demo interface built and tested (`7_interface/`, Script 17).** Streamlit app with the required prototype notice, a risk calculator (calibrated %, risk band, observed rate per band on test patients: 5.6% / 11.5% / 21.8%, SHAP explanation), a results tab, input checks, and refusal to score patients who died or went to hospice. Tested with Streamlit's test harness (default, high-risk and excluded cases, no exceptions). Not deployed: public deployment is the team's decision (steps in `7_interface/README.md`).

- **Tuning and embeddings finished (Script 12).** Tuned one-hot MLP 0.674 per seed (0.678 averaged, accuracy 0.630, sensitivity 0.619); entity embeddings 0.668 (no gain, -0.005); wider CatBoost search 0.676 (no gain). Tuning brings the neural net level with CatBoost; nothing beats it. The 5-hour machine-restart problem is handled by checkpointing.
- **Script 16 (RealMLP, TabM, FT-Transformer) running;** demo interface under construction (`7_interface/`).

## 2026-10-24

- **Larissa's reading folder** (`4_benchmarking/LARISSA/`): 22 papers from PubMed and PhysioNet-based (MIMIC) studies, with abstract-based summaries, links, licence-compliant local copies (14 files stored; the rest link-only), the PhysioNet dataset access table (MIMIC-IV, eICU, MIMIC-IV-Note are credentialed and not allowed for the project; the MIMIC-IV demo is open) and rules for PhysioNet material. 74 PubMed candidates are in `pubmed_candidates.md`.

- **LACE benchmark and comorbidity features (Script 15).** LACE alone: AUROC 0.575 (our model 0.677), conventional cut-off catches 27% of readmissions. Adding LACE, Charlson score and flags, or 3-digit diagnoses to CatBoost: no gain (changes between -0.0001 and +0.0004). Negative result; the model already uses this information.
- **EmbPred30 test finished (Script 11, with sensitivity):** SMOTE before CV scores AUROC 0.964 / accuracy 94.2% / sensitivity 89%; on real patients only, 0.640 with sensitivity 3%.

- **Consolidated experiment log created** (`5_results/model_variations_log.md` and `.csv`, built by `scripts/build_experiment_log.py`): all 63 model variations in order, with AUROC, AUPRC, accuracy, sensitivity, specificity and the take-away for each, plus the hyperparameter searches. Rerun the script after each new experiment.
- **Results moved into `5_results/`** (`main_models/`, `ablations/`) and the data profile into `3_dataset/profile_tables/`; all scripts and docs updated.
- **Machine restarted overnight**, killing three background runs (first-encounter combined-target test, SMOTE-leakage rerun with sensitivity, embedding/tuning experiment); relaunched in sequence.

## 2026-10-23

- **Target definitions compared (Script 13).** Within-30-days vs rest: AUROC 0.675; any readmission vs none: 0.696; within 30 days vs never (dropping later readmissions): 0.728. Higher scores come from an easier question (T3 is not a usable task), so the 30-day target stays the headline. Also corrected an overstatement: CMS penalizes hospitals for 30-day readmissions of six conditions, not diabetes itself.
- **SMOTE-before-CV test (Script 11).** First run: AUROC 0.964 and accuracy 94.2% when SMOTE is applied before 6-fold CV and synthetic rows are scored, vs 0.640 on real rows only and 0.614 with SMOTE kept inside training folds. Reproduces EmbPred30's claim via the protocol alone. Rerun with sensitivity in progress.

- **Patient-history features re-tested on the current pipeline (Script 08):** 0.675 to 0.682 AUROC, paired gain +0.007 (CI +0.003 to +0.011), now reliably positive. Depends on `encounter_id` order being chronological (unverifiable).
- **Accuracy always reported next to sensitivity** (team rule): scripts 02-06, 09, 12 and every results table updated; benchmarking now shows accuracy and sensitivity for each published source. Key finding: the Cureus 2025 benchmark's 0.89 accuracy equals the always-"no" baseline and it catches only 1.7% of readmissions with XGBoost.

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
