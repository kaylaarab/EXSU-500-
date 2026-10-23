# Where we stand: progress report

For the team, not for submission. Plain language, so everyone can follow the
state of the project without reading code.
**Last updated:** 2026-10-23. Details and reasoning behind every decision:
[`dataset_analysis_report.md`](dataset_analysis_report.md). Literature and
numbers to compare against: [`literature/LITERATURE_BENCHMARKS.md`](../literature/LITERATURE_BENCHMARKS.md).

## Key dates

| Date | What | Weight |
|---|---|---|
| **Mon 9 Nov** | **Midterm report** (3 pages IEEE, 15+ references, PDF + Word/LaTeX source) | 20% |
| Thu 3 Dec | Course wrap-up (Q&A only) | n/a |
| Fri 11 Dec | Final deliverables: manuscript (6-8 pages), 7-minute video, repo URL, peer evaluation | 45% |
| Fri 11 Dec | Optional bonuses: public repo (+3%), public interface (+2%) | up to +5% |

## The problem in one sentence
Can we predict which diabetic patients will be readmitted within 30 days of
discharge, using only what is known at discharge? (Dataset: Diabetes 130-US
Hospitals, 99,343 usable hospital stays, 11.4% readmitted.)

## Where the results are now

| Model | AUROC* | What it is |
|---|---|---|
| Baseline | 0.500 | Always says "not readmitted" |
| Logistic regression | 0.663 | Simple classic model |
| XGBoost | 0.665 | Boosted decision trees |
| MLP (deep learning) | 0.659 | Small neural network |
| CatBoost | **0.669** | Boosted trees, handles categories natively |
| Ensemble | **0.669** | Blend of XGBoost + CatBoost |

*AUROC: 0.5 = coin flip, 1.0 = perfect. All models are **statistically tied**
(95% intervals roughly 0.65-0.69). Published studies on this dataset land in
the same 0.64-0.68 range, so we are where a correct analysis should be.

**What that means in practice:** about 1 in 6 patients the model flags is
really readmitted, and about 45% of real readmissions are missed. Good enough
to rank risk and write a solid report; not good enough for clinical use. The
report must say so.

## What we did since the last update
- **Found and fixed three real problems** in our own first-pass pipeline:
  1. A bug made the "A1C measured and changed" feature wrong (the text "None"
     was read as missing).
  2. Patients who died or went to hospice (2.4% of stays) can't be readmitted
     but were included; removing them lowers scores slightly (the honest number).
  3. Admission/discharge codes were treated as numbers; now categories.
- **Re-ran every model** on the cleaned data. First-pass scores (e.g. 0.679)
  were inflated; the table above replaces them.
- **Profiled the real dataset** (`scripts/00_profile.py`): what's missing, repeat
  patients, subgroups. Tables in `results/data_profile/`.
- **Sensitivity checks** (Scripts 07-09):
  - Keeping `medical_specialty` and `payer_code` helps: **+0.008 AUROC** (reliable).
  - First-visit patients only: 0.652 (harder, no history).
  - Patient-history features (stay number, previous outcome): about +0.006 (borderline).
  - SMOTE (synthetic oversampling) is **clearly worse** than class weights.
- **Wrote the decision log** (what we did, why, cost) in `dataset_analysis_report.md`.
- **Pinned package versions** in `requirements.txt`; **removed the raw data and
  split CSVs from the repo** (course rule) and git-ignored them.

## Settled decisions (use these in the report)
- Predict **per stay**, split by **patient** (no patient in two sets).
- Binary label: readmitted within 30 days.
- Exclude death/hospice discharges.
- Class weights, **not** synthetic data.
- Tune on validation, score the test set once; report AUROC + AUPRC with
  patient-level confidence intervals.

## Still to do

**Code / analysis (Siham, with Claude):**
- [ ] Use specialty + payer in the main pipeline and rerun 02-06 (expected +0.008).
- [ ] Subgroup performance (race, age, sex) with confidence intervals.
- [ ] Calibration curve; pick a threshold on validation instead of 0.5.
- [ ] Decide whether history features join the headline model.
- [ ] Model card, open-source license, CRediT contribution statement,
      Generative-AI use statement (all required in the repo).

**Midterm report (everyone; biggest risk, due 9 Nov):**
- [ ] **15+ references**, mostly primary papers from the last 5 years. Each must
      be read by at least one team member (the instructors ask about specific
      citations). Currently only about 3 are logged; Kayla's and Magalie's
      sections in `literature/LITERATURE_BENCHMARKS_TEAM.md` are still empty.
- [ ] The current draft (`report/midterm_report.docx`) still contains IEEE template
      boilerplate and placeholder text (e.g. "Hygftbhjgftgbh"); remove it, keep
      to **3 pages including references**, and replace inline URLs with numbered references.
- [ ] Add: dataset description with numbers (use `dataset_analysis_report.md`),
      evaluation plan (splits, metrics), candidate models, work plan with the
      four roles (data, modelling, interface, writing lead) and risks.
- [ ] Required statements: CRediT contributions and Generative-AI use.

**Team / repo hygiene:**
- [ ] Confirm `@AHooshiar` and the TAs have read access to the repo.
- [ ] Everyone commits under their own account; commit history is graded and
      compared with the contribution statement. Only Siham and Kayla appear so far.
- [ ] Decide on the optional interface (bonus +2%).

## Where to find things
```
README.md                         formal project description (for grading)
docs/PROJECT_STATUS.md            this file
docs/dataset_analysis_report.md   dataset, decisions, trade-offs, bias, results
docs/CONTRIBUTING.md              how to use git here
literature/LITERATURE_BENCHMARKS.md       Siham's literature + comparison with other work
literature/LITERATURE_BENCHMARKS_TEAM.md  Kayla's and Magalie's literature log
report/midterm_report.docx        midterm draft (due 9 Nov)
course/                           specification, dataset list, group assignments (PDFs)
scripts/00-09_*.py                profile, preprocess, models, SHAP, ablations
results/                          all tables and figures (model_comparison.csv is the headline)
```
Run order is in `dataset_analysis_report.md` section 10. The dataset itself is
not in the repo; download it from UCI (instructions in README).
