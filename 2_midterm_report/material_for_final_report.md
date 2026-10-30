# Material for the Final Manuscript (not in the midterm report)

Removed from `Midterm_Report_CLAUDEgenerated.docx` on 2026-10-30 because it is
not required by the midterm rubric (problem statement, literature synthesis,
dataset description, evaluation protocol, candidate models, work plan/roles/
risks — no results section is scored at midterm). Everything below is
finished work, kept here so it drops straight into the final manuscript
(due Dec 11; 6-8 pages IEEE, needs ≥2 models, calibration/interpretability
artifact, variability quantified, repo statements).

## Table I. Preliminary Test Results (14,913 stays; always-"no" accuracy = 0.889)

| Model | AUROC | AUPRC | Acc. | Sens. | Spec. |
|---|---|---|---|---|---|
| Baseline (majority) | 0.500 | 0.111 | 0.889 | 0.000 | 1.000 |
| Logistic Regression | 0.668 | 0.214 | 0.670 | 0.547 | 0.685 |
| XGBoost | 0.672 | 0.226 | 0.682 | 0.539 | 0.700 |
| CatBoost | 0.677 | 0.228 | 0.671 | 0.555 | 0.685 |
| MLP (deep learning) | 0.668 | 0.222 | 0.616 | 0.617 | 0.616 |
| Ensemble (blend) | 0.676 | 0.228 | 0.674 | 0.560 | 0.688 |
| LACE score alone | 0.575 | 0.139 | 0.642 | 0.435 | 0.668 |

## Findings so far

1. All models are within about 0.015 AUROC of each other; pairwise differences
   fall inside bootstrap confidence intervals.
2. Machine learning adds roughly 0.10 AUROC over the LACE clinical score alone.
3. Real gains came from data handling — keeping payer code and medical
   specialty (+0.008 AUROC), encoding administrative ID fields as categories
   rather than numbers, and adding patient-history features (+0.007) — not
   from model choice.
4. Calibrated probabilities are reliable (Brier score 0.094, expected
   calibration error 0.007).
5. **Sensitivity-by-sex subgroup analysis (resolved, see below).**

## Sensitivity by sex — full analysis

Ran on the test set (8,268 female / 6,949 male / 2 unknown-invalid rows),
1000 bootstrap resamples per subgroup. Full numbers in
`5_results/fairness/sensitivity_by_sex.csv`, script in
`scripts/07_sensitivity_by_sex.py`.

| Model | Female Sens. (95% CI) | Male Sens. (95% CI) | Gap (F-M) |
|---|---|---|---|
| Logistic Regression | 0.540 [0.508, 0.572] | 0.492 [0.457, 0.528] | +0.048 |
| XGBoost | 0.603 [0.572, 0.637] | 0.566 [0.529, 0.601] | +0.038 |
| CatBoost | 0.600 [0.568, 0.632] | 0.555 [0.519, 0.589] | +0.046 |
| Ensemble | 0.603 [0.572, 0.635] | 0.555 [0.517, 0.589] | +0.049 |

**Conclusion:** Sensitivity differs by sex across all four models (female
higher than male by 0.04-0.05), but 95% CIs overlap in every case, so the
gap is not statistically significant at current sample sizes (732-948
positive cases per subgroup). The consistent direction across model
families is worth monitoring if the model is revisited, but does not
support a claim of differential performance by sex at this stage.
Specificity runs the opposite direction (~0.65 female vs. ~0.69 male,
smaller margin) in every model.

Suggested sentence for the final manuscript's Limitations/Fairness section:
> "Sensitivity differed by sex across all four models (female higher than
> male by 0.04-0.05), but 95% bootstrap confidence intervals for each
> subgroup overlap in every case, so this difference is not statistically
> significant at current sample sizes (n ≈ 7-8k per subgroup, 732-948
> positive cases). The consistent direction across model families warrants
> monitoring if the dataset or model is revisited, but does not support a
> claim of differential model performance by sex at this stage."

## Fuller Risks paragraph (midterm version was trimmed)

(a) Limited information: performance is held to roughly 0.68-0.70 AUROC by
the recorded variables (a learning curve shows only about +0.005 AUROC per
doubling of training data for boosted trees, and none for simpler models),
so the project's contribution is methodological rigour, not raw accuracy.
(b) Data age and single-country origin limit generalisability; no external
validation set is available. (c) Small subgroups cannot be assessed
reliably. (d) Time: verified contributions and read references from every
member are needed before the deadlines. (e) Metric misreading: mitigated by
always reporting sensitivity alongside accuracy.

## Draft CRediT contribution statement

Conceptualization — S. Khaldoun, K. Arab; Data curation, Methodology,
Software, Validation, Formal analysis, Investigation — S. Khaldoun;
Investigation (literature) — K. Arab, L. Chiu (in progress), M. Achim
(TBD); Writing, original draft — S. Khaldoun, K. Arab; Writing, review and
editing — TBD; Project administration — S. Khaldoun, K. Arab.

**Each team member must review and correct their own line before
submission.** Full draft table also in `6_project_progress/statements.md`.

## Draft Use of Generative AI statement

Generative AI (Claude Code, Anthropic) was used, directed by team members,
for: drafting and debugging the data preprocessing, model training,
evaluation, ablation, calibration, and interface code; drafting analysis
documents and report text from computed results and retrieved literature;
and literature search/retrieval. AI-generated literature summaries are
screening aids only, not evidence of human reading.

**The team must verify this statement is accurate and complete before
submission.** Full draft also in `6_project_progress/statements.md`.
