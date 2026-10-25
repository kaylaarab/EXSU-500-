# Calibration, decision curves and subgroup audit

Script `scripts/18_calibration_subgroups.py`, final model (CatBoost + Platt calibration, the one in the interface), **test set only** (14,913 stays, 11.1% readmitted), patient-level bootstrap (500 resamples) for every interval. The operating threshold (11.6% calibrated risk) was chosen on validation data. Accuracy is always shown with sensitivity; the always-"no" accuracy is 0.889.

![Calibration and decision curve](calibration_and_decision_curve.png)

## 1. Calibration
| Measure | Value | Meaning |
|---|---|---|
| Brier score, calibrated | **0.094** | lower is better; always predicting the 11.1% average gives 0.099 |
| Brier score, raw class-weighted score | 0.219 | class weighting inflates the scores, which is why we calibrate |
| Expected calibration error | 0.007 | average gap between predicted and observed risk across 10 bins |
| Calibration slope | 0.96 | 1.0 is ideal; slightly over-confident |
| Calibration intercept | -0.13 | slightly over-predicts on average |

**Reading:** after calibration, a predicted risk of 10% really corresponds to about 10% of such patients being readmitted (left panel, blue). The raw score is badly miscalibrated (red) and must not be read as a probability. The model slightly over-predicts for the highest-risk patients.

## 2. Decision curve (net benefit)
Net benefit weighs the readmissions caught against false alarms at the risk threshold a care team would act on. The model has a higher net benefit than flagging everyone or no one across the whole range 2%-40%, but the advantage is small at the lowest thresholds, where flagging nearly everyone is almost as good.

| If you act on patients with risk of at least | Share flagged | Net benefit, model | Net benefit, flag everyone |
|---|---|---|---|
| 10% | 46% | 0.033 | 0.012 |
| 15% | 22% | 0.017 | -0.046 |
| 20% | 12% | 0.007 | -0.111 |
| 30% | 4% | 0.003 | -0.270 |

The model is most useful when interventions are costly enough that you want to target the top 10-25% of patients; the absolute gain is modest.

## 3. Subgroup audit (`subgroup_performance.csv`)
At the operating threshold. Positives = readmitted stays in that group. "Too few" = fewer than 30 readmissions.

| Group | n | Readmitted | AUROC (95% CI) | Accuracy | Sensitivity (95% CI) | Specificity | Precision |
|---|---|---|---|---|---|---|---|
| **Race** | | | | | | | |
| Caucasian (largest) | 11,160 | 1,267 | 0.671 (0.652-0.692) | 0.650 | 0.571 (0.538-0.606) | 0.660 | 0.177 |
| African American | 2,775 | 295 | 0.692 (0.656-0.728) | 0.651 | 0.617 (0.551-0.682) | 0.655 | 0.175 |
| Hispanic | 305 | 31 | 0.742 (0.635-0.845) | 0.731 | 0.548 (0.357-0.750) | 0.752 | 0.200 |
| Unknown | 339 | 34 | 0.643 (0.544-0.740) | 0.726 | 0.382 (0.208-0.535) | 0.764 | 0.153 |
| Other | 233 | 22 | too few | 0.717 | too few | 0.730 | 0.186 |
| Asian | 101 | 9 | too few | 0.743 | too few | 0.772 | 0.160 |
| **Sex** | | | | | | | |
| Female | 8,079 | 889 | 0.694 (0.670-0.716) | 0.648 | **0.622** (0.580-0.662) | 0.652 | 0.181 |
| Male | 6,834 | 769 | 0.658 (0.634-0.681) | 0.664 | **0.519** (0.476-0.555) | 0.682 | 0.171 |
| **Age** | | | | | | | |
| under 50 | 2,363 | 246 | 0.721 (0.668-0.772) | 0.737 | 0.585 (0.485-0.688) | 0.754 | 0.217 |
| 50-69 | 5,832 | 622 | 0.695 (0.670-0.718) | 0.700 | 0.548 (0.502-0.589) | 0.719 | 0.189 |
| 70-79 | 3,823 | 428 | 0.645 (0.618-0.673) | 0.594 | 0.586 (0.538-0.638) | 0.594 | 0.154 |
| 80+ | 2,895 | 362 | 0.635 (0.606-0.668) | 0.580 | 0.597 (0.547-0.650) | 0.578 | 0.168 |

**What the audit shows**
1. **Sex gap in sensitivity:** the model catches about 62% of women's readmissions but only about 52% of men's, a difference of roughly 10 points whose confidence intervals do not overlap (equal-opportunity gap). Overall AUROC is also lower for men (0.658 vs 0.694). A real disparity to report and, if the model were ever used, to investigate.
2. **Race:** no reliable difference between Caucasian and African American patients (sensitivity 0.571 vs 0.617, overlapping intervals; AUROC 0.671 vs 0.692). Hispanic, Other and Asian groups are too small to conclude anything, and the wide intervals say so. The "Unknown race" group has the lowest sensitivity (0.382) but only 34 readmissions.
3. **Age:** discrimination falls with age (AUROC 0.72 under 50, 0.64 for 70 and over), so the model is least reliable for the largest patient groups. Sensitivity is similar across ages, but specificity falls (0.75 under 50 vs 0.58 for 80+), meaning older patients are flagged much more often with less precision.
4. **Calibration by group** is in the CSV (`mean_predicted` vs `prevalence`); predicted and observed averages agree within about 1 point in every group with enough data.

**Caveats:** 500 resamples; many comparisons (some "differences" will arise by chance); race is a recorded label from the 1999-2008 hospital systems, not a clean category; we audited three attributes only, with no intersections. This is a first audit, not a certification of fairness.

## Files
`calibration_and_decision_curve.png` · `calibration_bins.csv` · `calibration_summary.csv` · `decision_curve.csv` · `subgroup_performance.csv`
