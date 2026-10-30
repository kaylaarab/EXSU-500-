# Main models: comparison with patient-level confidence intervals

Script `scripts/21_patient_level_cis.py` (26-30 Oct 2026). Test set: 14,913 stays from 10,499 patients, scored once. Intervals come from resampling **patients** (1,000 resamples), not stays, so they allow for several stays per patient. Threshold 0.5 for accuracy, sensitivity and specificity, as in `model_comparison.csv`. Accuracy is shown beside sensitivity because always answering "no" gives 0.889 accuracy with 0 sensitivity.

| Model | AUROC (95% CI) | AUPRC | Accuracy | Sensitivity (95% CI) | Specificity |
|---|---|---|---|---|---|
| Baseline (majority-class) | 0.500 (0.500-0.500) | 0.111 | 0.889 | 0.000 (0.000-0.000) | 1.000 |
| Logistic Regression | 0.668 (0.652-0.683) | 0.214 | 0.670 | 0.547 (0.518-0.576) | 0.685 |
| XGBoost (tuned on val) | 0.673 (0.657-0.689) | 0.224 | 0.673 | 0.561 (0.532-0.590) | 0.687 |
| CatBoost (engineered features) | 0.676 (0.660-0.692) | 0.227 | 0.672 | 0.555 (0.527-0.584) | 0.686 |
| Ensemble (CatBoost+XGBoost+LogReg blend) | 0.676 (0.660-0.692) | 0.228 | 0.674 | 0.560 (0.531-0.588) | 0.688 |
| MLP (deep learning) | 0.668 (0.652-0.686) | 0.222 | 0.616 | 0.617 (0.590-0.644) | 0.616 |

Paired differences (same resampled patients for both models):

| Model versus CatBoost | AUROC difference (95% CI) | Reliable? |
|---|---|---|
| Baseline (majority-class) | -0.176 (-0.192 to -0.160) | yes |
| Logistic Regression | -0.008 (-0.015 to -0.002) | yes |
| XGBoost (tuned on val) | -0.003 (-0.007 to +0.001) | no (tied) |
| Ensemble (CatBoost+XGBoost+LogReg blend) | +0.000 (-0.002 to +0.002) | no (tied) |
| MLP (deep learning) | -0.008 (-0.015 to -0.000) | yes |

## Notes
- **Conclusion unchanged:** CatBoost, XGBoost and the ensemble are tied; logistic regression and the MLP are slightly but reliably lower (about 0.008 AUROC).
- **The numbers differ slightly from `model_comparison.csv`** (CatBoost 0.676 vs 0.677, XGBoost 0.673 vs 0.672) because script 21 refits each model once with fixed settings (CatBoost 791 trees, as in script 06), whereas scripts 02 and 05 pick settings by validation search. Both are valid; `model_comparison.csv` stays the headline table.
- **MLP: fixed on 30 Oct.** Script 04 used to store the "best epoch" weights with `state_dict().copy()`, which is a shallow copy, so it ended up using the weights from the last epoch, not the best one, and reported 0.660. Script 04 now keeps true copies and was rerun: MLP 0.668 (AUPRC 0.222, accuracy 0.616, sensitivity 0.617), the same as Script 21 finds, and `model_comparison.csv` is updated. The conclusion does not change (still 0.008 below CatBoost), and the tuned MLP in script 12 (0.674) is unaffected.
- Ensemble weights chosen on validation: 0.5 XGBoost, 0.5 CatBoost, 0 logistic regression.
- Test-set probabilities are saved in `data/preds_21.npz` (git-ignored) so the intervals can be recomputed without retraining.
