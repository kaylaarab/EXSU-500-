# EmbPred30 (arXiv:2002.11215): why the 0.974 AUROC is not credible

**Paper:** Sarthak, Shukla, Tripathi (2020), *EmbPred30: Assessing 30-days Readmission for Diabetic Patients using Categorical Embeddings*. arXiv preprint (v1, 25 Feb 2020), not peer-reviewed as far as we found. Link: <https://arxiv.org/abs/2002.11215>. Licence: arXiv non-exclusive, so the PDF is not stored in this repo.
**Claim:** 95.2% accuracy and 97.4% AUROC (6-fold cross-validation) on the UCI Diabetes 130-US Hospitals data, with an embedding-based neural network.
**Last updated:** 2026-10-23. Parts 1 and 2 are two independent reviews; part 3 is our own experiment.

---

## 1. Siham's assessment (from her review of her local PDF)

*Pasted into the project session on 2026-10-23. Reproduced in substance and attributed to Siham; points marked [verified] were also checked against the PDF text in part 2.*

**Verdict:** the reported 0.974 AUROC is highly questionable and plausibly inflated. The paper does not establish that it would hold on genuinely unseen patients. The PDF alone cannot prove leakage or fabrication.

| Issue | Evidence in the paper | Why it matters |
|---|---|---|
| **Possible SMOTE leakage** [verified] | Section 3.3 creates a balanced synthetic dataset; Section 4 then reports six-fold cross-validation. It never says SMOTE was restricted to each training fold. | If SMOTE preceded splitting, training and validation contain related synthetic examples and validation information influences training, which can inflate AUROC substantially. |
| **Unclear validation population** | The synthetic dataset has a 1:1 class ratio, but the paper does not say whether validation contains only original patients. | Success on synthetic examples does not show success on real patients. |
| **Possible feature-selection leakage** [verified] | Page 8: features were selected by importance after training for 70 epochs; no separate selection set or per-fold selection is described. | If validation labels influenced selection, the cross-validated score is optimistic. |
| **Categorical data in SMOTE** | Most inputs are categorical (diagnosis codes, binary medication flags) but only ordinary "SMOTE" is specified. | Interpolating category codes can create invalid values or artefacts that separate synthetic positives from real negatives. |
| **Misleading comparisons** [verified] | Table 1 places accuracy and precision-recall AUC values in a column headed "AUROC", with footnotes acknowledging some exceptions. | These are not interchangeable, so the table cannot support the claimed ranking. |

Further points from the assessment:
- **SMOTE and cross-validation ordering is the biggest concern.** The ordering suggests oversampling may have happened before cross-validation, but that is an inference, not a confirmed implementation fact. The [imbalanced-learn documentation](https://imbalanced-learn.org/stable/common_pitfalls.html) names resampling before splitting as a leakage pitfall. Using the stated 11.2% prevalence, about 87% of the resulting positive class would be synthetic; if those enter validation, much of the reported sensitivity measures performance on generated observations.
- **Balancing alone does not explain the AUROC.** The problem is contamination between training and validation, or evaluating on synthesised examples. Likewise 95.2% accuracy needs context: predicting "no readmission" for everyone already gives 88.8% on the original prevalence.
- **Internal consistency is not validity.** Figure 3 reports about 94.3% sensitivity and 96.0% specificity (average 95.15%, consistent with 95.2% accuracy on balanced classes) and Figure 2 shows a very strong ROC curve. A contaminated evaluation can produce perfectly consistent figures.
- **The medication finding is overinterpreted.** The engineered `count_meds` dominates feature importance (Figure 1). That deserves investigation, but medication changes during the admission can be legitimate predictors at discharge, and importance does not show causation.

**What would convince the reviewer:** split original patients first; confine all learned preprocessing and feature selection to training folds; apply SMOTE to training data only; validate on untouched real patients at natural prevalence; compare with simple models on identical splits; report AUROC, precision-recall AUC, calibration and uncertainty.

**Final verdict (Siham):** an unverified performance estimate with serious risks of inflation, not credible evidence of near-perfect readmission prediction. There is insufficient evidence to call it fabricated, and no defensible way to estimate the corrected AUROC from the PDF alone.

---

## 2. Checked against the PDF text (Claude, 2026-10-23)

We downloaded the paper and read its text.

**Confirmed**
- Preprocessing step iv, "Data Synthesis": SMOTE is applied "to deal with the imbalance in the original dataset, which has 88.8% negative samples", giving "a 1:1 ratio of samples from both the target classes". It sits in the preprocessing list, before the results.
- Results are "after evaluating it using k-fold cross-validation (k=6)". The text never says SMOTE was applied inside each training fold. The future-work section calls the SMOTE samples "data points ... generated by SMOTE", acknowledging they are not real.
- Feature selection: importance "analyzed after training our model for 70 epochs", dropping features with importance 0. No separate selection set is described.
- Table 1 (prior work) has one "AUROC" column with footnotes [1] accuracy and [2] precision-recall AUC, so metrics are mixed.

**Corrections to common assumptions (and to an earlier version of our own notes)**
- The paper **does** de-duplicate patients: "only the first encounter with the patient is kept, and the rest of the patient's records are dropped."
- It **does** remove death and end-of-life patients: "if a patient died or is referred to the end of life care ... deleted."
- So **repeated-patient leakage is not the explanation**; the suspect is SMOTE (and possibly feature selection) relative to the cross-validation.

**New observations**
- The paper quotes 88.8% negatives / 11.2% positives, which describe the full 101,766-row file. After keeping only first encounters the positive rate is about 9%, so the stated prevalence does not match the cohort it says it used. This is unexplained.
- With about 9% positives, SMOTE to 1:1 would make roughly 90% of the positive class synthetic.
- In Table 1 the other studies with data augmentation ("Aug. = Yes") also report 95-97%, while non-augmented studies report much lower values (63-81%, one 94%). This is a correlation across mixed metrics, not proof, but it points the same way.
- We could not check figure-derived numbers (sensitivity 94.3%, specificity 96.0%) from the text alone.

---

## 3. Our own test of the mechanism

`scripts/11_smote_leakage_demo.py` reproduces the *protocol*, not their network. On a first-encounter cohort like theirs (death/hospice excluded), with a fixed gradient-boosting model and the same 6 folds, only the evaluation protocol changes.

Cohort: 69,990 first encounters (9.0% readmitted), death/hospice excluded. SMOTE-NC to 1:1 makes 57,420 synthetic rows, **90% of the positive class**. Model: HistGradientBoosting, fixed settings. Same 6 folds throughout. Threshold 0.5. The always-"no" accuracy on the real cohort is 0.910.

| Protocol | AUROC | Accuracy | Sensitivity | Specificity | Scored on |
|---|---|---|---|---|---|
| **A. SMOTE before CV, scored on all held-out rows** | **0.964** | **0.942** | **0.890** | 0.994 | real + synthetic rows |
| B. Same folds, scored on real rows only | 0.640 | 0.908 | **0.033** | 0.994 | real rows |
| C. SMOTE inside training folds only (correct) | 0.614 | 0.908 | **0.027** | 0.995 | real rows |
| D. Class weights, no SMOTE (our method) | 0.654 | 0.683 | **0.519** | 0.699 | real rows |

**What this shows**
1. **Protocol A reproduces the scale of EmbPred30's claim** (theirs: 0.974 AUROC, 95.2% accuracy, about 94% sensitivity) with an ordinary gradient-boosting model. The inflation comes entirely from scoring on a mixture that is half synthetic: each synthetic positive is an interpolation of real training positives, so it is easy to recognise.
2. **The same model scores 0.640 on real patients (B)**, and only 0.614 when SMOTE is properly confined to training folds (C). The high score is evaluation artefact, not skill.
3. **Accuracy and sensitivity tell a striking story in B and C.** Accuracy is 0.908, almost the always-"no" baseline of 0.910, while sensitivity is 3%. A model trained on SMOTE data and evaluated on real data flags almost nobody at threshold 0.5, because it learned the synthetic distribution. This is why we report sensitivity beside accuracy.
4. **Class weights (D) are the only protocol that gives a usable operating point** (sensitivity 0.52) on real patients.

**Caveats.** We reproduced the protocol, not their embedding network or their exact preprocessing, so this shows that the protocol *can* produce their numbers, not that it did. Their paper does not say where SMOTE sits relative to the folds; our test is evidence about the mechanism, not proof about their run.

---

## 4. Suggested wording for the report

> Reported performance on this dataset ranges from about 0.60 to 0.68 AUROC in studies that evaluate on untouched real patients at natural prevalence. One preprint reports 0.974 [EmbPred30], but its preprocessing applies SMOTE to a 1:1 ratio before six-fold cross-validation without stating that oversampling was confined to training folds; we therefore treat it as unverified. Our own experiment shows that this evaluation order alone can raise AUROC well above what a leak-free protocol achieves (Table X).

Cite EmbPred30 only as a cautionary example and state clearly that we did not reproduce it. Each team member who cites it should have read it.
