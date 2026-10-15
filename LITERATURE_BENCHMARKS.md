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

| Model | AUROC | AUPRC | Sensitivity | Specificity |
|---|---|---|---|---|
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
- **Comparison to us:** Our logistic regression (0.643) is essentially
  identical to theirs (0.642) — good sign our preprocessing/split is
  sound. Our XGBoost (0.678) and MLP (0.655) both beat their equivalents.
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

## Team research log

Add your entries below, one per source, using this format:

```
### [Author(s), Year]
- Citation:
- Dataset used:
- Task/label definition:
- Models + reported metrics:
- How it relates to our project:
```

<!-- Kayla: add your findings below this line -->


<!-- Magalie: add your findings below this line -->

