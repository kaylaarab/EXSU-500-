# Where we stand — plain-language explainer

This file is for the team, not for the final submission. It explains what's
been done so far, in plain terms, so everyone can see where things stand
without reading code.

## The problem, in one sentence

Can we predict which diabetic patients are likely to be readmitted to the
hospital within 30 days of being discharged, using information we already
have at the time they leave?

**Why it matters:** if we can flag high-risk patients at discharge, hospitals
could prioritize follow-up calls, closer monitoring, or earlier
interventions for them — readmissions are expensive and often preventable.

## The dataset

**Diabetes 130-US Hospitals** — real hospital records from 130 US hospitals,
1999-2008. 101,766 hospital visits ("encounters") from diabetic patients,
each with demographics, diagnoses, lab results, medications, and whether
they were readmitted.

It's fully open (no approval needed), well documented, and has been used in
published research before — important for us because we need 15+ references
for the midterm report, and papers already exist on this exact dataset.

## What we checked before committing to it

- **Does it load and actually have the data we need?** Yes, verified.
- **Is the outcome we're predicting real?** Yes — readmission is an actual
  recorded event, not something we're guessing at or deriving indirectly.
- **Watch out:** the same patient can show up more than once (up to 40 times
  for one person!). If we're not careful, the same patient could end up in
  both our "training" data and our "test" data, which would make our results
  look artificially good. We built the split specifically to avoid this —
  see below.

## What "training a model" actually means here

Think of it like this: we show the computer thousands of past hospital
visits where we already know the answer ("was this patient readmitted
within 30 days or not?"). The computer looks for patterns — combinations of
age, diagnoses, medications, prior hospital visits, etc. — that tend to show
up before a readmission. Once it's learned those patterns, we test it on
patients it has *never seen before*, to see if it can correctly guess the
answer for new cases.

That "never seen before" part is critical — it's the only way to know if the
model actually learned something useful, versus just memorizing the data we
showed it.

## The steps we've completed so far

### 1. Cleaning and splitting the data (`01_preprocess.py`)
- Removed columns that were mostly missing or not useful (e.g., `weight`
  was missing for 97% of patients).
- Turned the outcome into a simple yes/no: readmitted within 30 days, or
  not. About 11% of visits are a "yes" — so this is a somewhat rare event,
  which makes the task harder and means plain "accuracy" would be a
  misleading way to measure success (a model that always says "no" would be
  "89% accurate" while being completely useless).
- Split the data into three groups: **training** (teach the model, 70%),
  **validation** (tune the model's settings, 15%), and **test** (final,
  one-time check, 15%) — split so that no single patient's visits are spread
  across more than one group. We double-checked this programmatically.

### 2. Building and comparing models (`02_train_evaluate.py`)
We built three things and compared them fairly (all tested on the exact
same, never-before-seen patients):

| Model | What it is | How good? (AUROC*) |
|---|---|---|
| Baseline | Just guesses "not readmitted" every time — the bare minimum any real model has to beat | 0.500 (= random guessing) |
| Logistic Regression | A simple, classic statistics-based model | 0.637 |
| XGBoost | A more powerful model that combines many decision trees | **0.677** (best so far) |

*AUROC: a score from 0.5 (no better than a coin flip) to 1.0 (perfect). Ours
isn't perfect, but it's clearly better than guessing, and it's in line with
what's been published by other researchers on this same dataset — a good
sign we're doing this correctly.

We also calculated a **confidence range** for our best score (0.662–0.690),
so we're not just reporting one lucky number — we're showing how much it
could reasonably vary.

### 3. Explaining *why* the model makes its predictions (`03_interpretability.py`)

This is the part we just finished. A model that just spits out "high risk"
or "low risk" isn't very useful to a doctor if it can't explain *why*. So we
used a tool called **SHAP** to open up the "black box" and see which
patient factors push the risk score up or down, and by how much, for
individual patients.

**What we found:** the single biggest driver of predicted readmission risk
is **how many times the patient was already hospitalized in the past
year** — which matches what's well known clinically (patients hospitalized
often tend to keep being hospitalized). Second is **where the patient was
discharged to** (e.g., sent home vs. sent to another care facility). This is
a good sign: it means the model is picking up on real, medically sensible
patterns, not random noise in the data.

This produces two files: a chart (`results/shap_summary.png`) and a table
of the top 15 most influential factors (`results/shap_feature_importance.csv`).

## What's NOT done yet

- **A third model type (deep learning)** — optional for this kind of
  structured/spreadsheet-style data; we need to decide as a team whether to
  add one or justify sticking with two strong models (the course rubric
  allows either, as long as we explain our reasoning).
- **Better features from diagnosis codes** — right now we're using the raw
  medical codes; grouping them into broader categories (like "circulatory
  disease," "diabetes complication," etc.) would likely improve results.
- **More thorough tuning** — we only tried a handful of settings combinations
  so far; there's room to search more carefully.
- **Model card, license, contribution statement** — administrative/reporting
  pieces the course requires in the final repo, not yet written.

## Where to find things in the repo

```
README.md                          <- formal project description (for grading)
PROJECT_STATUS.md                  <- this file (plain-language, for us)
scripts/01_preprocess.py           <- cleans data, makes the 3 splits
scripts/02_train_evaluate.py       <- trains and scores all 3 models
scripts/03_interpretability.py     <- SHAP explainability analysis
results/model_comparison.csv       <- the scores table above
results/shap_summary.png           <- SHAP chart
results/shap_feature_importance.csv <- SHAP top-15 table
```

To reproduce any of this yourself: download the data (instructions in
README.md), then run the three scripts in order.
