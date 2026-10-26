# Interface examples

Screenshots of the demo interface running locally (see `../README.md` for how to run it yourself). All patients shown are made-up examples; no real patient data is used anywhere in the app. The prototype notice ("not a medical device, not for clinical use") is shown at the top and bottom of every page.

| Screenshot | What it shows |
|---|---|
| [`01_start_screen.png`](01_start_screen.png) | The risk calculator when it opens: the notice, two example buttons, and the form (patient, hospital stay, history and diabetes care), pre-filled with the most common training values |
| [`02_lower_risk_example.png`](02_lower_risk_example.png) | The "lower-risk patient" example after pressing *Estimate risk*: about **5%**, band LOW, with the observed readmission rate for that band on unseen test patients and the chart of what pushed the estimate down or up |
| [`03_higher_risk_example.png`](03_higher_risk_example.png) | The "higher-risk patient" example (older, three inpatient and two emergency visits in the past year, nine days in hospital, discharged to a nursing facility): about **27%**, band HIGH |
| [`04_died_or_hospice_not_scored.png`](04_died_or_hospice_not_scored.png) | Choosing "Died or hospice care" as the discharge destination: the app refuses to score, explaining that such patients were excluded from training because they cannot be readmitted |
| [`05_about_results_and_limits.png`](05_about_results_and_limits.png) | The second tab: test AUROC, AUPRC, sensitivity and accuracy side by side, the model comparison, the calibration and decision curves, the subgroup audit (including the sex gap) and the limits |

**Worked low-risk and high-risk examples, with the inputs, outputs and main reasons written out: [`low_and_high_risk_examples.md`](low_and_high_risk_examples.md).**

How to read the output: the percentage is a calibrated estimate (after correction, a 10% prediction corresponds to about 10% of such patients being readmitted). The model ranks patients only moderately well (AUROC about 0.68), so the number is a rough indication of relative risk, not a diagnosis. The red and blue bars show which inputs raised or lowered the estimate for that patient (SHAP values), which describes what the model uses, not what causes readmission.

These images were produced with `streamlit run 7_interface/app.py` and a headless browser on 2026-10-26, and show the current version of the app. If the app changes, regenerate them.
