# 7. Demo interface (optional course bonus)

A small web app that shows the model's 30-day readmission risk for one hypothetical patient, why the model gave that estimate, and the project's results.

**Course prototype for research and educational purposes. Not a medical device. Not for clinical use.** The app shows this notice at the top and bottom of every page, as the specification requires.

## Run it on your computer
```bash
pip install -r 7_interface/requirements.txt
streamlit run 7_interface/app.py
```
It opens in your browser. No dataset is needed: the app uses the saved model and a small config file.

## What it shows
- **Risk calculator:** form for the main fields known at discharge, with range checks. Output: calibrated risk %, risk band (low, moderate, high), the observed readmission rate for that band on unseen test patients, and the features that pushed the estimate up or down (SHAP). Patients who died or went to hospice are not scored (they were excluded from training).
- **Example buttons** fill the form with a lower-risk and a higher-risk made-up patient so a reviewer can try the tool in one click.
- **About, results and limits:** calibration curve and decision curve, a subgroup audit table (race, sex, age), test AUROC, AUPRC, sensitivity and accuracy (always shown together), the model comparison table, the full experiment log, and the limitations.

## Files
| File | What it is |
|---|---|
| `app.py` | the Streamlit app |
| `model/catboost.cbm` | the trained model (about 1 MB) |
| `model/config.json` | feature list, allowed values, calibration, risk bands, test metrics |
| `assets/` | copies of the results tables the app displays |
| `requirements.txt` | packages the app needs |

Rebuild the model and config with `python scripts/17_build_demo_model.py` (needs the data and `scripts/01_preprocess.py` run first).

## Publishing it (bonus +2%; decision for the team)
The specification awards +2 percentage points if the interface is live at a public URL, without login, until at least 1 May 2027, runs our own trained model, and shows the prototype notice. Free options are listed on myCourses; Streamlit Community Cloud and Hugging Face Spaces both host Streamlit apps.

1. The repository must be **public** for Streamlit Community Cloud (a public repo also earns the separate +3% code bonus; the specification says to publish only when you are satisfied, because public commits are permanent and attributed by name).
2. Create the app in the host, point it at this repository, branch `main`, main file `7_interface/app.py`, and requirements file `7_interface/requirements.txt`.
3. Open the live URL, test the three cases in the checklist below, and add the URL to the top-level README.

Checklist before publishing: the notice is visible on both tabs; the default patient returns a score; a "died or hospice" discharge is refused with a message; wild values (for example 0 diagnoses) cannot be entered; no patient data or dataset files are in the repository.

## Known limits of the interface
- Unlisted diabetes drugs are assumed "No" (over 99% of training rows), and untouched fields use the training median or mode.
- Race is an input because it is in the dataset. Performance by race, age and sex has not yet been audited, so do not read the output as fair across groups.
- The calibrated percentage is only as good as a model with AUROC about 0.68.
