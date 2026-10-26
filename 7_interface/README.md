# 7. Demo interface (optional course bonus)

A small web app that shows the model's 30-day readmission risk for one hypothetical patient, why the model gave that estimate, and the project's results.

**Course prototype for research and educational purposes. Not a medical device. Not for clinical use.** The app shows this notice at the top and bottom of every page, as the specification requires.

## Run it on your computer (about 10 minutes the first time)
You need **Python 3.10 to 3.12** (3.12 is the safest choice for the CatBoost package) and a copy of this repository. The app uses the saved model in `model/`, so no dataset is needed. **Git is not required.**

### Windows, without Git (easiest)
1. **Download the repository as a ZIP:** open <https://github.com/skhaldoun/EXSU-500-/archive/refs/heads/main.zip> in your browser, then right-click the downloaded file, choose *Extract All*, and note where the new folder is (for example `C:\Users\Siham\Desktop\EXSU-500--main`).
2. **Check that Python is installed.** Open PowerShell and run `python --version`. If it says the command is not recognized, install Python from <https://www.python.org/downloads/>; on the first installer screen **tick "Add python.exe to PATH"**, then open a new PowerShell window.
3. **Go into the folder** (PowerShell must be inside the project folder, not in `C:\Users\Siham`):
   ```powershell
   cd "C:\Users\Siham\Desktop\EXSU-500--main"      # use your own path
   ```
4. **Install the packages and start the app:**
   ```powershell
   python -m pip install -r 7_interface/requirements.txt
   python -m streamlit run 7_interface/app.py
   ```
5. Your browser opens at <http://localhost:8501>. Stop the app with Ctrl+C in PowerShell. The page is visible only on your own computer.

Things that go wrong, and the fix:
| Message | Fix |
|---|---|
| `python` is not recognized | Install Python and tick "Add python.exe to PATH", then open a new PowerShell window. If `py --version` works, use `py -m pip ...` and `py -m streamlit ...` instead |
| `git` is not recognized | You do not need Git; use the ZIP route above |
| `No such file` for `7_interface/...` | PowerShell is in the wrong folder; `cd` into the extracted folder first (it contains `README.md` and the `7_interface` folder) |
| pip fails building `catboost` or numpy | Your Python is too new or too old; install Python 3.12 and try again |
| Browser did not open | Open <http://localhost:8501> by hand |

To update later, download the ZIP again (or use Git if you install it).

### Windows or Mac/Linux with Git
```bash
git pull origin main
python -m pip install -r 7_interface/requirements.txt
python -m streamlit run 7_interface/app.py
```
(Optional: create a virtual environment first with `python -m venv .venv`, then activate it: `.venv\Scripts\Activate.ps1` on Windows PowerShell, `source .venv/bin/activate` on Mac/Linux. If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope Process Bypass` first.)

## Run it without installing anything: GitHub Codespaces
Short version for teammates: see [`TEAM_ACCESS.md`](TEAM_ACCESS.md).

GitHub can run the app for you in the cloud and show it in your browser, with nothing installed on your computer. The repository contains the setup file (`.devcontainer/devcontainer.json`), so it starts and installs by itself.

1. Sign in to GitHub and open the repository page: Siham's copy <https://github.com/skhaldoun/EXSU-500-> or Kayla's copy <https://github.com/kaylaarab/EXSU-500-> (both hold the same files; each person's codespace uses their own free hours).
2. Click the green **Code** button, open the **Codespaces** tab, then **Create codespace on main**.
3. Wait 2-4 minutes the first time (it installs the packages). A window that looks like an editor opens in your browser.
4. When the app starts, a browser tab opens automatically, or open the **Ports** tab at the bottom, find port **8501**, and click the globe icon next to its address.
5. **Stop the codespace when you are done** (Codespaces menu on the repository page, or the bottom-left corner of the editor, then *Stop Current Codespace*), so it does not use up your free hours.

Notes
- **It stays private.** A codespace and its forwarded port are visible only to the person who created it by default. Do not change the port to *Public* while we have decided not to publish the interface.
- **Free allowance.** GitHub gives personal accounts a monthly free amount of Codespaces use (check the current limit under GitHub *Settings, Billing and plans*); a stopped codespace uses none.
- **A real Codespace has not been tested by us yet.** We rehearsed the same install and start commands in a clean Python 3.12 copy of the repository (the same Python as the Codespaces image) and the app ran correctly in a browser. If it does not start, open the **Terminal** tab in the codespace and run `python -m pip install -r 7_interface/requirements.txt` then `python -m streamlit run 7_interface/app.py`, and tell us what message you see.

## Other ways to see it
- **Screenshots only:** the `examples/` folder shows the main screens, no running needed.
- **GitHub itself cannot run it.** GitHub Pages serves only static web pages; this app needs Python and the model running on a server.
- **Public hosting services** (Streamlit Community Cloud, Hugging Face Spaces) would make it reachable by anyone, so we are not using them until the team decides to publish.

## What it looks like
| Higher-risk example | Lower-risk example |
|---|---|
| ![higher-risk example](examples/03_higher_risk_example.png) | ![lower-risk example](examples/02_lower_risk_example.png) |

More screenshots (start screen, the "not scored" case, the results tab) with captions: [`examples/`](examples/README.md).

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
| `examples/` | screenshots of the interface with captions |
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
