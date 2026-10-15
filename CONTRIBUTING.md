# How to save your work to this repo

We're using GitHub directly for everything (code, README, results) instead
of Google Drive, so the repo stays the single source of truth for grading.
Here's how to get set up and contribute.

## One-time setup

1. **Accept the collaborator invite** — check your email, or go to
   github.com/kaylaarab/EXSU-500-/invitations
2. **Install Git**, or use [GitHub Desktop](https://desktop.github.com/) if
   you'd rather avoid the command line — it does the same thing with buttons
   instead of commands.
3. **Clone the repo** (downloads a copy to your computer):
   ```bash
   git clone https://github.com/kaylaarab/EXSU-500-.git
   ```
   In GitHub Desktop: File → Clone Repository → paste the URL.
4. **Install the Python packages** we're using:
   ```bash
   pip install -r requirements.txt
   ```
5. **Get the dataset** (not stored in the repo — see README.md for the
   one-line download command).

## Every time you work on something

1. **Pull first** — gets everyone else's latest changes so you're not
   working on an outdated copy:
   ```bash
   git pull
   ```
2. **Do your work** — edit the README, add a script, add results, whatever.
3. **Save it:**
   ```bash
   git add <the files you changed>
   git commit -m "short description of what you did"
   git push
   ```

In GitHub Desktop, the same flow is: Pull → make your changes → type a
summary in the box → click **Commit** → click **Push**.

## A few ground rules

- **Don't commit the raw dataset file.** Only the download instructions
  belong in the repo (already in `README.md`). The `.gitignore` file is
  already set up to block this from happening by accident.
- **Don't commit model weight files over ~100 MB.** Host them externally
  (GitHub Releases or Hugging Face Hub) and link to them instead.
- **Commit as you go, not all at once at the end.** The course grades our
  commit history as evidence of how the work was distributed across the
  team — a repo with one commit from one person on the last day looks bad
  for everyone, regardless of who actually did the work.
- **Commit under your own GitHub account**, not someone else's.
- **If you're editing something someone else might also be touching**
  (e.g. the README), give the group a heads-up first to avoid conflicts.
- Use clear commit messages (e.g. "Add SHAP plot for XGBoost") instead of
  vague ones (e.g. "update").

## If something goes wrong

Send a screenshot of the error to the group chat — most Git errors are easy
to fix once we see what happened.
