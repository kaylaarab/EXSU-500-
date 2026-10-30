"""
Regression tests for the demo interface, using Streamlit's headless test harness.

Run from the repo root:
    python -m pip install -r 7_interface/requirements.txt pytest
    python -m pytest 7_interface/test_app.py -v

These are smoke/regression tests, not a correctness audit of the model
itself (that lives in scripts/ and 5_results/evaluation/). They exist so a
change to app.py, model/config.json or model/catboost.cbm that breaks the
app is caught automatically instead of only by someone noticing in a
browser. Written after an untracked ad-hoc run surfaced a real bug (see
"fresh-load race condition" below) that a committed test would have caught.
"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).parent / "app.py"


def _fresh():
    at = AppTest.from_file(str(APP))
    at.run(timeout=60)
    assert not at.exception, f"App raised on initial load: {at.exception}"
    return at


def test_loads_without_exception():
    _fresh()


def test_disclaimer_shown():
    at = _fresh()
    full_text = " ".join(e.value for e in at.get("warning") + at.get("caption") if isinstance(e.value, str))
    assert "not a medical device" in full_text.lower()


RISK_LABEL = "Estimated 30-day readmission risk"


def _risk_metric(at):
    return next((m for m in at.metric if m.label == RISK_LABEL), None)


def _click_by_label(at, label):
    btn = next(b for b in at.button if b.label == label)
    btn.click().run(timeout=60)


def test_default_patient_scores():
    """The default (unmodified) form should produce a risk estimate with no error."""
    at = _fresh()
    _click_by_label(at, "Estimate risk")
    assert not at.exception
    m = _risk_metric(at)
    assert m is not None, "Expected a risk metric after submitting the default form"
    assert m.value.endswith("%")


def test_lower_risk_example_runs():
    at = _fresh()
    _click_by_label(at, "Example: lower-risk patient")
    assert not at.exception
    _click_by_label(at, "Estimate risk")
    assert not at.exception
    assert _risk_metric(at) is not None


def test_higher_risk_example_runs():
    at = _fresh()
    _click_by_label(at, "Example: higher-risk patient")
    assert not at.exception
    _click_by_label(at, "Estimate risk")
    assert not at.exception
    assert _risk_metric(at) is not None


def test_died_or_hospice_is_not_scored():
    # Fields inside st.form only commit on submit, so the selectbox change
    # and the submit click must be queued together and run once -- running
    # them separately silently reverts the selectbox (a real Streamlit
    # forms quirk, confirmed while writing this test).
    at = _fresh()
    at.selectbox(key="k_disch").set_value("Died or hospice care")
    next(b for b in at.button if b.label == "Estimate risk").click()
    at.run(timeout=60)
    assert not at.exception
    info_text = " ".join(e.value for e in at.info)
    assert "excluded from training" in info_text
    # The About tab always renders its own 4 metrics regardless of the form,
    # so check specifically for the absence of the risk-calculator's metric.
    risk_labels = [m.label for m in at.metric]
    assert "Estimated 30-day readmission risk" not in risk_labels, \
        "A died/hospice patient should not receive a risk score"


def test_race_selectbox_default_matches_config():
    """
    Regression test for the fresh-load race condition logged 2026-10-26
    (progress_log.md): on a very fast interaction right after page load,
    the Race selectbox was observed to send its first option instead of
    the intended default. AppTest runs synchronously so it cannot reproduce
    the timing race itself, but this at least pins the default to what
    config.json declares, so a future code change that silently breaks the
    default (e.g. a choices/defaults mismatch) is caught.
    """
    import json

    cfg = json.loads((Path(__file__).parent / "model/config.json").read_text())
    at = _fresh()
    assert at.selectbox(key="k_race").value == cfg["defaults"]["race"]


def test_all_models_table_present():
    at = _fresh()
    at.tabs[1].run(timeout=60) if hasattr(at.tabs[1], "run") else None
    # The "About" tab content is rendered on initial load regardless of active tab.
    assert any("All models compared" in (h.value or "") for h in at.subheader)
