"""
Step 1: Load, clean, and split the Diabetes 130-US Hospitals dataset.

Produces train/val/test CSVs with the binary 30-day readmission target.
Splits are made at the patient level (patient_nbr) to avoid leakage from
patients with multiple encounters.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

RAW_PATH = "data/diabetic_data.csv"
RANDOM_STATE = 42

# Columns dropped outright: identifiers, and weight (96.9% missing). payer_code
# (39.6% missing) and medical_specialty (49.1% missing) are KEPT, with missing
# encoded as an explicit "Unknown" category: in the sensitivity analysis
# (scripts/07_sensitivity.py) keeping them gave +0.008 AUROC (paired patient-level
# 95% CI +0.005 to +0.011). Payer is a socioeconomic proxy -- see
# docs/dataset_analysis_report.md for the fairness trade-off.
DROP_COLS = ["encounter_id", "weight"]

# discharge_disposition_id codes for death / hospice (see IDS_mapping.csv):
# 11 expired, 13 hospice-home, 14 hospice-medical facility, 19/20/21 expired
# (Medicaid hospice). These patients cannot be readmitted, so keeping them
# lets the model learn a trivial "died -> no readmission" rule (positive
# rate 1.8% vs 11.4% for everyone else). Strack et al. (2014) excluded them.
DEAD_HOSPICE_CODES = [11, 13, 14, 19, 20, 21]

# Integer ID columns that are category codes, not quantities
ID_COLS_CATEGORICAL = [
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
]

# The 23 medication columns, each coded Up/Down/Steady/No
MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]


def group_icd9(code):
    """Map a raw ICD-9 diagnosis code to a broad clinical category.

    Raw diag_1/2/3 codes have ~700-900 unique values combined, which makes
    for a huge, sparse one-hot encoding the model can't learn much from.
    Grouping into ~9 categories (same approach as the dataset's original
    paper, Strack et al. 2014) gives the model denser, more learnable signal.
    """
    if pd.isna(code):
        return "Missing"
    code = str(code)
    if code.startswith("V") or code.startswith("E"):
        return "Other"
    try:
        c = float(code)
    except ValueError:
        return "Other"

    if 390 <= c <= 459 or c == 785:
        return "Circulatory"
    if 460 <= c <= 519 or c == 786:
        return "Respiratory"
    if 520 <= c <= 579 or c == 787:
        return "Digestive"
    if int(c) == 250 or (250 <= c < 251):
        return "Diabetes"
    if 800 <= c <= 999:
        return "Injury"
    if 710 <= c <= 739:
        return "Musculoskeletal"
    if 580 <= c <= 629 or c == 788:
        return "Genitourinary"
    if 140 <= c <= 239:
        return "Neoplasms"
    return "Other"


def load_and_clean(path: str, drop_cols=None, dead_codes=None) -> pd.DataFrame:
    drop_cols = DROP_COLS if drop_cols is None else drop_cols
    dead_codes = DEAD_HOSPICE_CODES if dead_codes is None else dead_codes
    # keep_default_na=False: pandas would otherwise read the literal string
    # "None" (labs not ordered: A1Cresult, max_glu_serum) as missing, which
    # silently breaks the A1C feature below. Only "?" means missing.
    df = pd.read_csv(path, keep_default_na=False, na_values=["?"], low_memory=False)

    # Drop encounters that end in death/hospice (cannot be readmitted)
    dead = df["discharge_disposition_id"].isin(dead_codes)
    print(f"Excluding {dead.sum():,} death/hospice discharges "
          f"({dead.mean() * 100:.2f}%)")
    df = df[~dead].copy()

    # Binary target: readmitted within 30 days vs not
    df["readmit_30d"] = (df["readmitted"] == "<30").astype(int)
    df = df.drop(columns=["readmitted"])

    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Group raw ICD-9 diagnosis codes into broad clinical categories
    for col in ["diag_1", "diag_2", "diag_3"]:
        df[col + "_group"] = df[col].apply(group_icd9)
    df = df.drop(columns=["diag_1", "diag_2", "diag_3"])

    # Engineered features (added after first-pass results plateaued at ~0.68
    # AUROC -- see literature/LITERATURE_BENCHMARKS.md; these are standard derived
    # features used in follow-on work on this dataset):

    # 1. Combined prior-utilization signal. The three raw counts
    # (outpatient/emergency/inpatient visits in the year before this
    # encounter) are individually sparse (mostly 0); a single combined count
    # is denser and captures "how much healthcare has this patient already
    # been using" in one feature, on top of keeping the originals.
    df["total_prior_visits"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    )

    # 2. Medication-change intensity. 23 separate Up/Down/Steady/No columns
    # are individually very sparse. Collapsing to "how many medications were
    # adjusted this encounter" is a single denser signal for treatment
    # instability, which is clinically linked to readmission risk.
    df["num_med_changes"] = (df[MED_COLS] == "Up").sum(axis=1) + (
        df[MED_COLS] == "Down"
    ).sum(axis=1)

    # 3. HbA1c measured-and-acted-on flag. The dataset's original paper
    # (Strack et al. 2014) found that whether HbA1c was measured AND
    # resulted in a treatment change mattered more than the raw medication
    # list -- encode that interaction directly.
    df["a1c_measured_and_changed"] = (
        (df["A1Cresult"] != "None") & (df["change"] == "Ch")
    ).astype(int)

    # Category-code IDs -> strings so downstream models treat them as
    # categorical (prefix keeps them strings when the splits are re-read).
    for c in ID_COLS_CATEGORICAL:
        df[c] = c.replace("_id", "") + "_" + df[c].astype(str)

    # Remaining missing categoricals -> explicit "Unknown" category
    cat_cols = df.select_dtypes(exclude="number").columns
    for c in cat_cols:
        df[c] = df[c].fillna("Unknown")

    return df


def patient_level_split(df: pd.DataFrame, group_col: str = "patient_nbr"):
    # 70/15/15 train/val/test, split by patient so no patient's encounters
    # appear in more than one split
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_STATE)
    train_idx, temp_idx = next(gss1.split(df, groups=df[group_col]))
    train_df, temp_df = df.iloc[train_idx], df.iloc[temp_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_STATE)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df[group_col]))
    val_df, test_df = temp_df.iloc[val_idx], temp_df.iloc[test_idx]

    return train_df, val_df, test_df


def main():
    df = load_and_clean(RAW_PATH)
    print(f"Loaded and cleaned: {df.shape}")
    print(f"Positive rate (readmit <30d): {df['readmit_30d'].mean():.3f}")

    train_df, val_df, test_df = patient_level_split(df)

    # Confirm no patient leakage across splits
    train_p = set(train_df["patient_nbr"])
    val_p = set(val_df["patient_nbr"])
    test_p = set(test_df["patient_nbr"])
    assert not (train_p & val_p), "Patient leakage between train/val"
    assert not (train_p & test_p), "Patient leakage between train/test"
    assert not (val_p & test_p), "Patient leakage between val/test"

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_df.to_csv(f"data/{name}.csv", index=False)
        print(
            f"{name}: {len(split_df)} encounters, "
            f"{split_df['patient_nbr'].nunique()} patients, "
            f"positive rate {split_df['readmit_30d'].mean():.3f}"
        )

    print("\nNo patient appears in more than one split (verified).")


if __name__ == "__main__":
    main()
