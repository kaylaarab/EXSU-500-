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

# Columns dropped outright: identifiers, or >=40% missing and not clinically
# recoverable (weight is 96.9% missing; payer_code and medical_specialty are
# administrative, not clinical, and ~40-50% missing)
DROP_COLS = ["encounter_id", "weight", "payer_code", "medical_specialty"]


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


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.replace("?", np.nan)

    # Binary target: readmitted within 30 days vs not
    df["readmit_30d"] = (df["readmitted"] == "<30").astype(int)
    df = df.drop(columns=["readmitted"])

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # Group raw ICD-9 diagnosis codes into broad clinical categories
    for col in ["diag_1", "diag_2", "diag_3"]:
        df[col + "_group"] = df[col].apply(group_icd9)
    df = df.drop(columns=["diag_1", "diag_2", "diag_3"])

    # Remaining missing categoricals -> explicit "Unknown" category
    cat_cols = df.select_dtypes(include="object").columns
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
