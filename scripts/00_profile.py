"""
Step 0: Profile the raw Diabetes 130-US Hospitals data.

Prints structure, missingness, target balance, and the data-quality checks
that matter for the readmission task (repeat encounters, death/hospice
discharges, near-constant columns, subgroup positive rates). Writes the
tables to 3_dataset/profile_tables/ so they can be cited in the report.

Usage (from the repo root):
    python scripts/00_profile.py [path/to/diabetic_data.csv]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/diabetic_data.csv"
OUT = Path("3_dataset/profile_tables")
OUT.mkdir(parents=True, exist_ok=True)

# discharge_disposition_id codes for death / hospice (IDS_mapping.csv):
# 11 expired, 13 hospice-home, 14 hospice-medical facility,
# 19 expired at home (Medicaid), 20 expired in a medical facility (Medicaid),
# 21 expired, place unknown (Medicaid)
DEAD_HOSPICE = [11, 13, 14, 19, 20, 21]
MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


raw = pd.read_csv(RAW_PATH, keep_default_na=False)
df = raw.replace("?", np.nan)  # keep_default_na=False above: "None" is a real value (lab not ordered)
df["readmit_30d"] = (df["readmitted"] == "<30").astype(int)

section("1. Shape and dtypes")
print(f"rows={len(df):,}  cols={df.shape[1]}  unique patients={df['patient_nbr'].nunique():,}")
print(df.dtypes.value_counts().to_string())

section("2. Missingness ('?' coded as NaN)")
miss = pd.DataFrame({
    "n_missing": df.isna().sum(),
    "pct_missing": (df.isna().mean() * 100).round(2),
    "n_unique": df.nunique(),
}).sort_values("pct_missing", ascending=False)
print(miss[miss.n_missing > 0].to_string())
miss.to_csv(OUT / "missingness.csv")

section("2b. 'None'-coded labs (not ordered, not missing)")
for c in ["max_glu_serum", "A1Cresult"]:
    print(f"{c}:\n{df[c].value_counts(dropna=False).to_string()}\n")

section("3. Target balance")
print(raw["readmitted"].value_counts().to_string())
print(f"positive rate (<30) = {df['readmit_30d'].mean():.4f}")

section("4. Repeat encounters")
per_pt = df.groupby("patient_nbr").size()
print(per_pt.describe().to_string())
print(f"patients with >1 encounter: {(per_pt > 1).sum():,} "
      f"({(per_pt > 1).mean() * 100:.1f}%)")
print(f"encounters belonging to repeat patients: "
      f"{df['patient_nbr'].map(per_pt).gt(1).mean() * 100:.1f}%")
first = df.sort_values("encounter_id").drop_duplicates("patient_nbr")
print(f"positive rate, first encounter only: {first['readmit_30d'].mean():.4f} "
      f"(n={len(first):,})")

section("5. Death / hospice discharges (cannot be readmitted)")
dead = df["discharge_disposition_id"].isin(DEAD_HOSPICE)
print(df.loc[dead, "discharge_disposition_id"].value_counts().to_string())
print(f"n={dead.sum():,} ({dead.mean() * 100:.2f}%), "
      f"positive rate among them = {df.loc[dead, 'readmit_30d'].mean():.4f}")
print(f"positive rate among the rest = {df.loc[~dead, 'readmit_30d'].mean():.4f}")

section("6. Near-constant columns (>=99% one value)")
rows = []
for c in df.columns:
    top = df[c].value_counts(normalize=True, dropna=False)
    if top.iloc[0] >= 0.99:
        rows.append((c, top.index[0], round(top.iloc[0] * 100, 2)))
print(pd.DataFrame(rows, columns=["column", "top_value", "pct"]).to_string(index=False))

section("7. Medication columns: share not 'No'")
print(((df[MED_COLS] != "No").mean() * 100).round(2).sort_values(ascending=False).to_string())

section("8. Numeric distributions")
num = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
       "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]
print(df[num].describe().T.round(2).to_string())
df[num].describe().T.to_csv(OUT / "numeric_summary.csv")

section("9. Positive rate by subgroup")
for c in ["age", "gender", "race", "admission_type_id", "admission_source_id",
          "discharge_disposition_id", "A1Cresult", "change", "diabetesMed"]:
    g = (df.groupby(c, dropna=False)["readmit_30d"]
           .agg(n="size", pos_rate="mean").round(4))
    print(f"\n-- {c}\n{g.to_string()}")
    g.to_csv(OUT / f"subgroup_{c}.csv")

section("10. Diagnosis codes")
for c in ["diag_1", "diag_2", "diag_3"]:
    print(f"{c}: {df[c].nunique()} unique, {df[c].isna().mean() * 100:.2f}% missing")

section("11. Duplicates and ID sanity")
print(f"duplicate encounter_id: {df['encounter_id'].duplicated().sum()}")
print(f"fully duplicated rows: {df.duplicated().sum()}")
print(f"gender 'Unknown/Invalid' rows: {(df['gender'] == 'Unknown/Invalid').sum()}")

print(f"\nTables written to {OUT}/")
