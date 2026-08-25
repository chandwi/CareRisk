"""
Feature engineering for CareRisk.

Design principle: features should map to something a hospital analyst could
explain in one sentence (prior utilization, medication burden, comorbidity
mix) — not black-box interaction terms. See notebooks/04_feature_engineering.ipynb
for the rationale behind each one.
"""
import numpy as np
import pandas as pd

MEDICATION_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]

AGE_MIDPOINT = {
    "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35, "[40-50)": 45,
    "[50-60)": 55, "[60-70)": 65, "[70-80)": 75, "[80-90)": 85, "[90-100)": 95,
}

# Standard ICD-9 grouping used in the original Strack et al. (2014) study on
# this dataset and widely replicated since. Codes are strings; V/E-prefixed
# codes are supplementary/external-cause codes, bucketed as "Other".
def _icd9_group(code: str) -> str:
    if pd.isna(code) or code in ("Unknown", "?"):
        return "Unknown"
    if code.startswith("V") or code.startswith("E"):
        return "Other"
    try:
        num = float(code)
    except ValueError:
        return "Other"
    if 250 <= num < 251:
        return "Diabetes"
    if (390 <= num <= 459) or num == 785:
        return "Circulatory"
    if (460 <= num <= 519) or num == 786:
        return "Respiratory"
    if (520 <= num <= 579) or num == 787:
        return "Digestive"
    if 800 <= num <= 999:
        return "Injury"
    if 710 <= num <= 739:
        return "Musculoskeletal"
    if (580 <= num <= 629) or num == 788:
        return "Genitourinary"
    if 140 <= num <= 239:
        return "Neoplasms"
    return "Other"


def add_diagnosis_groups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["diag_1", "diag_2", "diag_3"]:
        df[f"{col}_group"] = df[col].apply(_icd9_group)
    diabetes_dx_count = (df[["diag_1_group", "diag_2_group", "diag_3_group"]] == "Diabetes").sum(axis=1)
    circulatory_dx_count = (df[["diag_1_group", "diag_2_group", "diag_3_group"]] == "Circulatory").sum(axis=1)
    df["has_diabetes_comorbidity"] = (diabetes_dx_count > 0).astype(int)
    df["has_circulatory_comorbidity"] = (circulatory_dx_count > 0).astype(int)
    return df


def add_utilization_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["prior_utilization"] = df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    df["prior_inpatient_flag"] = (df["number_inpatient"] > 0).astype(int)
    df["prior_emergency_flag"] = (df["number_emergency"] > 0).astype(int)
    df["high_utilizer"] = (df["prior_utilization"] >= df["prior_utilization"].quantile(0.90)).astype(int)
    return df


def add_medication_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["med_count_changed"] = 0
    for col in MEDICATION_COLUMNS:
        df["med_count_changed"] += df[col].isin(["Up", "Down"]).astype(int)
    df["medication_complexity"] = df["num_medications"] + df["med_count_changed"]
    return df


def add_intensity_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hospitalization_intensity"] = (
        df["time_in_hospital"] * (df["num_procedures"] + df["num_lab_procedures"])
    )
    df["comorbidity_burden"] = df["number_diagnoses"]
    return df


def add_demographic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["age_midpoint"] = df["age"].map(AGE_MIDPOINT)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_diagnosis_groups(df)
    df = add_utilization_features(df)
    df = add_medication_features(df)
    df = add_intensity_features(df)
    df = add_demographic_features(df)
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/processed/diabetic_data_clean.csv")
    out = engineer_features(df)
    out.to_csv("data/processed/diabetic_data_features.csv", index=False)
    print(f"In: {df.shape} -> Out: {out.shape}")
    new_cols = sorted(set(out.columns) - set(df.columns))
    print("New columns:", new_cols)
