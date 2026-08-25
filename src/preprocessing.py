"""
Cleaning and target construction for the CareRisk dataset.

Every decision here is justified in notebooks/01_data_audit.ipynb — this module
just implements those decisions so they're reusable and versioned, instead of
copy-pasted across notebooks.
"""
import pandas as pd

DROP_COLUMNS = ["weight", "examide", "citoglipton"]
UNKNOWN_SENTINEL_COLUMNS = ["race", "payer_code", "medical_specialty", "diag_1", "diag_2", "diag_3"]
NOT_TESTED_COLUMNS = ["max_glu_serum", "A1Cresult"]
EXPIRED_DISCHARGE_IDS = [11, 19, 20, 21]  # "Expired" — see IDS_mapping.csv


def load_raw(data_dir: str = "data/raw") -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(f"{data_dir}/diabetic_data.csv")
    ids = pd.read_csv(f"{data_dir}/IDS_mapping.csv", header=None)
    return df, ids


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.drop(columns=DROP_COLUMNS)

    for col in UNKNOWN_SENTINEL_COLUMNS:
        df[col] = df[col].replace("?", "Unknown")

    for col in NOT_TESTED_COLUMNS:
        df[col] = df[col].fillna("Not tested")

    df["readmitted_30d"] = (df["readmitted"] == "<30").astype(int)

    n_before = len(df)
    df = df[~df["discharge_disposition_id"].isin(EXPIRED_DISCHARGE_IDS)].copy()
    n_dropped = n_before - len(df)
    print(f"Dropped {n_dropped} expired-discharge encounters ({100 * n_dropped / n_before:.2f}%)")

    return df


def save_processed(df: pd.DataFrame, out_dir: str = "data/processed") -> None:
    df.to_csv(f"{out_dir}/diabetic_data_clean.csv", index=False)


if __name__ == "__main__":
    raw, ids = load_raw()
    cleaned = clean(raw)
    save_processed(cleaned)
    print(f"Raw: {raw.shape} -> Clean: {cleaned.shape}")
