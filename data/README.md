# Data

Raw data is not committed to this repo (19MB CSV, and it's trivially re-downloadable).

## Source

UCI Machine Learning Repository — [Diabetes 130-US hospitals for years 1999-2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

- 101,766 encounters, 130 hospitals/integrated delivery networks, 1999-2008
- 50 columns: demographics, admission/discharge info, diagnoses, medications, lab results, prior utilization, readmission outcome
- Use the original files (`diabetic_data.csv`, `IDS_mapping.csv`) — not a pre-cleaned Kaggle/GitHub copy. The messiness (missing values encoded as `?`, mixed-granularity ID mappings, high-cardinality diagnosis codes) is part of the project.

## Setup

1. Download the zip from the UCI page above.
2. Extract `diabetic_data.csv` and `IDS_mapping.csv` into `data/raw/`.
3. Run `src/preprocessing.py` (or `notebooks/01_data_audit.ipynb`) to produce `data/processed/`.

## Known data quality issues (see `notebooks/01_data_audit.ipynb` for the full audit)

- `weight` is 97% missing (`?`) — dropped.
- `payer_code` (40%) and `medical_specialty` (49%) are heavily missing but not dropped outright — missingness itself may be informative, kept as an explicit "Unknown" category.
- `examide` and `citoglipton` are constant (single value) — dropped, zero information.
- `race` has 2,273 `?` values — kept, `?` recoded to "Unknown" rather than imputed.
- **Target leakage risk:** `discharge_disposition_id` values 11, 19, 20, 21 correspond to "Expired" — a patient who died cannot be readmitted. These encounters are excluded before modeling.
- **Non-independence:** 71,518 unique patients across 101,766 encounters — 16,773 patients have more than one encounter (up to 40). Encounters from the same patient are not independent, which matters for train/test splitting (must split by `patient_nbr`, not by encounter) and for any fairness analysis.
- Target is collapsed from 3 classes (`<30`, `>30`, `NO`) to binary: `readmitted_30d = 1` if `<30`, else `0`. Positive rate ≈ 11.2% (imbalanced — accuracy is not a usable headline metric).
