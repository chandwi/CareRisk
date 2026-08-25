"""
Model training for CareRisk: Logistic Regression (interpretable baseline),
Random Forest, and XGBoost, all behind one preprocessing pipeline so they're
comparable on the exact same feature matrix.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from rare_category import RareCategoryCollapser

TARGET_COL = "readmitted_30d"
GROUP_COL = "patient_nbr"

# Columns intentionally excluded: identifiers (encounter_id, patient_nbr),
# the original 3-class `readmitted` (that *is* the target, pre-collapse —
# including it would be direct leakage), and raw diag_1/2/3 (superseded by
# the lower-cardinality diag_*_group features from features.py).
DROP_COLS = ["encounter_id", "patient_nbr", "readmitted", "diag_1", "diag_2", "diag_3"]

NUMERIC_FEATURES = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
    "age_midpoint", "prior_utilization", "high_utilizer", "prior_inpatient_flag",
    "prior_emergency_flag", "medication_complexity", "med_count_changed",
    "hospitalization_intensity", "comorbidity_burden", "has_diabetes_comorbidity",
    "has_circulatory_comorbidity",
]

CATEGORICAL_FEATURES = [
    "race", "gender", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "payer_code", "medical_specialty",
    "diag_1_group", "diag_2_group", "diag_3_group",
    "max_glu_serum", "A1Cresult", "change", "diabetesMed",
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]


def get_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype(str)
    y = df[TARGET_COL]
    groups = df[GROUP_COL]
    return X, y, groups


def group_train_test_split(X, y, groups, test_size=0.2, random_state=42):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def build_preprocessor(min_category_count: int = 100) -> ColumnTransformer:
    categorical_pipe = Pipeline([
        ("collapse_rare", RareCategoryCollapser(min_count=min_category_count)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )


def build_models(random_state=42, scale_pos_weight=1.0) -> dict[str, Pipeline]:
    preprocessor = build_preprocessor()
    return {
        "logistic_regression": Pipeline([
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)),
        ]),
        "random_forest": Pipeline([
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=12, min_samples_leaf=20,
                class_weight="balanced", n_jobs=-1, random_state=random_state,
            )),
        ]),
        "xgboost": Pipeline([
            ("prep", preprocessor),
            ("clf", XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight, eval_metric="logloss",
                n_jobs=-1, random_state=random_state,
            )),
        ]),
    }


def train_all(df: pd.DataFrame, random_state=42):
    X, y, groups = get_feature_matrix(df)
    X_train, X_test, y_train, y_test = group_train_test_split(X, y, groups, random_state=random_state)
    spw = (y_train == 0).sum() / (y_train == 1).sum()

    models = build_models(random_state=random_state, scale_pos_weight=spw)
    fitted = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
    return fitted, (X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    df = pd.read_csv("data/processed/diabetic_data_features.csv")
    fitted, splits = train_all(df)
    X_train, X_test, y_train, y_test = splits
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train positive rate: {y_train.mean():.3f}, Test positive rate: {y_test.mean():.3f}")

    import os
    os.makedirs("data/processed/models", exist_ok=True)
    for name, pipe in fitted.items():
        joblib.dump(pipe, f"data/processed/models/{name}.joblib")
    joblib.dump(splits, "data/processed/models/splits.joblib")
    print("Saved:", list(fitted.keys()))
