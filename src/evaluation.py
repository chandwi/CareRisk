"""
Evaluation for CareRisk models. Accuracy is deliberately absent from the
headline metrics — at an 11% positive rate, a model that predicts "no
readmission" for everyone scores ~89% accuracy while being useless.
"""
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)


def evaluate_model(pipe, X_test, y_test) -> dict:
    proba = pipe.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "precision@0.5": precision_score(y_test, preds, zero_division=0),
        "recall@0.5": recall_score(y_test, preds, zero_division=0),
        "f1@0.5": f1_score(y_test, preds, zero_division=0),
        "brier_score": brier_score_loss(y_test, proba),
    }


def recall_at_capacity(y_test, proba, capacity_frac: float) -> float:
    """Recall achieved if only the top `capacity_frac` of patients by
    predicted risk can receive an intervention — the metric that actually
    matches the decision problem, since a hospital never treats everyone."""
    n_capacity = int(len(proba) * capacity_frac)
    top_idx = np.argsort(proba)[::-1][:n_capacity]
    flagged = np.zeros_like(proba, dtype=int)
    flagged[top_idx] = 1
    return recall_score(y_test, flagged, zero_division=0)


def compare_models(fitted: dict, X_test, y_test, capacity_fracs=(0.05, 0.10, 0.20)) -> pd.DataFrame:
    rows = []
    for name, pipe in fitted.items():
        metrics = evaluate_model(pipe, X_test, y_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        for frac in capacity_fracs:
            metrics[f"recall@top{int(frac*100)}pct"] = recall_at_capacity(y_test, proba, frac)
        metrics["model"] = name
        rows.append(metrics)
    cols = ["model"] + [c for c in rows[0] if c != "model"]
    return pd.DataFrame(rows)[cols].set_index("model").round(4)


def get_calibration(pipe, X_test, y_test, n_bins=10):
    proba = pipe.predict_proba(X_test)[:, 1]
    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=n_bins, strategy="quantile")
    return mean_pred, frac_pos


if __name__ == "__main__":
    import joblib

    fitted = {
        name: joblib.load(f"data/processed/models/{name}.joblib")
        for name in ["logistic_regression", "random_forest", "xgboost"]
    }
    _, X_test, _, y_test = joblib.load("data/processed/models/splits.joblib")

    comparison = compare_models(fitted, X_test, y_test)
    print(comparison)
    comparison.to_csv("data/processed/model_comparison.csv")
