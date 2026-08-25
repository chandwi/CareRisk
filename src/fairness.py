"""
Subgroup performance and allocation-fairness audit.

We don't claim to "solve" fairness — we measure whether performance and who
gets selected for intervention differ meaningfully across demographic
subgroups, and how that trade-off moves as the classification threshold
changes. race/gender are in the raw UCI data; treated here strictly as
audit dimensions, never as model features (they aren't in models.NUMERIC_
FEATURES / CATEGORICAL_FEATURES).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def subgroup_metrics(y_true: np.ndarray, risk: np.ndarray, group: pd.Series, threshold: float = 0.5) -> pd.DataFrame:
    preds = (risk >= threshold).astype(int)
    rows = []
    for g in sorted(group.dropna().unique()):
        mask = (group == g).values
        if mask.sum() < 30:
            continue
        yt, pr, rk = y_true[mask], preds[mask], risk[mask]
        tn, fp, fn, tp = confusion_matrix(yt, pr, labels=[0, 1]).ravel()
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        fnr = fn / (tp + fn) if (tp + fn) > 0 else np.nan
        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        selection_rate = pr.mean()
        rows.append({
            "group": g,
            "n": int(mask.sum()),
            "base_rate": round(yt.mean(), 3),
            "mean_predicted_risk": round(rk.mean(), 3),
            "selection_rate": round(selection_rate, 3),
            "recall": round(recall, 3) if not np.isnan(recall) else None,
            "false_negative_rate": round(fnr, 3) if not np.isnan(fnr) else None,
            "precision": round(precision, 3) if not np.isnan(precision) else None,
        })
    return pd.DataFrame(rows)


def allocation_rates_by_group(selected: np.ndarray, group: pd.Series) -> pd.DataFrame:
    """Of the patients actually chosen for intervention under a capacity
    constraint, how does the selection rate vary by group? This is the
    metric that matters once optimization enters the picture — subgroup
    recall at a fixed global threshold doesn't capture what happens once
    only `capacity` patients are chosen."""
    df = pd.DataFrame({"selected": selected, "group": group.values})
    out = df.groupby("group").agg(n=("selected", "size"), n_selected=("selected", "sum"))
    out["selection_rate"] = (out["n_selected"] / out["n"]).round(3)
    return out.reset_index()


def threshold_fairness_sweep(y_true, risk, group: pd.Series, thresholds=(0.1, 0.2, 0.3, 0.4, 0.5)) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        sm = subgroup_metrics(y_true, risk, group, threshold=t)
        sm["threshold"] = t
        rows.append(sm)
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    import joblib

    pipe = joblib.load("data/processed/models/xgboost.joblib")
    X_train, X_test, y_train, y_test = joblib.load("data/processed/models/splits.joblib")

    df = pd.read_csv("data/processed/diabetic_data_features.csv")
    demo = df.loc[X_test.index, ["race", "gender", "age"]]

    risk = pipe.predict_proba(X_test)[:, 1]
    y_true = y_test.values

    print("=== By race ===")
    print(subgroup_metrics(y_true, risk, demo["race"]).to_string(index=False))
    print()
    print("=== By gender ===")
    print(subgroup_metrics(y_true, risk, demo["gender"]).to_string(index=False))
    print()
    print("=== By age band ===")
    print(subgroup_metrics(y_true, risk, demo["age"]).to_string(index=False))

    from optimization import expected_net_benefit, select_top_k_greedy
    net_benefit = expected_net_benefit(risk, 0.20, 10_000, 100)
    selected = select_top_k_greedy(net_benefit, capacity=500)
    print()
    print("=== Allocation rate by race, utility-optimized, capacity=500 ===")
    print(allocation_rates_by_group(selected, demo["race"]).to_string(index=False))
