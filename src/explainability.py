"""
SHAP explainability for the CareRisk model. Produces both a global view
("what drives readmission risk overall") and per-patient local explanations
("why was this specific encounter flagged").
"""
import numpy as np
import pandas as pd
import shap


def get_shap_values(pipe, X: pd.DataFrame, sample_size: int | None = 2000, random_state=42):
    """Returns (explainer, shap_values, X_transformed_df, X_sample) for the
    tree model inside a fitted sklearn Pipeline(prep, clf)."""
    if sample_size is not None and len(X) > sample_size:
        X_sample = X.sample(sample_size, random_state=random_state)
    else:
        X_sample = X

    prep = pipe.named_steps["prep"]
    clf = pipe.named_steps["clf"]
    X_transformed = prep.transform(X_sample)
    feature_names = prep.get_feature_names_out()

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names, index=X_sample.index)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_transformed_df)
    return explainer, shap_values, X_transformed_df, X_sample


def global_importance(shap_values: np.ndarray, feature_names, top_n=20) -> pd.DataFrame:
    mean_abs = np.abs(shap_values).mean(axis=0)
    imp = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    return imp.sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)


def explain_patient(explainer, shap_values, X_transformed_df, row_idx: int, top_n=8) -> pd.DataFrame:
    """Top contributing features for one row, by |SHAP value|, signed so the
    caller can label direction (pushes risk up vs. down)."""
    row_shap = shap_values[row_idx]
    row_vals = X_transformed_df.iloc[row_idx]
    out = pd.DataFrame({
        "feature": X_transformed_df.columns,
        "value": row_vals.values,
        "shap_contribution": row_shap,
    })
    out["abs_contribution"] = out["shap_contribution"].abs()
    return out.sort_values("abs_contribution", ascending=False).head(top_n).drop(columns="abs_contribution")


if __name__ == "__main__":
    import joblib

    pipe = joblib.load("data/processed/models/xgboost.joblib")
    X_train, X_test, y_train, y_test = joblib.load("data/processed/models/splits.joblib")

    explainer, shap_values, X_transformed_df, X_sample = get_shap_values(pipe, X_test, sample_size=2000)
    imp = global_importance(shap_values, X_transformed_df.columns, top_n=15)
    print("Top 15 global drivers of readmission risk:")
    print(imp.to_string(index=False))
    imp.to_csv("data/processed/shap_global_importance.csv", index=False)

    print()
    print("Example local explanation, first patient in sample:")
    local = explain_patient(explainer, shap_values, X_transformed_df, row_idx=0)
    print(local.to_string(index=False))
