"""
Shared sklearn-compatible transformers. Kept in their own module (rather than
inline in models.py) so joblib can always resolve them by the same import
path regardless of which script trained vs. loads the pickled pipeline.
"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RareCategoryCollapser(BaseEstimator, TransformerMixin):
    """Merges categories seen fewer than `min_count` times in the training
    fold into that column's most frequent value, rather than into a new
    'Other' bucket. Several columns here (discharge_disposition_id,
    individual medication Up/Down levels, rare medical specialties) have
    levels with under ~100 observations; a lone rare level, or an 'Other'
    bucket that just pools two rare levels, is still small enough for tree
    models to fit noise to it — SHAP importance ends up dominated by
    statistically unreliable rare levels instead of real drivers. Routing
    the tail into the mode (not a fresh small bucket) actually removes the
    sparse column. Fit only on train to avoid leakage."""

    def __init__(self, min_count: int = 300):
        self.min_count = min_count

    def fit(self, X: pd.DataFrame, y=None):
        self.frequent_ = {}
        self.mode_ = {}
        for col in X.columns:
            counts = X[col].value_counts()
            self.frequent_[col] = set(counts[counts >= self.min_count].index)
            self.mode_[col] = counts.idxmax()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in X.columns:
            is_rare = ~X[col].isin(self.frequent_[col])
            X.loc[is_rare, col] = self.mode_[col]
        return X

    def get_feature_names_out(self, input_features=None):
        return pd.Index(input_features if input_features is not None else self.frequent_.keys())
