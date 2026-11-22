"""Feature engineering transformers and pipeline definitions for Ames housing."""

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ames_house_prices.config import (
    AREA_COLUMNS,
    FEATURE_COLUMNS,
    RAW_FEATURE_COLUMNS,
)


class TotalSFTransformer(BaseEstimator, TransformerMixin):
    """Derives TotalSF from 1stFlrSF, 2ndFlrSF, and TotalBsmtSF, then keeps the selected feature subset."""

    def __init__(self) -> None:
        pass

    def fit(self, X: pd.DataFrame, y: Any = None) -> "TotalSFTransformer":
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Derive TotalSF and select only [TotalSF, OverallQual, YearBuilt, GarageCars]."""
        missing_columns = [column for column in RAW_FEATURE_COLUMNS if column not in X.columns]
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"Input data is missing required feature columns: {missing}")

        features = X.loc[:, list(RAW_FEATURE_COLUMNS)].copy()
        features["TotalSF"] = features.loc[:, list(AREA_COLUMNS)].sum(axis=1)
        return features.loc[:, list(FEATURE_COLUMNS)]


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the existing TotalSF feature and select baseline predictors.

    Kept for backward compatibility with existing code and notebook.
    """
    transformer = TotalSFTransformer()
    return transformer.transform(frame)


def build_preprocessing_pipeline(*, scale_features: bool = False) -> Pipeline:
    """Build a scikit-learn preprocessing pipeline.

    If scale_features=True, adds StandardScaler to the derived features.
    """
    steps: list[tuple[str, Any]] = [("totalsf_builder", TotalSFTransformer())]
    if scale_features:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)
