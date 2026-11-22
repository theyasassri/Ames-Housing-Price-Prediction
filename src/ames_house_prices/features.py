"""Feature preparation shared by training, evaluation, and prediction."""

import pandas as pd

from ames_house_prices.config import AREA_COLUMNS, RAW_FEATURE_COLUMNS


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the existing TotalSF feature and select the baseline predictors."""
    missing_columns = [column for column in RAW_FEATURE_COLUMNS if column not in frame.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Input data is missing required feature columns: {missing}")

    features = frame.loc[:, list(RAW_FEATURE_COLUMNS)].copy()
    features["TotalSF"] = features.loc[:, AREA_COLUMNS].sum(axis=1)
    return features.loc[:, ["TotalSF", "OverallQual", "YearBuilt", "GarageCars"]]
