"""Data validation and schema boundary checks for training and inference."""

from typing import Any

import numpy as np
import pandas as pd

from ames_house_prices.config import (
    IDENTIFIER_COLUMN,
    RAW_FEATURE_COLUMNS,
    TARGET_COLUMN,
)


class DataValidationError(ValueError):
    """Raised when an input DataFrame fails schema or data integrity validation."""


def validate_training_data(frame: pd.DataFrame) -> None:
    """Validate data loaded for training or holdout evaluation.

    Checks:
    - Non-empty dataframe
    - No duplicate column names
    - Presence of required target column
    - Presence of raw feature columns
    - Target integrity: non-null, finite, and strictly positive (> 0)
    - Feature integrity: numeric data types and finite values
    """
    if frame.empty:
        raise DataValidationError("Training data contains no rows.")

    duplicate_cols = frame.columns[frame.columns.duplicated()].tolist()
    if duplicate_cols:
        raise DataValidationError(f"Training data contains duplicate column names: {duplicate_cols}")

    if TARGET_COLUMN not in frame.columns:
        raise DataValidationError(f"Training data is missing required target column: '{TARGET_COLUMN}'")

    missing_features = [col for col in RAW_FEATURE_COLUMNS if col not in frame.columns]
    if missing_features:
        raise DataValidationError(
            f"Training data is missing required feature columns: {', '.join(missing_features)}"
        )

    # Validate target
    target_series = frame[TARGET_COLUMN]
    if target_series.isna().any():
        raise DataValidationError(f"Target '{TARGET_COLUMN}' contains null or missing values.")

    try:
        target_numeric = pd.to_numeric(target_series, errors="raise")
    except Exception as exc:
        raise DataValidationError(f"Target '{TARGET_COLUMN}' must be numeric: {exc}") from exc

    if not np.isfinite(target_numeric.to_numpy(dtype=float)).all():
        raise DataValidationError(f"Target '{TARGET_COLUMN}' contains non-finite values (inf or -inf).")

    if (target_numeric <= 0).any():
        raise DataValidationError(f"Target '{TARGET_COLUMN}' contains non-positive values (<= 0).")

    # Validate required raw features with row-specific index tracking
    for col in RAW_FEATURE_COLUMNS:
        series = frame[col]
        null_indices = series[series.isna()].index.tolist()
        if null_indices:
            raise DataValidationError(
                f"Feature column '{col}' contains missing values at row indices: {null_indices[:10]}"
            )
        try:
            numeric_series = pd.to_numeric(series, errors="raise")
        except Exception as exc:
            raise DataValidationError(f"Feature column '{col}' must be numeric: {exc}") from exc
        non_finite_indices = numeric_series[~np.isfinite(numeric_series.to_numpy(dtype=float))].index.tolist()
        if non_finite_indices:
            raise DataValidationError(
                f"Feature column '{col}' contains non-finite values at row indices: {non_finite_indices[:10]}"
            )


def validate_prediction_data(frame: pd.DataFrame) -> None:
    """Validate data provided for batch prediction.

    Checks:
    - Non-empty dataframe
    - Explicit rejection of target column 'SalePrice' to prevent leakage
    - No duplicate column names
    - Presence of raw feature columns
    - Feature integrity: numeric data types, non-null, and finite values with row-specific indices
    """
    if frame.empty:
        raise DataValidationError("Prediction data contains no rows.")

    duplicate_cols = frame.columns[frame.columns.duplicated()].tolist()
    if duplicate_cols:
        raise DataValidationError(f"Prediction data contains duplicate column names: {duplicate_cols}")

    if TARGET_COLUMN in frame.columns:
        raise DataValidationError(
            f"Target column '{TARGET_COLUMN}' detected in prediction input. "
            "To prevent target leakage, prediction inputs must not contain the target."
        )

    missing_features = [col for col in RAW_FEATURE_COLUMNS if col not in frame.columns]
    if missing_features:
        raise DataValidationError(
            f"Prediction data is missing required feature columns: {', '.join(missing_features)}"
        )

    for col in RAW_FEATURE_COLUMNS:
        series = frame[col]
        null_indices = series[series.isna()].index.tolist()
        if null_indices:
            raise DataValidationError(
                f"Feature column '{col}' contains missing values at row indices: {null_indices[:10]}"
            )
        try:
            numeric_series = pd.to_numeric(series, errors="raise")
        except Exception as exc:
            raise DataValidationError(f"Feature column '{col}' must be numeric: {exc}") from exc
        non_finite_indices = numeric_series[~np.isfinite(numeric_series.to_numpy(dtype=float))].index.tolist()
        if non_finite_indices:
            raise DataValidationError(
                f"Feature column '{col}' contains non-finite values at row indices: {non_finite_indices[:10]}"
            )


def generate_data_audit(frame: pd.DataFrame) -> dict[str, Any]:
    """Generate summary data audit statistics for a dataset."""
    audit: dict[str, Any] = {
        "num_rows": len(frame),
        "num_columns": len(frame.columns),
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_values_by_column": frame.isna().sum().loc[lambda x: x > 0].to_dict(),
    }
    if IDENTIFIER_COLUMN in frame.columns:
        audit["duplicate_identifiers"] = int(frame[IDENTIFIER_COLUMN].duplicated().sum())

    if TARGET_COLUMN in frame.columns:
        target_series = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
        audit["target_summary"] = {
            "mean": float(target_series.mean()),
            "std": float(target_series.std()),
            "min": float(target_series.min()),
            "25%": float(target_series.quantile(0.25)),
            "50%": float(target_series.median()),
            "75%": float(target_series.quantile(0.75)),
            "max": float(target_series.max()),
        }
    return audit
