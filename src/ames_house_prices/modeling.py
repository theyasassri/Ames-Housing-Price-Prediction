"""Baseline model training and holdout evaluation."""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from ames_house_prices.config import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from ames_house_prices.features import build_features


@dataclass(frozen=True)
class HoldoutEvaluation:
    """Results from the project's existing deterministic random holdout."""

    model: LinearRegression
    actual: pd.Series
    predictions: pd.Series
    mae: float

    @property
    def results(self) -> pd.DataFrame:
        """Return actual, predicted, and signed-error values indexed by input row."""
        return pd.DataFrame(
            {
                "Actual($)": self.actual,
                "Predicted($)": self.predictions,
                "Variance(Error)($)": self.predictions - self.actual,
            }
        )


def _training_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Training data must include the target column {TARGET_COLUMN!r}.")

    features = build_features(frame)
    target = frame[TARGET_COLUMN]
    if target.isna().any():
        raise ValueError(f"Training target {TARGET_COLUMN!r} contains missing values.")
    if not np.isfinite(target.to_numpy(dtype=float)).all():
        raise ValueError(f"Training target {TARGET_COLUMN!r} contains non-finite values.")
    if features.isna().any().any():
        raise ValueError(
            "Baseline model features contain missing values. "
            "Missing-value preprocessing is planned for Phase 2."
        )

    return features.loc[:, list(FEATURE_COLUMNS)], target


def evaluate_holdout(
    frame: pd.DataFrame,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    test_size: float = DEFAULT_TEST_SIZE,
    fit_intercept: bool = True,
) -> HoldoutEvaluation:
    """Fit and evaluate the current linear regression on a deterministic holdout."""
    features, target = _training_data(frame)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(x_train, y_train)
    predictions = pd.Series(model.predict(x_test), index=y_test.index, name="Predicted($)")

    return HoldoutEvaluation(
        model=model,
        actual=y_test.rename("Actual($)"),
        predictions=predictions,
        mae=float(mean_absolute_error(y_test, predictions)),
    )


def fit_model(frame: pd.DataFrame, *, fit_intercept: bool = True) -> LinearRegression:
    """Fit the current baseline model on all supplied training rows."""
    features, target = _training_data(frame)
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(features, target)
    return model


def save_model(model: LinearRegression, path: str | Path) -> Path:
    """Persist a model artifact to the requested local path."""
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model_path


def load_model(path: str | Path) -> LinearRegression:
    """Load a locally generated model artifact."""
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact does not exist: {model_path}")
    return joblib.load(model_path)


def predict(frame: pd.DataFrame, model: LinearRegression) -> np.ndarray:
    """Predict prices for rows using the shared baseline feature preparation."""
    return model.predict(build_features(frame))
