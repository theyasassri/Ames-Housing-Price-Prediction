"""Baseline model training, holdout evaluation, and pipeline serialization."""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ames_house_prices.config import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    TARGET_COLUMN,
)
from ames_house_prices.features import TotalSFTransformer, build_preprocessing_pipeline
from ames_house_prices.validation import validate_prediction_data, validate_training_data


@dataclass(frozen=True)
class HoldoutEvaluation:
    """Results from the project's deterministic random holdout."""

    model: BaseEstimator
    actual: pd.Series
    predictions: pd.Series
    mae: float
    rmse: float
    r2: float

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


def build_model_pipeline(*, fit_intercept: bool = True, scale_features: bool = False) -> Pipeline:
    """Create an end-to-end Pipeline that includes feature extraction and the estimator."""
    preprocessor = build_preprocessing_pipeline(scale_features=scale_features)
    estimator = LinearRegression(fit_intercept=fit_intercept)
    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", estimator),
    ])


def evaluate_holdout(
    frame: pd.DataFrame,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    test_size: float = DEFAULT_TEST_SIZE,
    fit_intercept: bool = True,
) -> HoldoutEvaluation:
    """Fit and evaluate the end-to-end pipeline on a deterministic holdout."""
    validate_training_data(frame)

    y = frame[TARGET_COLUMN]
    X = frame.drop(columns=[TARGET_COLUMN])

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    pipeline = build_model_pipeline(fit_intercept=fit_intercept)
    pipeline.fit(x_train, y_train)

    preds = pipeline.predict(x_test)
    predictions = pd.Series(preds, index=y_test.index, name="Predicted($)")

    mae = float(mean_absolute_error(y_test, predictions))
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = float(r2_score(y_test, predictions))

    return HoldoutEvaluation(
        model=pipeline,
        actual=y_test.rename("Actual($)"),
        predictions=predictions,
        mae=mae,
        rmse=rmse,
        r2=r2,
    )


def fit_model(frame: pd.DataFrame, *, fit_intercept: bool = True) -> Pipeline:
    """Fit the end-to-end model pipeline on all supplied training rows."""
    validate_training_data(frame)

    y = frame[TARGET_COLUMN]
    X = frame.drop(columns=[TARGET_COLUMN])

    pipeline = build_model_pipeline(fit_intercept=fit_intercept)
    pipeline.fit(X, y)
    return pipeline


def save_model(model: BaseEstimator, path: str | Path) -> Path:
    """Persist a model pipeline artifact to the requested local path."""
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model_path


def load_model(path: str | Path) -> BaseEstimator:
    """Load a locally generated model artifact (Pipeline or BaseEstimator)."""
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact does not exist: {model_path}")
    return joblib.load(model_path)


def predict(frame: pd.DataFrame, model: BaseEstimator) -> np.ndarray:
    """Validate prediction data and predict prices using the trained pipeline or model."""
    validate_prediction_data(frame)
    if isinstance(model, Pipeline):
        return model.predict(frame)
    # Support backward compatibility if a raw LinearRegression was loaded
    transformer = TotalSFTransformer()
    features = transformer.transform(frame)
    return model.predict(features)
