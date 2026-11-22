"""Model definitions, cross-validation benchmarking, and evaluation metrics."""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline

from ames_house_prices.config import (
    DEFAULT_CV_SPLITS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEMPORAL_CUTOFF_YEAR,
    DEFAULT_TEST_SIZE,
    SUPPORTED_MODELS,
    TARGET_COLUMN,
    TEMPORAL_SPLIT_COLUMN,
)
from ames_house_prices.features import TotalSFTransformer, build_preprocessing_pipeline
from ames_house_prices.validation import validate_prediction_data, validate_training_data


@dataclass(frozen=True)
class RegressionMetrics:
    """Standard evaluation metrics for regression models."""

    mae: float
    rmse: float
    r2: float
    medae: float

    def to_dict(self) -> dict[str, float]:
        return {
            "MAE": self.mae,
            "RMSE": self.rmse,
            "R2": self.r2,
            "MedAE": self.medae,
        }


@dataclass(frozen=True)
class HoldoutEvaluation:
    """Results from an evaluation split."""

    model: BaseEstimator
    actual: pd.Series
    predictions: pd.Series
    metrics: RegressionMetrics

    @property
    def mae(self) -> float:
        return self.metrics.mae

    @property
    def rmse(self) -> float:
        return self.metrics.rmse

    @property
    def r2(self) -> float:
        return self.metrics.r2

    @property
    def medae(self) -> float:
        return self.metrics.medae

    @property
    def results(self) -> pd.DataFrame:
        """Return actual, predicted, and error values."""
        return pd.DataFrame(
            {
                "Actual($)": self.actual,
                "Predicted($)": self.predictions,
                "Variance(Error)($)": self.predictions - self.actual,
            }
        )


@dataclass(frozen=True)
class ModelBenchmarkResult:
    """Cross-validation and holdout metrics for a model candidate."""

    model_name: str
    cv_mae_mean: float
    cv_mae_std: float
    cv_rmse_mean: float
    cv_rmse_std: float
    cv_r2_mean: float
    cv_r2_std: float
    holdout_metrics: RegressionMetrics


def compute_metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> RegressionMetrics:
    """Compute regression metrics: MAE, RMSE, R2, and Median Absolute Error."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    medae = float(median_absolute_error(y_true, y_pred))
    return RegressionMetrics(mae=mae, rmse=rmse, r2=r2, medae=medae)


def create_estimator(
    model_name: str = "linear",
    *,
    fit_intercept: bool = True,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[BaseEstimator, bool]:
    """Instantiate an estimator and return whether feature scaling should be applied."""
    name = model_name.lower()
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model '{model_name}'. Choose from: {', '.join(SUPPORTED_MODELS)}")

    if name == "dummy":
        return DummyRegressor(strategy="mean"), False
    if name == "linear":
        return LinearRegression(fit_intercept=fit_intercept), False
    if name == "ridge":
        return Ridge(alpha=1.0, fit_intercept=fit_intercept, random_state=random_state), True
    if name == "elasticnet":
        return ElasticNet(alpha=0.1, l1_ratio=0.5, fit_intercept=fit_intercept, random_state=random_state), True
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=100, max_depth=8, random_state=random_state), False
    if name == "gradient_boosting":
        return GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=random_state), False

    raise ValueError(f"Unknown model name {model_name}")


def build_model_pipeline(
    model_name: str = "linear",
    *,
    fit_intercept: bool = True,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Pipeline:
    """Create an end-to-end Pipeline that includes feature extraction and the estimator."""
    estimator, scale_features = create_estimator(
        model_name=model_name,
        fit_intercept=fit_intercept,
        random_state=random_state,
    )
    preprocessor = build_preprocessing_pipeline(scale_features=scale_features)
    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", estimator),
    ])


def evaluate_holdout(
    frame: pd.DataFrame,
    *,
    model_name: str = "linear",
    random_state: int = DEFAULT_RANDOM_STATE,
    test_size: float = DEFAULT_TEST_SIZE,
    fit_intercept: bool = True,
) -> HoldoutEvaluation:
    """Fit and evaluate a chosen model pipeline on a deterministic holdout."""
    validate_training_data(frame)

    y = frame[TARGET_COLUMN]
    X = frame.drop(columns=[TARGET_COLUMN])

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    pipeline = build_model_pipeline(
        model_name=model_name,
        fit_intercept=fit_intercept,
        random_state=random_state,
    )
    pipeline.fit(x_train, y_train)

    preds = pipeline.predict(x_test)
    predictions = pd.Series(preds, index=y_test.index, name="Predicted($)")
    metrics = compute_metrics(y_test, predictions)

    return HoldoutEvaluation(
        model=pipeline,
        actual=y_test.rename("Actual($)"),
        predictions=predictions,
        metrics=metrics,
    )


def evaluate_temporal_holdout(
    frame: pd.DataFrame,
    *,
    model_name: str = "linear",
    cutoff_year: int = DEFAULT_TEMPORAL_CUTOFF_YEAR,
    fit_intercept: bool = True,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> HoldoutEvaluation:
    """Evaluate generalization using a time-aware split based on YrSold.

    Training data: YrSold < cutoff_year
    Testing data:  YrSold >= cutoff_year
    """
    validate_training_data(frame)
    if TEMPORAL_SPLIT_COLUMN not in frame.columns:
        raise ValueError(f"Temporal column '{TEMPORAL_SPLIT_COLUMN}' missing from training data.")

    train_mask = frame[TEMPORAL_SPLIT_COLUMN] < cutoff_year
    test_mask = frame[TEMPORAL_SPLIT_COLUMN] >= cutoff_year

    if not train_mask.any() or not test_mask.any():
        raise ValueError(
            f"Cutoff year {cutoff_year} does not produce non-empty train and test sets."
        )

    y_train = frame.loc[train_mask, TARGET_COLUMN]
    X_train = frame.loc[train_mask].drop(columns=[TARGET_COLUMN])

    y_test = frame.loc[test_mask, TARGET_COLUMN]
    X_test = frame.loc[test_mask].drop(columns=[TARGET_COLUMN])

    pipeline = build_model_pipeline(
        model_name=model_name,
        fit_intercept=fit_intercept,
        random_state=random_state,
    )
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    predictions = pd.Series(preds, index=y_test.index, name="Predicted($)")
    metrics = compute_metrics(y_test, predictions)

    return HoldoutEvaluation(
        model=pipeline,
        actual=y_test.rename("Actual($)"),
        predictions=predictions,
        metrics=metrics,
    )


def benchmark_models(
    frame: pd.DataFrame,
    *,
    candidate_models: tuple[str, ...] = SUPPORTED_MODELS,
    cv_splits: int = DEFAULT_CV_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
    test_size: float = DEFAULT_TEST_SIZE,
    fit_intercept: bool = True,
) -> list[ModelBenchmarkResult]:
    """Run K-Fold cross validation on development data and evaluate on an untouched final holdout."""
    validate_training_data(frame)

    y = frame[TARGET_COLUMN]
    X = frame.drop(columns=[TARGET_COLUMN])

    # Keep a reserved untouched holdout
    x_dev, x_holdout, y_dev, y_holdout = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    results: list[ModelBenchmarkResult] = []

    for model_name in candidate_models:
        fold_maes: list[float] = []
        fold_rmses: list[float] = []
        fold_r2s: list[float] = []

        for train_idx, val_idx in kf.split(x_dev, y_dev):
            x_tr, x_val = x_dev.iloc[train_idx], x_dev.iloc[val_idx]
            y_tr, y_val = y_dev.iloc[train_idx], y_dev.iloc[val_idx]

            pipe = build_model_pipeline(
                model_name=model_name,
                fit_intercept=fit_intercept,
                random_state=random_state,
            )
            pipe.fit(x_tr, y_tr)
            val_preds = pipe.predict(x_val)
            metrics = compute_metrics(y_val, val_preds)

            fold_maes.append(metrics.mae)
            fold_rmses.append(metrics.rmse)
            fold_r2s.append(metrics.r2)

        # Train on entire dev set and evaluate on final untouched holdout
        final_pipe = build_model_pipeline(
            model_name=model_name,
            fit_intercept=fit_intercept,
            random_state=random_state,
        )
        final_pipe.fit(x_dev, y_dev)
        holdout_preds = final_pipe.predict(x_holdout)
        holdout_metrics = compute_metrics(y_holdout, holdout_preds)

        results.append(
            ModelBenchmarkResult(
                model_name=model_name,
                cv_mae_mean=float(np.mean(fold_maes)),
                cv_mae_std=float(np.std(fold_maes)),
                cv_rmse_mean=float(np.mean(fold_rmses)),
                cv_rmse_std=float(np.std(fold_rmses)),
                cv_r2_mean=float(np.mean(fold_r2s)),
                cv_r2_std=float(np.std(fold_r2s)),
                holdout_metrics=holdout_metrics,
            )
        )

    # Sort candidates by CV MAE ascending (best first)
    results.sort(key=lambda r: r.cv_mae_mean)
    return results


def fit_model(
    frame: pd.DataFrame,
    *,
    model_name: str = "linear",
    fit_intercept: bool = True,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Pipeline:
    """Fit the end-to-end model pipeline on all supplied training rows."""
    validate_training_data(frame)

    y = frame[TARGET_COLUMN]
    X = frame.drop(columns=[TARGET_COLUMN])

    pipeline = build_model_pipeline(
        model_name=model_name,
        fit_intercept=fit_intercept,
        random_state=random_state,
    )
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
    transformer = TotalSFTransformer()
    features = transformer.transform(frame)
    return model.predict(features)
