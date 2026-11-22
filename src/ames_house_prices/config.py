"""Shared configuration for the current baseline workflow."""

from dataclasses import dataclass
from pathlib import Path

TARGET_COLUMN = "SalePrice"
IDENTIFIER_COLUMN = "Id"
AREA_COLUMNS = ("1stFlrSF", "2ndFlrSF", "TotalBsmtSF")
FEATURE_COLUMNS = ("TotalSF", "OverallQual", "YearBuilt", "GarageCars")
RAW_FEATURE_COLUMNS = (
    *AREA_COLUMNS,
    "OverallQual",
    "YearBuilt",
    "GarageCars",
)
DEFAULT_DATA_PATH = Path("train.csv")
DEFAULT_MODEL_PATH = Path("artifacts/ames_house_prices.joblib")
DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_CV_SPLITS = 5
SUPPORTED_MODELS = ("dummy", "linear", "ridge", "elasticnet", "random_forest", "gradient_boosting")
TEMPORAL_SPLIT_COLUMN = "YrSold"
DEFAULT_TEMPORAL_CUTOFF_YEAR = 2010


@dataclass(frozen=True)
class RunConfig:
    """Paths, split settings, and estimator options for CLI invocations."""

    data_path: Path = DEFAULT_DATA_PATH
    model_path: Path = DEFAULT_MODEL_PATH
    output_path: Path = Path("predictions.csv")
    random_state: int = DEFAULT_RANDOM_STATE
    test_size: float = DEFAULT_TEST_SIZE
    fit_intercept: bool = True
    model_name: str = "linear"
    cv_splits: int = DEFAULT_CV_SPLITS
    temporal_cutoff_year: int = DEFAULT_TEMPORAL_CUTOFF_YEAR
