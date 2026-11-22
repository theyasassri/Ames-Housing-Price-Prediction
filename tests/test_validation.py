import pandas as pd
import pytest

from ames_house_prices.config import TARGET_COLUMN
from ames_house_prices.validation import (
    DataValidationError,
    generate_data_audit,
    validate_prediction_data,
    validate_training_data,
)


@pytest.fixture
def valid_training_df() -> pd.DataFrame:
    return pd.DataFrame({
        "1stFlrSF": [856, 1262],
        "2ndFlrSF": [854, 0],
        "TotalBsmtSF": [856, 1262],
        "OverallQual": [7, 6],
        "YearBuilt": [2003, 1976],
        "GarageCars": [2, 2],
        TARGET_COLUMN: [208500.0, 181500.0],
    })


@pytest.fixture
def valid_prediction_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Id": [101, 102],
        "1stFlrSF": [856, 1262],
        "2ndFlrSF": [854, 0],
        "TotalBsmtSF": [856, 1262],
        "OverallQual": [7, 6],
        "YearBuilt": [2003, 1976],
        "GarageCars": [2, 2],
    })


def test_validate_training_data_success(valid_training_df: pd.DataFrame) -> None:
    validate_training_data(valid_training_df)


def test_validate_training_data_empty() -> None:
    with pytest.raises(DataValidationError, match="Training data contains no rows"):
        validate_training_data(pd.DataFrame())


def test_validate_training_data_missing_target(valid_training_df: pd.DataFrame) -> None:
    df = valid_training_df.drop(columns=[TARGET_COLUMN])
    with pytest.raises(DataValidationError, match=f"missing required target column: '{TARGET_COLUMN}'"):
        validate_training_data(df)


def test_validate_training_data_non_positive_target(valid_training_df: pd.DataFrame) -> None:
    df = valid_training_df.copy()
    df.loc[0, TARGET_COLUMN] = 0.0
    with pytest.raises(DataValidationError, match="non-positive values"):
        validate_training_data(df)


def test_validate_training_data_missing_feature(valid_training_df: pd.DataFrame) -> None:
    df = valid_training_df.drop(columns=["1stFlrSF"])
    with pytest.raises(DataValidationError, match="missing required feature columns"):
        validate_training_data(df)


def test_validate_prediction_data_success(valid_prediction_df: pd.DataFrame) -> None:
    validate_prediction_data(valid_prediction_df)


def test_validate_prediction_data_rejects_target(valid_prediction_df: pd.DataFrame) -> None:
    df = valid_prediction_df.copy()
    df[TARGET_COLUMN] = [200000.0, 150000.0]
    with pytest.raises(DataValidationError, match=f"Target column '{TARGET_COLUMN}' detected in prediction input"):
        validate_prediction_data(df)


def test_validate_prediction_data_missing_values(valid_prediction_df: pd.DataFrame) -> None:
    df = valid_prediction_df.copy()
    df.loc[0, "OverallQual"] = None
    with pytest.raises(DataValidationError, match="contains missing values"):
        validate_prediction_data(df)


def test_generate_data_audit(valid_training_df: pd.DataFrame) -> None:
    audit = generate_data_audit(valid_training_df)
    assert audit["num_rows"] == 2
    assert audit["num_columns"] == 7
    assert audit["duplicate_rows"] == 0
    assert "target_summary" in audit
