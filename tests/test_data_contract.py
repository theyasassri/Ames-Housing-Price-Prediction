from pathlib import Path

import pandas as pd

from ames_house_prices.config import (
    DEFAULT_DATA_PATH,
    TARGET_COLUMN,
)
from ames_house_prices.validation import validate_training_data


def test_checked_in_dataset_contract() -> None:
    """Verify that the checked-in dataset train.csv meets strict contract invariants.

    Invariants:
    - File exists and is non-empty.
    - Exactly 1,460 rows and 81 columns (standard Ames dataset specification).
    - Contains required target column 'SalePrice' with no nulls, all finite, strictly > 0.
    - Contains all required model features with valid non-null numeric values.
    - Id column is unique with 0 duplicate records.
    """
    data_path = Path(DEFAULT_DATA_PATH)
    assert data_path.is_file(), f"Dataset file {data_path} is missing."

    df = pd.read_csv(data_path)
    assert len(df) == 1460, f"Expected 1,460 rows, found {len(df)}"
    assert len(df.columns) == 81, f"Expected 81 columns, found {len(df.columns)}"

    # Schema integrity check
    validate_training_data(df)

    # Identifiers check
    assert "Id" in df.columns
    assert df["Id"].is_unique, "Dataset contains duplicate IDs"

    # Target bounds sanity check (Ames residential prices)
    assert df[TARGET_COLUMN].min() >= 30000.0, "Found unusually low property price"
    assert df[TARGET_COLUMN].max() <= 1000000.0, "Found unusually high property price"

    # Feature distribution bounds check
    assert (df["OverallQual"] >= 1).all() and (df["OverallQual"] <= 10).all()
    assert (df["YearBuilt"] >= 1870).all() and (df["YearBuilt"] <= 2011).all()
    assert (df["1stFlrSF"] > 0).all()
