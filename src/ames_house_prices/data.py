"""Dataset loading for training and evaluation."""

from pathlib import Path

import pandas as pd

from ames_house_prices.config import RAW_FEATURE_COLUMNS, TARGET_COLUMN


def load_training_data(path: str | Path) -> pd.DataFrame:
    """Load the training CSV and check the columns needed by the baseline model."""
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Training data file does not exist: {data_path}")

    frame = pd.read_csv(data_path)
    required_columns = (*RAW_FEATURE_COLUMNS, TARGET_COLUMN)
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Training data is missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"Training data contains no rows: {data_path}")

    return frame
