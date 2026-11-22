"""Dataset loading and ingestion checks for training and evaluation."""

from pathlib import Path

import pandas as pd

from ames_house_prices.validation import validate_training_data


def load_training_data(path: str | Path) -> pd.DataFrame:
    """Load the training CSV and validate required schema and data constraints."""
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Training data file does not exist: {data_path}")

    frame = pd.read_csv(data_path)
    validate_training_data(frame)
    return frame
