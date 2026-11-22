from pathlib import Path

import pandas as pd
import pytest

from ames_house_prices.config import TARGET_COLUMN
from ames_house_prices.modeling import evaluate_holdout, fit_model, load_model, predict, save_model


@pytest.fixture
def sample_training_data() -> pd.DataFrame:
    return pd.DataFrame({
        "1stFlrSF": [800, 1000, 1200, 1400, 1600],
        "2ndFlrSF": [0, 500, 600, 700, 800],
        "TotalBsmtSF": [800, 900, 1000, 1100, 1200],
        "OverallQual": [5, 6, 7, 8, 9],
        "YearBuilt": [1980, 1990, 2000, 2005, 2015],
        "GarageCars": [1, 2, 2, 3, 3],
        TARGET_COLUMN: [120000.0, 180000.0, 240000.0, 310000.0, 420000.0],
    })


def test_fit_and_predict_pipeline(sample_training_data: pd.DataFrame, tmp_path: Path) -> None:
    pipeline = fit_model(sample_training_data)

    test_input = pd.DataFrame({
        "1stFlrSF": [1000],
        "2ndFlrSF": [500],
        "TotalBsmtSF": [900],
        "OverallQual": [6],
        "YearBuilt": [1990],
        "GarageCars": [2],
    })

    preds = predict(test_input, pipeline)
    assert len(preds) == 1
    assert preds[0] > 0

    # Test serialization and loading
    model_file = tmp_path / "model.joblib"
    save_model(pipeline, model_file)
    loaded_pipe = load_model(model_file)

    loaded_preds = predict(test_input, loaded_pipe)
    assert loaded_preds[0] == pytest.approx(preds[0])


def test_evaluate_holdout(sample_training_data: pd.DataFrame) -> None:
    eval_result = evaluate_holdout(sample_training_data, test_size=0.4, random_state=42)
    assert eval_result.mae >= 0
    assert eval_result.rmse >= 0
    assert len(eval_result.predictions) == 2
