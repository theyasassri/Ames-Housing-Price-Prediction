import pandas as pd
import pytest

from ames_house_prices.config import SUPPORTED_MODELS, TARGET_COLUMN
from ames_house_prices.modeling import (
    benchmark_models,
    evaluate_holdout,
    evaluate_temporal_holdout,
    fit_model,
    predict,
)


@pytest.fixture
def sample_training_data() -> pd.DataFrame:
    return pd.DataFrame({
        "1stFlrSF": [800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600],
        "2ndFlrSF": [0, 500, 600, 700, 800, 0, 500, 600, 700, 800],
        "TotalBsmtSF": [800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700],
        "OverallQual": [5, 6, 7, 8, 9, 5, 6, 7, 8, 9],
        "YearBuilt": [1980, 1990, 2000, 2005, 2015, 1985, 1995, 2002, 2008, 2018],
        "GarageCars": [1, 2, 2, 3, 3, 1, 2, 2, 3, 3],
        "YrSold": [2006, 2007, 2008, 2009, 2010, 2006, 2007, 2008, 2009, 2010],
        TARGET_COLUMN: [
            120000.0,
            180000.0,
            240000.0,
            310000.0,
            420000.0,
            135000.0,
            195000.0,
            255000.0,
            325000.0,
            435000.0,
        ],
    })


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_all_models_fit_and_predict(sample_training_data: pd.DataFrame, model_name: str) -> None:
    pipeline = fit_model(sample_training_data, model_name=model_name)

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


def test_evaluate_holdout_metrics(sample_training_data: pd.DataFrame) -> None:
    eval_result = evaluate_holdout(sample_training_data, model_name="linear", test_size=0.3, random_state=42)
    assert eval_result.mae > 0
    assert eval_result.rmse > 0
    assert eval_result.medae > 0
    assert len(eval_result.predictions) == 3


def test_evaluate_temporal_holdout(sample_training_data: pd.DataFrame) -> None:
    eval_result = evaluate_temporal_holdout(
        sample_training_data,
        model_name="ridge",
        cutoff_year=2010,
    )
    assert eval_result.mae > 0
    assert len(eval_result.predictions) == 2


def test_benchmark_models(sample_training_data: pd.DataFrame) -> None:
    candidates = ("dummy", "linear", "ridge", "gradient_boosting")
    results = benchmark_models(
        sample_training_data,
        candidate_models=candidates,
        cv_splits=2,
        test_size=0.3,
        random_state=42,
    )
    assert len(results) == len(candidates)
    dummy_res = next(r for r in results if r.model_name == "dummy")
    linear_res = next(r for r in results if r.model_name == "linear")
    assert linear_res.cv_mae_mean < dummy_res.cv_mae_mean
