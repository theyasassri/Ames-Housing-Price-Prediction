import pandas as pd
import pytest

from ames_house_prices.analysis import run_error_analysis
from ames_house_prices.config import TARGET_COLUMN


@pytest.fixture
def sample_analysis_data() -> pd.DataFrame:
    return pd.DataFrame({
        "Id": range(1, 21),
        "1stFlrSF": [800 + i * 50 for i in range(20)],
        "2ndFlrSF": [0 if i % 2 == 0 else 400 for i in range(20)],
        "TotalBsmtSF": [800 + i * 40 for i in range(20)],
        "OverallQual": [5 + (i % 5) for i in range(20)],
        "YearBuilt": [1980 + i * 2 for i in range(20)],
        "GarageCars": [1 if i < 5 else (2 if i < 15 else 3) for i in range(20)],
        "Neighborhood": ["CollgCr" if i < 10 else "NAmes" for i in range(20)],
        "YrSold": [2006 + (i % 5) for i in range(20)],
        TARGET_COLUMN: [100000.0 + i * 15000.0 for i in range(20)],
    })


def test_run_error_analysis(sample_analysis_data: pd.DataFrame) -> None:
    report = run_error_analysis(
        sample_analysis_data,
        model_name="gradient_boosting",
        test_size=0.3,
        random_state=42,
    )
    assert report.total_samples == 6
    assert report.overall_mae > 0
    assert report.overall_rmse > 0
    assert len(report.price_bracket_slices) > 0
    assert len(report.high_leverage_outliers) > 0
    summary = report.summary_table()
    assert "=== Error Analysis Report for 'gradient_boosting' ===" in summary
