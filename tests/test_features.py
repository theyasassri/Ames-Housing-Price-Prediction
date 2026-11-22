import pandas as pd
import pytest

from ames_house_prices.features import (
    TotalSFTransformer,
    build_preprocessing_pipeline,
)


def test_totalsf_transformer_success() -> None:
    df = pd.DataFrame({
        "1stFlrSF": [800, 1000],
        "2ndFlrSF": [200, 500],
        "TotalBsmtSF": [500, 600],
        "OverallQual": [7, 8],
        "YearBuilt": [2000, 2010],
        "GarageCars": [2, 3],
        "ExtraColumn": ["ignore", "me"],
    })
    transformer = TotalSFTransformer()
    transformed = transformer.transform(df)

    expected_cols = ["TotalSF", "OverallQual", "YearBuilt", "GarageCars"]
    assert list(transformed.columns) == expected_cols
    assert list(transformed["TotalSF"]) == [1500, 2100]


def test_totalsf_missing_column() -> None:
    df = pd.DataFrame({
        "1stFlrSF": [800],
        "2ndFlrSF": [200],
        "OverallQual": [7],
        "YearBuilt": [2000],
        "GarageCars": [2],
    })
    transformer = TotalSFTransformer()
    with pytest.raises(ValueError, match="missing required feature columns"):
        transformer.transform(df)


def test_preprocessing_pipeline() -> None:
    df = pd.DataFrame({
        "1stFlrSF": [800, 1000],
        "2ndFlrSF": [200, 500],
        "TotalBsmtSF": [500, 600],
        "OverallQual": [7, 8],
        "YearBuilt": [2000, 2010],
        "GarageCars": [2, 3],
    })
    pipeline = build_preprocessing_pipeline(scale_features=True)
    out = pipeline.fit_transform(df)
    assert out.shape == (2, 4)
