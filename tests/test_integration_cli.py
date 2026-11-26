import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_cli_end_to_end_workflow(tmp_path: Path) -> None:
    """Integration test verifying full CLI lifecycle: train -> inspect-manifest -> predict -> audit."""
    train_csv = tmp_path / "synthetic_train.csv"
    predict_csv = tmp_path / "synthetic_predict.csv"
    model_path = tmp_path / "model.joblib"
    manifest_path = tmp_path / "manifest.json"
    output_predictions = tmp_path / "predictions.csv"

    # Create synthetic dataset
    train_data = pd.DataFrame({
        "Id": [1, 2, 3, 4, 5, 6, 7, 8],
        "1stFlrSF": [800, 1000, 1200, 1400, 900, 1100, 1300, 1500],
        "2ndFlrSF": [0, 500, 600, 700, 0, 450, 550, 650],
        "TotalBsmtSF": [800, 900, 1000, 1100, 850, 950, 1050, 1150],
        "OverallQual": [5, 6, 7, 8, 5, 6, 7, 8],
        "YearBuilt": [1980, 1990, 2000, 2010, 1985, 1995, 2005, 2015],
        "GarageCars": [1, 2, 2, 3, 1, 2, 2, 3],
        "SalePrice": [120000.0, 180000.0, 240000.0, 310000.0, 130000.0, 190000.0, 250000.0, 320000.0],
    })
    train_data.to_csv(train_csv, index=False)

    predict_data = train_data.drop(columns=["SalePrice"])
    predict_data.to_csv(predict_csv, index=False)

    # 1. Run 'train' command
    cmd_train = [
        sys.executable,
        "-m",
        "ames_house_prices.cli",
        "train",
        "--data",
        str(train_csv),
        "--model",
        str(model_path),
        "--manifest",
        str(manifest_path),
        "--model-name",
        "gradient_boosting",
    ]
    res_train = subprocess.run(cmd_train, capture_output=True, text=True, check=True)
    assert "Trained [gradient_boosting]" in res_train.stdout
    assert model_path.is_file()
    assert manifest_path.is_file()

    # 2. Run 'inspect-manifest' command
    cmd_inspect = [
        sys.executable,
        "-m",
        "ames_house_prices.cli",
        "inspect-manifest",
        "--manifest",
        str(manifest_path),
    ]
    res_inspect = subprocess.run(cmd_inspect, capture_output=True, text=True, check=True)
    assert "=== Artifact Lineage Manifest" in res_inspect.stdout
    assert "gradient_boosting" in res_inspect.stdout

    # 3. Run 'predict' command
    cmd_predict = [
        sys.executable,
        "-m",
        "ames_house_prices.cli",
        "predict",
        "--data",
        str(predict_csv),
        "--model",
        str(model_path),
        "--output",
        str(output_predictions),
    ]
    res_predict = subprocess.run(cmd_predict, capture_output=True, text=True, check=True)
    assert "Wrote 8 predictions" in res_predict.stdout
    assert output_predictions.is_file()

    preds_df = pd.read_csv(output_predictions)
    assert len(preds_df) == 8
    assert "Id" in preds_df.columns
    assert "PredictedSalePrice" in preds_df.columns
    assert (preds_df["PredictedSalePrice"] > 0).all()

    # 4. Run 'audit' command
    cmd_audit = [
        sys.executable,
        "-m",
        "ames_house_prices.cli",
        "audit",
        "--data",
        str(train_csv),
    ]
    res_audit = subprocess.run(cmd_audit, capture_output=True, text=True, check=True)
    assert "Dataset Audit Summary:" in res_audit.stdout
    assert "Rows: 8" in res_audit.stdout
