from pathlib import Path

from ames_house_prices.manifest import (
    ArtifactManifest,
    compute_file_sha256,
    create_artifact_manifest,
)


def test_compute_file_sha256(tmp_path: Path) -> None:
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello ames housing", encoding="utf-8")
    digest = compute_file_sha256(test_file)
    assert len(digest) == 64
    assert digest != "unknown"


def test_compute_missing_file_sha256(tmp_path: Path) -> None:
    missing = tmp_path / "nonexistent.file"
    assert compute_file_sha256(missing) == "unknown"


def test_create_and_save_artifact_manifest(tmp_path: Path) -> None:
    model_file = tmp_path / "model.joblib"
    model_file.write_bytes(b"dummy_model_bytes")

    data_file = tmp_path / "train.csv"
    data_file.write_text("Id,1stFlrSF,SalePrice\n1,1000,200000\n", encoding="utf-8")

    manifest = create_artifact_manifest(
        model_name="gradient_boosting",
        artifact_path=model_file,
        data_path=data_file,
        training_rows=1,
        metrics={"MAE": 15000.0, "RMSE": 20000.0, "R2": 0.85},
    )

    assert manifest.model_name == "gradient_boosting"
    assert manifest.training_rows == 1
    assert manifest.target_column == "SalePrice"
    assert "platform" in manifest.environment_details
    assert "scikit-learn" in manifest.environment_details

    out_json = tmp_path / "manifest.json"
    manifest.save(out_json)
    assert out_json.is_file()

    loaded = ArtifactManifest.load(out_json)
    assert loaded.model_name == manifest.model_name
    assert loaded.artifact_sha256 == manifest.artifact_sha256
    assert loaded.evaluation_metrics["MAE"] == 15000.0
