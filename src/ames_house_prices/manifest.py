"""Model artifact manifest, serialization, and lineage tracking."""

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ames_house_prices import __version__ as pkg_version
from ames_house_prices.config import (
    FEATURE_COLUMNS,
    RAW_FEATURE_COLUMNS,
    TARGET_COLUMN,
)


def compute_file_sha256(path: str | Path) -> str:
    """Compute the SHA-256 hex digest of a file."""
    filepath = Path(path)
    if not filepath.is_file():
        return "unknown"
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def get_git_commit_sha() -> str:
    """Return the current Git commit SHA or 'unknown'."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"
    return "unknown"


@dataclass(frozen=True)
class ArtifactManifest:
    """Metadata detailing model lineage, data fingerprint, configuration, and environment."""

    manifest_version: str
    package_version: str
    model_name: str
    artifact_filename: str
    artifact_sha256: str
    training_timestamp_utc: str
    git_commit_sha: str
    data_path: str
    data_sha256: str
    target_column: str
    raw_feature_columns: list[str]
    derived_feature_columns: list[str]
    training_rows: int
    evaluation_metrics: dict[str, float]
    python_version: str
    environment_details: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return out_path

    @classmethod
    def load(cls, path: str | Path) -> "ArtifactManifest":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


def create_artifact_manifest(
    model_name: str,
    artifact_path: str | Path,
    data_path: str | Path,
    training_rows: int,
    metrics: dict[str, float],
) -> ArtifactManifest:
    """Construct an ArtifactManifest for a trained model pipeline."""
    artifact_file = Path(artifact_path)
    data_file = Path(data_path)

    import joblib
    import numpy as np
    import pandas as pd
    import sklearn

    env_details = {
        "platform": platform.platform(),
        "joblib": joblib.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
    }

    return ArtifactManifest(
        manifest_version="1.0.0",
        package_version=pkg_version,
        model_name=model_name,
        artifact_filename=artifact_file.name,
        artifact_sha256=compute_file_sha256(artifact_file),
        training_timestamp_utc=datetime.now(timezone.utc).isoformat(),
        git_commit_sha=get_git_commit_sha(),
        data_path=str(data_file),
        data_sha256=compute_file_sha256(data_file),
        target_column=TARGET_COLUMN,
        raw_feature_columns=list(RAW_FEATURE_COLUMNS),
        derived_feature_columns=list(FEATURE_COLUMNS),
        training_rows=training_rows,
        evaluation_metrics=metrics,
        python_version=sys.version,
        environment_details=env_details,
    )
