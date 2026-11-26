"""FastAPI REST application for Ames Housing price prediction."""

import io
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from ames_house_prices import __version__ as pkg_version
from ames_house_prices.config import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_MODEL_PATH,
    IDENTIFIER_COLUMN,
)
from ames_house_prices.manifest import ArtifactManifest
from ames_house_prices.modeling import load_model, predict
from ames_house_prices.validation import DataValidationError, validate_prediction_data

app = FastAPI(
    title="Ames Housing Price Prediction API",
    description="REST API for offline and online batch property price estimation.",
    version=pkg_version,
)

# Enable CORS for local and web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cached global references
_MODEL: Any = None
_MANIFEST: ArtifactManifest | None = None


def get_model(model_path: Path = DEFAULT_MODEL_PATH) -> Any:
    """Retrieve or lazy-load the serialized model pipeline."""
    global _MODEL
    if _MODEL is None:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model artifact not found at {model_path}. Train the model first."
            )
        _MODEL = load_model(model_path)
    return _MODEL


def get_manifest(manifest_path: Path = DEFAULT_MANIFEST_PATH) -> ArtifactManifest | None:
    """Retrieve or lazy-load the artifact manifest."""
    global _MANIFEST
    if _MANIFEST is None and manifest_path.is_file():
        _MANIFEST = ArtifactManifest.load(manifest_path)
    return _MANIFEST


class PropertyInput(BaseModel):
    """Pydantic schema representing the validated physical property features.

    No arbitrary default biases are applied; realistic residential bounds are validated.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "id": 1,
                "1stFlrSF": 1050.0,
                "2ndFlrSF": 650.0,
                "TotalBsmtSF": 900.0,
                "OverallQual": 7,
                "YearBuilt": 2002,
                "GarageCars": 2,
            }
        },
    )

    id: int | None = Field(default=None, description="Optional property identifier")
    first_flr_sf: float = Field(
        ...,
        alias="1stFlrSF",
        ge=100.0,
        le=6000.0,
        description="First floor square footage (sq ft)",
    )
    second_flr_sf: float = Field(
        ...,
        alias="2ndFlrSF",
        ge=0.0,
        le=5000.0,
        description="Second floor square footage (sq ft)",
    )
    total_bsmt_sf: float = Field(
        ...,
        alias="TotalBsmtSF",
        ge=0.0,
        le=6000.0,
        description="Total basement area (sq ft)",
    )
    overall_qual: int = Field(
        ...,
        alias="OverallQual",
        ge=1,
        le=10,
        description="Overall material and finish quality rating (1 to 10)",
    )
    year_built: int = Field(
        ...,
        alias="YearBuilt",
        ge=1850,
        le=2030,
        description="Original construction year",
    )
    garage_cars: int = Field(
        ...,
        alias="GarageCars",
        ge=0,
        le=10,
        description="Size of garage in car capacity",
    )


class PredictionResponse(BaseModel):
    """Response containing valuation prediction and metadata."""

    id: int | None = None
    predicted_price: float
    formatted_price: str
    model_version: str


class BatchPredictionItem(BaseModel):
    id: int | None = None
    predicted_price: float


class BatchPredictionResponse(BaseModel):
    total_records: int
    predictions: list[BatchPredictionItem]


@app.get("/health", tags=["System"])
def health_check() -> dict[str, str]:
    """Check service readiness and model availability."""
    try:
        get_model()
        model_status = "loaded"
    except (FileNotFoundError, OSError, ValueError) as exc:
        model_status = f"unavailable: {exc}"
    return {
        "status": "ok",
        "package_version": pkg_version,
        "model_status": model_status,
    }


@app.get("/api/v1/metadata", tags=["Model Governance"])
def get_model_metadata() -> dict[str, Any]:
    """Retrieve lineage, dataset hashes, and evaluation metrics for the active model."""
    manifest = get_manifest()
    if manifest is None:
        return {
            "status": "warning",
            "message": "Manifest not found. Run 'ames-housing train' to generate full lineage.",
        }
    return manifest.to_dict()


@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_single(property_data: PropertyInput) -> PredictionResponse:
    """Predict the sale price for a single residential property."""
    model = get_model()

    row = {
        "1stFlrSF": property_data.first_flr_sf,
        "2ndFlrSF": property_data.second_flr_sf,
        "TotalBsmtSF": property_data.total_bsmt_sf,
        "OverallQual": property_data.overall_qual,
        "YearBuilt": property_data.year_built,
        "GarageCars": property_data.garage_cars,
    }
    input_df = pd.DataFrame([row])

    try:
        raw_pred = predict(input_df, model)
        val = float(raw_pred[0])
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Inference error: {exc}",
        ) from exc

    return PredictionResponse(
        id=property_data.id,
        predicted_price=round(val, 2),
        formatted_price=f"${val:,.2f}",
        model_version=pkg_version,
    )


@app.post("/api/v1/predict-batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_batch_csv(file: Annotated[UploadFile, File()]) -> BatchPredictionResponse:
    """Upload a CSV of properties to generate batch price predictions."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a CSV.",
        )

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not parse CSV: {exc}",
        ) from exc

    try:
        validate_prediction_data(df)
    except DataValidationError as err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(err),
        ) from err

    model = get_model()
    try:
        predictions = predict(df, model)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch scoring failure: {exc}",
        ) from exc

    items: list[BatchPredictionItem] = []
    has_id = IDENTIFIER_COLUMN in df.columns
    for idx, pred_val in enumerate(predictions):
        rec_id = int(df.loc[idx, IDENTIFIER_COLUMN]) if has_id else None
        items.append(BatchPredictionItem(id=rec_id, predicted_price=round(float(pred_val), 2)))

    return BatchPredictionResponse(
        total_records=len(items),
        predictions=items,
    )
