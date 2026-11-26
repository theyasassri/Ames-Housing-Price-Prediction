import io
from fastapi.testclient import TestClient
import pytest

from ames_house_prices.api import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_api_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "package_version" in data


def test_api_metadata_endpoint(client: TestClient) -> None:
    response = client.get("/api/v1/metadata")
    assert response.status_code == 200


def test_api_predict_single_success(client: TestClient) -> None:
    payload = {
        "id": 42,
        "1stFlrSF": 1050.0,
        "2ndFlrSF": 650.0,
        "TotalBsmtSF": 900.0,
        "OverallQual": 7,
        "YearBuilt": 2002,
        "GarageCars": 2,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["id"] == 42
    assert res_data["predicted_price"] > 0
    assert "$" in res_data["formatted_price"]


def test_api_predict_single_validation_error(client: TestClient) -> None:
    payload = {
        "1stFlrSF": 1000.0,
        "2ndFlrSF": 0.0,
        "TotalBsmtSF": 1000.0,
        "OverallQual": 15,  # Out of bounds
        "YearBuilt": 2000,
        "GarageCars": 2,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422


def test_api_predict_batch_csv_success(client: TestClient) -> None:
    csv_content = (
        "Id,1stFlrSF,2ndFlrSF,TotalBsmtSF,OverallQual,YearBuilt,GarageCars\n"
        "101,1000,500,800,7,2001,2\n"
        "102,1200,0,1100,6,1985,2\n"
    )
    files = {"file": ("test.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    response = client.post("/api/v1/predict-batch", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["total_records"] == 2
    assert len(data["predictions"]) == 2
    assert data["predictions"][0]["id"] == 101
    assert data["predictions"][0]["predicted_price"] > 0


def test_api_predict_batch_csv_rejects_target_leakage(client: TestClient) -> None:
    csv_with_target = (
        "1stFlrSF,2ndFlrSF,TotalBsmtSF,OverallQual,YearBuilt,GarageCars,SalePrice\n"
        "1000,500,800,7,2001,2,250000\n"
    )
    files = {"file": ("leaked.csv", io.BytesIO(csv_with_target.encode("utf-8")), "text/csv")}
    response = client.post("/api/v1/predict-batch", files=files)
    assert response.status_code == 422
    assert "SalePrice" in response.json()["detail"]
