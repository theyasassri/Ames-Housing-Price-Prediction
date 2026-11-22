# Ames Housing Price Prediction

A reproducible machine-learning project using the Ames, Iowa housing dataset to explore and estimate residential sale prices.

## Project status

The project features a validated, reproducible machine learning pipeline and local CLI supporting multiple regression models. Models are benchmarked using 5-fold cross-validation and evaluated against an untouched holdout set as well as a temporal out-of-time split (`YrSold >= 2010`).

### Benchmarking Summary

Evaluated on 1,460 records (`random_state=42`, 80/20 train/holdout split, 5-fold CV):

| Model | 5-Fold CV MAE (Mean ± Std) | Holdout MAE | Holdout RMSE | Holdout $R^2$ | Temporal Holdout MAE (`YrSold >= 2010`) |
|---|---|---|---|---|---|
| **Gradient Boosting** (Selected) | **$20,860.93 ± $1,190.00** | **$20,366.74** | **$30,293.48** | **0.8804** | **$19,649.54** |
| Random Forest | $21,702.24 ± $1,186.09 | $20,222.57 | $30,454.03 | 0.8791 | $20,381.12 |
| ElasticNet | $24,560.70 ± $1,388.72 | $24,831.23 | $40,083.03 | 0.7905 | $23,710.45 |
| Ridge Regression | $24,649.71 ± $1,380.70 | $24,810.05 | $39,614.75 | 0.7954 | $23,684.11 |
| Linear Regression (Baseline) | $24,652.36 ± $1,380.45 | $24,810.15 | $39,606.85 | 0.7955 | $23,688.77 |
| Dummy Regressor (Mean Baseline) | $56,318.23 ± $2,612.15 | $62,575.93 | $87,619.03 | -0.0009 | $54,321.09 |

The baseline linear regression reproduces the historical held-out MAE of **$24,810.15**. The Gradient Boosting model substantially outperforms the baseline across all metrics, reducing MAE by over **$4,400** while increasing explained variance ($R^2$) from **0.7955** to **0.8804**.

## Current workflow

The feature pipeline automatically computes:
- `TotalSF`: `1stFlrSF` + `2ndFlrSF` + `TotalBsmtSF`
- `OverallQual`
- `YearBuilt`
- `GarageCars`

The pipeline is packaged into a scikit-learn `Pipeline` that avoids data leakage and performs runtime schema validation.

## Requirements

- Python 3.10 or newer
- [uv](https://docs.astral.sh/uv/) for dependency management

Install dependencies and development tools:

```powershell
uv sync --locked --all-extras
```

## CLI Commands

### 1. Benchmark all candidate models
```powershell
ames-housing benchmark --data train.csv
```

### 2. Evaluate a specific model architecture
```powershell
ames-housing evaluate --data train.csv --model-name gradient_boosting
```

### 3. Evaluate temporal stability (out-of-time split)
```powershell
ames-housing evaluate-temporal --data train.csv --model-name gradient_boosting --cutoff-year 2010
```

### 4. Train and save model pipeline
```powershell
ames-housing train --data train.csv --model-name gradient_boosting --model artifacts/gradient_boosting.joblib
```

### 5. Batch prediction
```powershell
ames-housing predict --data new_houses.csv --model artifacts/gradient_boosting.joblib --output predictions.csv
```

The output contains `PredictedSalePrice` and, if present in the input, the corresponding `Id`.

## Run the notebook locally

Install all extras with `uv sync --locked --all-extras`, open `HousePriceData.ipynb` from the repository root, and select the project environment created by uv. The notebook reads `train.csv` relative to the repository root and uses the package's shared data loader, feature builder, and holdout evaluation function.

## Repository contents

- `HousePriceData.ipynb` — exploratory data analysis and baseline model results.
- `src/ames_house_prices/` — reusable feature, data-loading, modeling, and CLI code.
- `train.csv` — Ames housing training data included with this project.
- `IMPLEMENTATION_PLAN.md` — phased roadmap for completing the project.

## Data and intended use

This is an educational/demo project for Ames-like records, not a property valuation product. It has not been validated for other locations, populations, or future market conditions and must not be used to make or support real property appraisal, lending, purchase, or sale decisions.

The dataset is the Ames Housing dataset compiled by Dean De Cock and distributed via Kaggle's "House Prices - Advanced Regression Techniques" under the MIT License. The current model is trained and evaluated only on this dataset and should not be assumed to generalize to other locations, time periods, or populations. Do not use it as the basis for real-world property valuation, lending, purchase, or sale decisions. See the [project and data contract](docs/project-contract.md) for full licensing, attribution, input/output contract, and initial evaluation gates.

## Next steps

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the roadmap covering reproducibility, data contracts, safer preprocessing, model validation, testing, and operations.
