# Ames Housing Price Prediction

A reproducible machine-learning project using the Ames, Iowa housing dataset to explore and estimate residential sale prices.

## Project status

The current model is an educational baseline, not a production appraisal tool. The repository includes the original exploratory notebook and a local Python package/CLI for running its four-feature linear regression model. Broader data validation, fold-safe preprocessing, model comparisons, automated tests, and operational monitoring are planned work.

The notebook originally reported a mean absolute error (MAE) of **$24,810.15** for one 80/20 random split with `random_state=42`. Treat this as a historical baseline result, not a guarantee. It is a regression error in dollars, not “90% accuracy”; validation and uncertainty analysis are not yet sufficient to make a real-world performance claim.

## Current workflow

The baseline uses:

- `TotalSF`: first-floor area + second-floor area + total basement area
- `OverallQual`
- `YearBuilt`
- `GarageCars`

The model is scikit-learn `LinearRegression`. `Id` is retained only as an optional identifier in batch prediction output. The `SalePrice` column is the training target and is not used as a predictor.

## Requirements

- Python 3.10 or newer
- [uv](https://docs.astral.sh/uv/) for the locked, reproducible environment

Install uv using its [official installation instructions](https://docs.astral.sh/uv/getting-started/installation/). The project has a lock file (`uv.lock`); use `uv sync --locked --all-extras` to install the package, notebook libraries, and development tools without changing the lock file.

## Install

From the repository root:

```powershell
uv sync --locked --all-extras
```

Then run the commands from the repository root, where `train.csv` is located.

## Train and evaluate

Fit the baseline on all rows and save the model artifact under the ignored `artifacts/` directory:

```powershell
uv run --locked ames-housing train --data train.csv --model artifacts/ames_house_prices.joblib
```

Evaluate the existing baseline with the fixed random holdout:

```powershell
uv run --locked ames-housing evaluate --data train.csv --random-state 42 --test-size 0.2
```

The evaluation command reports holdout MAE. It does not tune or select a model.
Training and evaluation also accept `--no-fit-intercept` to disable the baseline model's intercept; the default is enabled.

## Batch prediction

Provide a CSV containing the raw feature columns used by the baseline (`1stFlrSF`, `2ndFlrSF`, `TotalBsmtSF`, `OverallQual`, `YearBuilt`, and `GarageCars`). `Id` is optional. Do not include `SalePrice` as an input feature.

```powershell
uv run --locked ames-housing predict --data new_houses.csv --model artifacts/ames_house_prices.joblib --output predictions.csv
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

The project owner reports that `train.csv` came from Kaggle, but the exact Kaggle listing, dataset version, attribution requirements, and redistribution terms have not been verified. The CSV remains in the repository for now; that is not confirmation that redistribution is permitted. Verify the source terms before publishing or redistributing the dataset. See the [project and data contract](docs/project-contract.md) for the input/output contract and initial evaluation gate.

## Next steps

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the roadmap covering reproducibility, data contracts, safer preprocessing, model validation, testing, and operations.
