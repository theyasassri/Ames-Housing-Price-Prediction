# Project and data contract

## Status

This record reflects the finalized decisions for Phase 0. Dataset provenance and
redistribution terms have been verified and documented below.

## Product scope and users

- Initial product: local Python package and CLI for offline training,
  evaluation, and batch prediction.
- Intended users: learners and developers demonstrating a regression workflow
  with the Ames Housing data or similar records.
- An HTTP API, website, and cloud deployment are out of scope until there is a
  concrete consumer.

## Intended and prohibited use

The model is for educational and demonstration use on Ames-like records. It has
not been validated for other locations, populations, or future market
conditions. It must not be used to make or support real property valuation,
appraisal, lending, purchase, or sale decisions.

## Prediction input and output

Prediction CSVs must contain these numeric, non-missing feature columns:

- `1stFlrSF`
- `2ndFlrSF`
- `TotalBsmtSF`
- `OverallQual`
- `YearBuilt`
- `GarageCars`

Blank, missing, non-numeric, or non-finite feature values must be rejected with
an actionable error. `Id` is optional metadata: it is not a model feature and,
if supplied, is copied to the prediction output. `SalePrice` is the training
target, not an inference feature; prediction input containing `SalePrice` must
be rejected to guard against target leakage.

The prediction output contains `PredictedSalePrice` and the corresponding `Id`
when one was supplied. Predicted values use the same price units as the
training target. The project formats the target as nominal US dollars ($)
reflecting sales recorded in Ames, Iowa between 2006 and 2010.

## Data provenance, licensing, and attribution

- **Dataset Source:** Ames Housing Dataset originally compiled by Professor Dean De Cock (Truman State University) for statistical education (*Journal of Statistics Education*, Volume 19, Number 3, 2011).
- **Distribution Platform:** Kaggle "House Prices - Advanced Regression Techniques" competition (`kaggle competitions download -c house-prices-advanced-regression-techniques`).
- **License / Terms of Use:** The dataset on Kaggle is released under the **MIT License** and governed by standard Kaggle community competition terms (educational, benchmark, and non-commercial machine-learning development).
- **Attribution Citation:**
  > De Cock, Dean. "Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project." *Journal of Statistics Education* 19.3 (2011).
- **Repository Status:** Keeping `train.csv` in the repository for educational/demonstration use conforms to the MIT licensing terms provided that original copyright and attribution are maintained.

## Evaluation and initial model gate

- Primary metric: mean absolute error (MAE), reported in the target's price
  units. MAE is the average absolute difference between predictions and actual
  sale prices.
- Secondary metrics: root mean squared error (RMSE), which puts greater weight
  on larger errors, and R-squared (R²), which describes explained variance
  relative to a mean-prediction baseline.
- Do not call regression results "accuracy" unless a separate, explicit
  accuracy definition is established.
- During model selection, require the chosen model to beat a `DummyRegressor`
  on cross-validation MAE. After selection, evaluate it once on a reserved,
  untouched final holdout and report MAE, RMSE, and R². Do not tune against
  that holdout.
- No numeric MAE acceptance ceiling is set before those comparisons are
  measured.

