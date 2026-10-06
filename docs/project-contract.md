# Project and data contract

## Status

This record reflects the decisions made for Phase 0. Dataset provenance and
redistribution permission remain unverified; do not treat this document as a
license determination.

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
training target. The project has historically formatted the target as dollars,
but its exact price basis and source documentation have not yet been verified.

## Data provenance and redistribution

The project owner reports that `train.csv` came from Kaggle, but the exact
Kaggle listing, dataset version, attribution requirements, and redistribution
terms have not been confirmed. The file is currently kept in the repository
pending that verification; this is not a conclusion that redistribution is
permitted. Verify the source terms before publishing or redistributing the CSV
or a package that bundles it. Record the canonical source URL and required
attribution here once confirmed.

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

