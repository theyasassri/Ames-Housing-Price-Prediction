# Model Card: Ames Housing Price Prediction Model

## 1. Model Details
- **Model Name:** Gradient Boosting Regressor Pipeline (`ames_house_prices`)
- **Version:** 0.2.0 (Phase 4 Candidate)
- **Model Type:** Ensemble Gradient Boosted Trees (`sklearn.ensemble.GradientBoostingRegressor`)
- **Trained By:** Ames Housing Price Prediction Project
- **Target Variable:** `SalePrice` (nominal US dollars, $).
- **Features Used:**
  - `TotalSF` (composite: `1stFlrSF` + `2ndFlrSF` + `TotalBsmtSF`)
  - `OverallQual` (material and finish quality, scale 1-10)
  - `YearBuilt` (original construction date)
  - `GarageCars` (size of garage in car capacity)

---

## 2. Intended Use & Limitations

### Intended Use
- **Educational and Demonstration Use Only:** Demonstrates leak-free data preprocessing, regression benchmarking, out-of-time evaluation, and automated error diagnostics.
- **Offline Batch Estimation:** Generates property price estimates on datasets with identical feature structure to the Ames housing records.

### Prohibited Use & High-Risk Applications
- **Appraisals & Mortgage Underwriting:** Must **NOT** be used to conduct property appraisals, establish mortgage or loan eligibility, determine insurance premiums, or settle real estate transactions.
- **Geographic Generalization:** The model is calibrated exclusively on Ames, Iowa residential properties sold between 2006 and 2010. It cannot be applied to other metropolitan or regional housing markets.
- **Inflation & Modern Market Conditions:** Does not account for post-2010 inflation, macroeconomic interest rate shifts, or regional zoning changes.

---

## 3. Quantitative Performance & Validation

### Cross-Validation & Reserved Holdout ($N=1,460$ total; holdout $N=292$, 20% random split)

| Evaluation Slice / Model | MAE | Median AE | RMSE | $R^2$ | Bias ($\text{Actual} - \text{Pred}$) |
|---|---|---|---|---|---|
| **Gradient Boosting (Overall Holdout)** | **$20,366.74** | **$12,871.33** | **$30,293.48** | **0.8804** | **+$2,114.28** |
| Linear Regression Baseline | $24,810.15 | $17,372.90 | $39,606.85 | 0.7955 | +$2,714.49 |
| Dummy Regressor Baseline | $62,575.93 | $45,821.92 | $87,619.03 | -0.0009 | $0.00 |
| **Temporal Holdout (`YrSold >= 2010`, $N=175$)** | **$19,649.54** | **$14,190.49** | **$28,708.54** | **0.8719** | +$1,180.20 |

---

## 4. Subgroup & Error Analysis (Slice Metrics)

Residual diagnostics reveal distinct behavior across price tiers on the holdout set ($N=292$):

| Price Bracket | Count | MAE | RMSE | Mean Bias ($\text{Actual} - \text{Pred}$) | Behavior Characterization |
|---|---|---|---|---|---|
| **Low (<$130,000)** | 83 | $15,552.18 | $21,662.53 | -$12,012.75 | Mild over-prediction for distressed/low-cost homes. |
| **Mid-Low ($130,000 - $200,000)** | 127 | $13,861.07 | $17,883.12 | -$1,360.44 | **Optimal performance**; unbiased center of distribution. |
| **Mid-High ($200,000 - $300,000)** | 58 | $30,606.19 | $41,920.52 | +$2,038.47 | Well-calibrated, slightly higher variance. |
| **High (>$300,000)** | 24 | $46,697.58 | $60,026.84 | +$37,431.45 | Systematic **under-prediction** of luxury properties. |

### High-Value Under-Prediction Finding
Properties over $300,000 (such as ID 692, actual $755,000 vs. predicted $632,180) are under-predicted by the model. 
- **Cause:** Luxury property values are driven by bespoke amenities not captured in the 4 core features (e.g., luxury finishes, pool/tennis courts, acreage, custom architectural designs).
- **Comparison:** While the linear regression had an catastrophic MAE of **$79,399.93** on homes >$300k, Gradient Boosting reduces that to **$46,697.58**, but luxury properties remain a known limitation.

### Outlier Handling Policy
1. High-leverage outliers (such as partial sales or extreme custom estates) are retained in training to avoid artificially inflating score metrics.
2. In production usage, records with `TotalSF > 4,000` sq ft or luxury features should be flagged for human appraisal review rather than automatic scoring.

---

## 5. Fair Housing & Proxy Attribute Considerations
- Geographic predictors like `Neighborhood` can act as socioeconomic or demographic proxies. In this release, `Neighborhood` is **excluded** from predictors and used exclusively for post-hoc slice evaluations and auditing.
