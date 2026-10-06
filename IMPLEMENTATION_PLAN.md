# Industrialization Implementation Plan

## 1. Current project assessment

### What exists today

The repository is a small, single-purpose machine-learning analysis project:

- `HousePriceData.ipynb` contains the analysis, visualizations, feature creation, and one trained model.
- `train.csv` contains the Ames training dataset (1,460 rows, 81 columns, including `SalePrice`).
- `README.md` describes the project and reports a held-out mean absolute error (MAE) of `$24,810.15`.
- Git history is present and the working tree was clean during the review.

The notebook loads a CSV from a Google Colab/Drive-specific absolute path, explores correlations and missingness, fills selected missing values, creates `TotalSF`, trains a `LinearRegression` model on `TotalSF`, `OverallQual`, `YearBuilt`, and `GarageCars`, then reports MAE from one 80/20 random split (`random_state=42`).

### Work already completed

- Initial exploratory analysis of numeric relationships and missing values.
- A `TotalSF` feature combining first-floor, second-floor, and basement square footage.
- A hand-selected four-feature linear regression.
- A reproducible random split seed and a reported held-out MAE.
- A project overview, selected results, and example predictions in the README.

### Gaps blocking an industrial-quality handoff

1. **Not reproducible outside the original notebook environment.** The notebook hard-codes a Google Drive path; there is no executable package, CLI, dependency manifest/lock, or documented local run path.
2. **No automated verification.** There are no tests, lint/type checks, CI workflow, or data/schema validation.
3. **The reported score is under-specified.** It comes from one random split; there is no baseline comparison, cross-validation, uncertainty estimate, or documented model-selection protocol. “~90% accuracy” is not a defined regression metric.
4. **Preprocessing is not a fitted end-to-end pipeline.** Notebook mutations and model fitting are separate; future imputation, encoding, or feature selection must be fitted within each training fold to prevent leakage.
5. **Missingness handling is incomplete and not reusable.** Some columns with meaningful “feature absent” values are filled manually, but many other columns have missing values. This does not invalidate the current four complete predictors, but it is not a general input-preparation strategy.
6. **No stable prediction interface or artifact.** There is no defined input schema, serialized model, version metadata, batch prediction command, or service contract.
7. **No operational, governance, or data-provenance controls.** Data source/licensing, intended use, limitations, monitoring, retraining, rollback, and model ownership are undocumented.
8. **The README run instructions are incomplete.** The clone command's code block has no completed setup/run instructions, and some claims are stronger than the evidence shown in the notebook.
9. **Notebook quality issues.** It has a captured Seaborn deprecation warning and visible encoding/typo issues in prose. Notebook outputs should be reproducible from a clean environment.

## 2. Target outcome and scope assumptions

Build a maintainable, reproducible house-price modeling project that a second engineer can install, train, evaluate, and use for batch predictions without depending on Colab. Keep the first release deliberately small:

- **Required interface:** a local command-line workflow for training, evaluation, and batch prediction.
- **Optional deployment interface:** add an HTTP API only if a consuming application or user explicitly needs online predictions. Cloud hosting, authentication, or a UI are not prerequisites for the initial release.
- **Data:** retain the existing Ames `train.csv` as the training source, with provenance/licensing documented. Do not imply the model is calibrated for other cities, future market conditions, or lending/appraisal decisions without separate validation.
- **Model objective:** optimize and report clear regression metrics and subgroup behavior. Preserve the current MAE as a historical reference, not as a guaranteed target or production acceptance threshold.

## 3. Recommended repository structure

Introduce this structure incrementally; preserve the notebook as an educational/reporting surface rather than the only executable implementation.

```text
.
├── data/
│   └── raw/                    # existing source data, if redistribution is permitted
├── docs/
│   ├── model-card.md
│   └── data-dictionary.md
├── notebooks/
│   └── HousePriceData.ipynb
├── src/
│   └── ames_house_prices/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── features.py
│       ├── train.py
│       ├── evaluate.py
│       ├── predict.py
│       └── cli.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── artifacts/                  # local outputs; exclude generated artifacts from Git
├── pyproject.toml
├── lock file
├── .gitignore
└── README.md
```

Keep the module split smaller if the implementation remains cohesive. Make paths and file names configurable; do not bake machine-specific paths into code.

## 4. Phased implementation roadmap

### Phase 0 — Define the product contract and data rights

**Tasks**

- State the target user and supported workflow: offline training and batch prediction for Ames-like records.
- Define prediction-time inputs separately from training data: `SalePrice` is the label and must not be accepted as an inference feature; `Id` is an identifier, not a predictor.
- Confirm the dataset source, redistribution terms, attribution requirements, and whether the checked-in CSV may remain in the repository.
- Document supported population, currency/units, intended use, prohibited use, and known limitations.
- Agree that the initial delivery is a CLI/package, with an API/UI/cloud deployment deferred unless there is a concrete consumer.

**Exit criteria**

- Data provenance/licensing and intended-use notes are recorded.
- Input/output contract and scope are agreed, including which fields may be missing at prediction time.
- Evaluation metric definitions and initial release gate are documented before model selection.

### Phase 1 — Make the repository reproducible

**Tasks**

- Create the `src/` package and move executable logic out of notebook cells into importable functions.
- Add a `pyproject.toml` with supported Python versions, direct dependencies, dev tools, package metadata, and CLI entry point.
- Select a lock-file workflow and commit the resolved lock file for repeatable installs.
- Add `.gitignore` rules for virtual environments, caches, test/lint output, generated model artifacts, local data copies, and secrets.
- Use one configuration mechanism (CLI arguments plus a small typed config layer) for input paths, random seed, output directory, and model settings.
- Replace the Colab-only path with repository-relative or configured paths and update the notebook to call the package rather than duplicate production logic.
- Add setup, train, evaluate, and predict examples to the README.

**Exit criteria**

- A clean environment can install the project from documented commands.
- Training and evaluation run from a shell on Windows, macOS, and Linux without Colab/Drive.
- Notebook results can be regenerated locally from the same package code.

### Phase 2 — Validate data and make preprocessing safe

**Tasks**

- Implement an ingestion boundary that checks file readability, required columns, duplicate column names, expected target presence for training, non-empty rows, and acceptable data types.
- Validate identifiers and reject invalid target values (null, non-finite, or non-positive if the chosen business contract requires positivity).
- Report row/column counts, missingness by column, duplicate rows/IDs, category cardinalities, and target distribution as machine-readable and human-readable data audits.
- Separate missing values that mean “feature absent” (for example, no pool) from unknown/unrecorded values. Encode this distinction explicitly and consistently.
- Build preprocessing with scikit-learn `Pipeline`/`ColumnTransformer`: numeric imputation, categorical imputation, one-hot encoding with unknown-category handling, and optional scaling for models that need it.
- Keep all learned preprocessing inside the training pipeline so it is fit only on each training fold.
- Derive `TotalSF` in one tested transformation. Validate required input fields and ensure no target-derived values enter the feature matrix.
- Keep raw training data immutable; write cleaned/intermediate data only to generated, ignored output locations.

**Exit criteria**

- Invalid schemas fail with actionable errors instead of silently producing predictions.
- Data preparation is deterministic for a fixed input/configuration.
- Preprocessing can be fit, serialized, loaded, and applied to an unseen row with a previously unseen category.
- Tests prove the target and identifier are excluded from predictors and fold-specific preprocessing is used.

### Phase 3 — Establish a defensible model-selection baseline

**Tasks**

- Create a simple baseline (`DummyRegressor`) and compare it to the current four-feature linear regression.
- Evaluate a short, justified candidate set such as regularized linear models (Ridge/ElasticNet) and a tree-based regressor; avoid an unbounded model search.
- Use one untouched final holdout set and cross-validation on the remaining training data for model selection. Fix and record split seeds.
- Add a time-aware validation view using `YrSold` where sample sizes permit; compare it with random-split validation to expose market-time sensitivity.
- Report MAE as the primary interpretable metric plus RMSE and R². Report median/quantile absolute error and avoid calling regression performance “accuracy” without a defined formula.
- Save per-fold metrics, aggregate mean/spread, and the final holdout result. Do not tune against the final holdout.
- Compare the new pipeline to the notebook's MAE on a comparable split before replacing historical claims; note when changed splits/preprocessing make scores incomparable.

**Exit criteria**

- The selected model beats the baseline on the pre-agreed primary metric and is not materially worse on secondary metrics.
- The final holdout is evaluated after selection and its split and metrics are recorded.
- The README states the exact protocol, split, metric, and limitations; no undefined “~90% accuracy” claim remains.

### Phase 4 — Analyze errors and model limitations

**Tasks**

- Plot and tabulate actual-vs-predicted values, residual distribution, residuals vs. prediction, and calibration by price range.
- Measure performance slices by sale year, neighborhood, quality band, and price quantile when sample sizes are adequate; identify small-sample caveats.
- Investigate under/over-prediction for high-value homes and determine whether it is systematic, data-driven, or caused by the small feature set.
- Inspect high-leverage outliers; define a documented policy for retaining, excluding, or separately flagging observations. Do not drop rows solely because a chart looks unusual.
- Review features such as neighborhood for sensitive/proxy implications and restrict use to a clearly documented decision-support context.
- Add acceptance thresholds for overall and key-slice performance only after baseline results and intended use are agreed. Thresholds must be testable and justified.

**Exit criteria**

- Evaluation includes slice-level metrics and a concise error-analysis report.
- Any exclusion or special handling of training examples has documented, reproducible criteria.
- A model card records intended use, limitations, validation population, and residual risks.

### Phase 5 — Package model training and batch inference

**Tasks**

- Provide CLI commands such as `ames-housing train`, `ames-housing evaluate`, and `ames-housing predict`.
- Persist the complete preprocessing-plus-estimator pipeline, not only the final estimator.
- Store a versioned artifact manifest containing model version, training timestamp, source data fingerprint, feature list, Python/library versions, configuration, split protocol, and evaluation metrics.
- Implement prediction input validation and clear, row-specific error reporting; return predictions with a stable output schema and preserve a supplied record ID only as output metadata.
- Make training and prediction paths/configuration explicit; ensure no data or artifact is overwritten without a deliberate output path.
- Version model artifacts separately from source code. Avoid checking large generated model binaries into Git.

**Exit criteria**

- A model trained from a clean checkout can be loaded in a new process and score a batch of valid unseen records.
- Invalid and partially missing inputs produce documented behavior or explicit errors.
- The artifact manifest identifies the code, data, configuration, and environment used to produce the model.

### Phase 6 — Automated quality gates and CI

**Tasks**

- Add unit tests for schema checks, missing-value semantics, feature engineering, target/ID exclusion, and prediction output.
- Add integration tests for train-save-load-predict and CLI behavior using small synthetic fixtures.
- Add a data contract test against the checked-in dataset, including expected schema and documented row-count policy.
- Add deterministic tests for fixed seeds; avoid asserting one exact score when a range or invariant is more robust.
- Configure formatting, linting, static typing where practical, and test coverage reporting.
- Add CI to install the locked environment and run tests, lint, and type checks on pull requests.
- Add dependency vulnerability scanning and an update cadence; keep data/model tests separate from external network dependencies.

**Exit criteria**

- CI is green from a clean checkout and blocks merges on failed tests or static checks.
- The core train/predict path has unit and integration coverage.
- Test reports and dependency scan results are available for every main-branch change.

### Phase 7 — Release, monitoring, and controlled retraining

**Tasks**

- Release the package with a versioned changelog, reproducible installation instructions, and a tagged model/code release process.
- Define monitoring for input schema failures, missingness/category drift, feature/target distribution drift, latency/throughput if served, and realized MAE when later sale prices become available.
- Define retraining triggers, review/approval ownership, minimum validation checks, and rollback to the last accepted model.
- Retain model lineage and evaluation reports; do not automatically replace a deployed model solely because a scheduled retrain completed.
- If online inference is required, add a separate API phase with request/response schema, health/readiness endpoints, authentication, rate limits, logging that avoids sensitive payloads, container build, deployment, and operational alerting.

**Exit criteria**

- A release can be traced to its source, data fingerprint, environment, configuration, and evaluation report.
- An identified owner can determine when to retrain, approve a candidate, and roll back.
- For an API deployment, service-level objectives, security controls, monitoring, and rollback have been tested in the target environment.

## 5. Work sequencing and dependencies

1. Phase 0 decisions precede data redistribution and public release.
2. Phase 1 establishes the package/environment used by all subsequent phases.
3. Phase 2 data validation and preprocessing must be complete before model comparison in Phase 3.
4. Phase 3 model selection produces the candidate evaluated in Phase 4.
5. Phase 5 packages the selected pipeline; Phase 6 tests package behavior and CLI workflows.
6. Phase 7 release controls depend on artifact/version metadata from Phase 5 and automated checks from Phase 6.
7. The API portion of Phase 7 is optional and should be scheduled only after an online consumer is identified.

## 6. Definition of done

- [ ] Dataset source, license, intended use, and limitations are documented.
- [ ] A clean install and local train/evaluate/predict workflow are documented and verified.
- [ ] Notebook paths and outputs are reproducible from the package code.
- [ ] Ingestion validates schema and reports data quality issues clearly.
- [ ] Preprocessing is part of a single serializable pipeline and is learned only within training folds.
- [ ] Model selection uses a baseline, cross-validation, a final untouched holdout, and explicit regression metrics.
- [ ] Error and subgroup analysis are documented; justified thresholds are checked automatically where appropriate.
- [ ] Saved artifacts include sufficient lineage metadata and can be loaded in a fresh process.
- [ ] Unit/integration tests and CI quality gates pass.
- [ ] README claims match measured results and include complete setup/use instructions.
- [ ] A release, monitoring, retraining, and rollback process is documented.
- [ ] If an API is in scope, its contract, security, deployment, observability, and rollback are independently verified.

## 7. Immediate next actions

1. Confirm data provenance/redistribution rights and whether this is an offline portfolio project or a service for a real consumer.
2. Define the initial CLI scope and supported inference input contract.
3. Create the package scaffold, dependency lock, and local commands; remove the Colab-only data path.
4. Implement schema validation and the end-to-end preprocessing/model pipeline.
5. Re-run baseline and candidate comparisons with an untouched holdout and cross-validation; replace README metrics only with verified results.
6. Add tests/CI, complete documentation, and then decide whether online serving is actually required.