# Changelog

All notable changes to the Ames Housing Price Prediction package and workflow are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-11-27

### Added
- **Multi-Model Support:** Introduced `DummyRegressor`, `Ridge`, `ElasticNet`, `RandomForestRegressor`, and `GradientBoostingRegressor`.
- **Benchmarking Engine:** Added $K$-Fold cross-validation and time-aware evaluation (`YrSold`) subcommands in the CLI.
- **Runtime Schema Validation:** Added strict bounds checking, row-specific error indices, and target leakage prevention.
- **Residual & Error Diagnostics:** Added slice-level evaluation across price tiers, quality scores, and neighborhoods via `ames-housing analyze-errors`.
- **Artifact Manifests:** Automated generation and inspection of cryptographic artifact manifests tracking SHA-256 fingerprints, Git commits, training rows, and metrics.
- **Comprehensive Quality Gates:** Added 27 unit, integration, and dataset contract tests with multi-OS GitHub Actions CI workflow.
- **Governance Documentation:** Published `docs/model-card.md` and `docs/operations-runbook.md`.

### Changed
- Promoted **Gradient Boosting** as the primary production model, improving held-out MAE from $24,810.15 to $20,366.74 and $R^2$ from 0.7955 to 0.8804.
- Encapsulated feature transformations (`TotalSF`) and estimator inside a leak-free scikit-learn `Pipeline`.

## [0.1.0] - 2026-10-07

### Added
- Initial modular Python package under `src/ames_house_prices/`.
- Local CLI supporting `train`, `evaluate`, and `predict` for the 4-feature linear regression baseline.
- `pyproject.toml` and locked dependencies via `uv`.
- Reproducible random holdout split replicating historical notebook results ($MAE = \$24,810.15$).
