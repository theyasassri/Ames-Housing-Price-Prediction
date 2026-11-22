"""Error analysis, residual diagnostics, slice-level metrics, and reporting."""

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from ames_house_prices.config import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    TARGET_COLUMN,
)
from ames_house_prices.modeling import build_model_pipeline, compute_metrics
from ames_house_prices.validation import validate_training_data


@dataclass(frozen=True)
class SliceMetric:
    """Metrics for a specific cohort or data slice."""

    slice_name: str
    slice_value: str
    sample_count: int
    mae: float
    rmse: float
    r2: float
    mean_residual: float  # Positive = under-predicting, Negative = over-predicting


@dataclass(frozen=True)
class ErrorAnalysisReport:
    """Structured findings from residual diagnostics and subgroup slice evaluations."""

    model_name: str
    total_samples: int
    overall_mae: float
    overall_rmse: float
    overall_r2: float
    median_absolute_error: float
    price_bracket_slices: list[SliceMetric]
    neighborhood_slices: list[SliceMetric]
    quality_slices: list[SliceMetric]
    year_sold_slices: list[SliceMetric]
    high_leverage_outliers: list[dict[str, Any]]

    def summary_table(self) -> str:
        """Format a human-readable summary of the error analysis."""
        lines = [
            f"=== Error Analysis Report for '{self.model_name}' ===",
            f"Holdout Size: {self.total_samples}",
            f"Overall MAE:  ${self.overall_mae:,.2f}",
            f"Overall RMSE: ${self.overall_rmse:,.2f}",
            f"Overall R²:   {self.overall_r2:.4f}",
            f"Median AE:    ${self.median_absolute_error:,.2f}",
            "",
            "--- Performance by Price Tier ---",
            f"{'Tier':<22} | {'Count':<6} | {'MAE':<12} | {'RMSE':<12} | {'Bias (Actual-Pred)':<18}",
            "-" * 78,
        ]
        for s in self.price_bracket_slices:
            bias_str = f"+${s.mean_residual:,.2f}" if s.mean_residual >= 0 else f"-${abs(s.mean_residual):,.2f}"
            lines.append(
                f"{s.slice_value:<22} | {s.sample_count:<6} | ${s.mae:<11,.2f} | ${s.rmse:<11,.2f} | {bias_str:<18}"
            )

        lines.extend([
            "",
            "--- Top Outliers (Highest Absolute Residuals) ---",
            f"{'Index/ID':<10} | {'Actual ($)':<12} | {'Predicted ($)':<14} | {'Error ($)':<14}",
            "-" * 56,
        ])
        for out in self.high_leverage_outliers[:5]:
            idx_id = str(out.get("Id", out.get("index", "N/A")))
            lines.append(
                f"{idx_id:<10} | ${out['actual']:<11,.2f} | ${out['predicted']:<13,.2f} | ${out['abs_error']:<13,.2f}"
            )
        return "\n".join(lines)


def _compute_slice_metrics(
    df: pd.DataFrame,
    slice_col: str,
    min_samples: int = 5,
) -> list[SliceMetric]:
    """Calculate regression and bias metrics grouped by a specific column."""
    slices: list[SliceMetric] = []
    for val, group in df.groupby(slice_col, observed=True):
        count = len(group)
        if count < min_samples:
            continue
        m = compute_metrics(group["actual"], group["predicted"])
        mean_res = float((group["actual"] - group["predicted"]).mean())
        slices.append(
            SliceMetric(
                slice_name=slice_col,
                slice_value=str(val),
                sample_count=count,
                mae=m.mae,
                rmse=m.rmse,
                r2=m.r2,
                mean_residual=mean_res,
            )
        )
    return slices


def run_error_analysis(
    frame: pd.DataFrame,
    *,
    model_name: str = "gradient_boosting",
    random_state: int = DEFAULT_RANDOM_STATE,
    test_size: float = DEFAULT_TEST_SIZE,
    fit_intercept: bool = True,
) -> ErrorAnalysisReport:
    """Perform holdout evaluation with deep residual and slice-level diagnostics."""
    validate_training_data(frame)

    y = frame[TARGET_COLUMN]
    X = frame.drop(columns=[TARGET_COLUMN])

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    pipeline = build_model_pipeline(
        model_name=model_name,
        fit_intercept=fit_intercept,
        random_state=random_state,
    )
    pipeline.fit(x_train, y_train)
    preds = pipeline.predict(x_test)

    eval_df = x_test.copy()
    eval_df["actual"] = y_test
    eval_df["predicted"] = preds
    eval_df["residual"] = eval_df["actual"] - eval_df["predicted"]
    eval_df["abs_error"] = eval_df["residual"].abs()

    overall_metrics = compute_metrics(y_test, preds)

    # 1. Price Bracket Slices
    # Group into quartiles/standard price bands
    bins = [0, 130000, 200000, 300000, float("inf")]
    labels = ["Low (<$130k)", "Mid-Low ($130k-$200k)", "Mid-High ($200k-$300k)", "High (>$300k)"]
    eval_df["price_tier"] = pd.cut(eval_df["actual"], bins=bins, labels=labels)
    price_slices = _compute_slice_metrics(eval_df, "price_tier", min_samples=1)

    # 2. Neighborhood Slices (if present in frame)
    neighborhood_slices: list[SliceMetric] = []
    if "Neighborhood" in eval_df.columns:
        neighborhood_slices = _compute_slice_metrics(eval_df, "Neighborhood", min_samples=5)
        neighborhood_slices.sort(key=lambda s: s.mae, reverse=True)

    # 3. OverallQual Slices
    quality_slices: list[SliceMetric] = []
    if "OverallQual" in eval_df.columns:
        quality_slices = _compute_slice_metrics(eval_df, "OverallQual", min_samples=5)
        quality_slices.sort(key=lambda s: s.slice_value)

    # 4. YrSold Slices
    year_slices: list[SliceMetric] = []
    if "YrSold" in eval_df.columns:
        year_slices = _compute_slice_metrics(eval_df, "YrSold", min_samples=5)
        year_slices.sort(key=lambda s: s.slice_value)

    # 5. Outliers (top 10 absolute errors)
    top_outliers = (
        eval_df.sort_values(by="abs_error", ascending=False)
        .head(10)[["actual", "predicted", "residual", "abs_error"] + ([col for col in ["Id", "OverallQual", "TotalBsmtSF", "GrLivArea"] if col in eval_df.columns])]
        .to_dict(orient="records")
    )

    return ErrorAnalysisReport(
        model_name=model_name,
        total_samples=len(eval_df),
        overall_mae=overall_metrics.mae,
        overall_rmse=overall_metrics.rmse,
        overall_r2=overall_metrics.r2,
        median_absolute_error=overall_metrics.medae,
        price_bracket_slices=price_slices,
        neighborhood_slices=neighborhood_slices,
        quality_slices=quality_slices,
        year_sold_slices=year_slices,
        high_leverage_outliers=top_outliers,
    )
