"""Command-line interface for the local baseline and pipeline workflow."""

import argparse
from pathlib import Path

import pandas as pd

from ames_house_prices.config import (
    DEFAULT_CV_SPLITS,
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEMPORAL_CUTOFF_YEAR,
    DEFAULT_TEST_SIZE,
    IDENTIFIER_COLUMN,
    SUPPORTED_MODELS,
    RunConfig,
)
from ames_house_prices.data import load_training_data
from ames_house_prices.modeling import (
    benchmark_models,
    evaluate_holdout,
    evaluate_temporal_holdout,
    fit_model,
    load_model,
    predict,
    save_model,
)
from ames_house_prices.validation import generate_data_audit, validate_prediction_data


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ames-housing",
        description="Train, evaluate, benchmark, audit, or run batch predictions with Ames models.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Fit the model pipeline and save it.")
    train.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    train.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    train.add_argument(
        "--model-name",
        type=str,
        default="linear",
        choices=SUPPORTED_MODELS,
        help="Model architecture to train.",
    )
    train.add_argument(
        "--fit-intercept",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether linear/ridge models should fit an intercept (default: enabled).",
    )

    evaluate = commands.add_parser("evaluate", help="Evaluate a model on a random holdout split.")
    evaluate.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    evaluate.add_argument(
        "--model-name",
        type=str,
        default="linear",
        choices=SUPPORTED_MODELS,
        help="Model architecture to evaluate.",
    )
    evaluate.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    evaluate.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    evaluate.add_argument(
        "--fit-intercept",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether linear/ridge models should fit an intercept (default: enabled).",
    )

    temporal = commands.add_parser("evaluate-temporal", help="Evaluate a model on time-split (YrSold).")
    temporal.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    temporal.add_argument(
        "--model-name",
        type=str,
        default="linear",
        choices=SUPPORTED_MODELS,
        help="Model architecture to evaluate.",
    )
    temporal.add_argument(
        "--cutoff-year",
        type=int,
        default=DEFAULT_TEMPORAL_CUTOFF_YEAR,
        help="Test cutoff year (train on < cutoff, test on >= cutoff).",
    )
    temporal.add_argument(
        "--fit-intercept",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    benchmark_cmd = commands.add_parser("benchmark", help="Compare all candidate models with CV & holdout.")
    benchmark_cmd.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    benchmark_cmd.add_argument("--cv-splits", type=int, default=DEFAULT_CV_SPLITS)
    benchmark_cmd.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    benchmark_cmd.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)

    audit_cmd = commands.add_parser("audit", help="Audit dataset schema, missingness, and statistics.")
    audit_cmd.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)

    prediction = commands.add_parser("predict", help="Write predictions for an input CSV.")
    prediction.add_argument("--data", type=Path, required=True)
    prediction.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    prediction.add_argument("--output", type=Path, default=Path("predictions.csv"))
    return parser


def main() -> None:
    args = _parser().parse_args()

    if args.command == "audit":
        data = pd.read_csv(args.data)
        audit_res = generate_data_audit(data)
        print("Dataset Audit Summary:")
        print(f"  Rows: {audit_res['num_rows']}, Columns: {audit_res['num_columns']}")
        print(f"  Duplicate rows: {audit_res['duplicate_rows']}")
        if "duplicate_identifiers" in audit_res:
            print(f"  Duplicate IDs: {audit_res['duplicate_identifiers']}")
        if audit_res["missing_values_by_column"]:
            print(f"  Columns with missing values: {len(audit_res['missing_values_by_column'])}")
        else:
            print("  No missing values detected.")
        if "target_summary" in audit_res:
            ts = audit_res["target_summary"]
            print(f"  Target summary: min=${ts['min']:,.0f}, median=${ts['50%']:,.0f}, max=${ts['max']:,.0f}")
        return

    if args.command == "benchmark":
        data = load_training_data(args.data)
        print(f"Running {args.cv_splits}-Fold Cross-Validation & Reserved Holdout across candidate models...\n")
        benchmarks = benchmark_models(
            data,
            cv_splits=args.cv_splits,
            random_state=args.random_state,
            test_size=args.test_size,
        )

        header = f"{'Model':<20} | {'CV MAE (Mean ± Std)':<24} | {'Holdout MAE':<14} | {'Holdout RMSE':<14} | {'Holdout R²':<10}"
        print(header)
        print("-" * len(header))
        for b in benchmarks:
            cv_mae_str = f"${b.cv_mae_mean:,.2f} ± ${b.cv_mae_std:,.2f}"
            holdout_mae_str = f"${b.holdout_metrics.mae:,.2f}"
            holdout_rmse_str = f"${b.holdout_metrics.rmse:,.2f}"
            holdout_r2_str = f"{b.holdout_metrics.r2:.4f}"
            print(f"{b.model_name:<20} | {cv_mae_str:<24} | {holdout_mae_str:<14} | {holdout_rmse_str:<14} | {holdout_r2_str:<10}")
        return

    if args.command == "train":
        config = RunConfig(
            data_path=args.data,
            model_path=args.model,
            fit_intercept=args.fit_intercept,
            model_name=args.model_name,
        )
        training_data = load_training_data(config.data_path)
        pipeline = fit_model(
            training_data,
            model_name=config.model_name,
            fit_intercept=config.fit_intercept,
        )
        model_path = save_model(pipeline, config.model_path)
        print(f"Trained [{config.model_name}] on {len(training_data)} rows; model pipeline saved to {model_path}")
        return

    if args.command == "evaluate":
        config = RunConfig(
            data_path=args.data,
            random_state=args.random_state,
            test_size=args.test_size,
            fit_intercept=args.fit_intercept,
            model_name=args.model_name,
        )
        training_data = load_training_data(config.data_path)
        result = evaluate_holdout(
            training_data,
            model_name=config.model_name,
            random_state=config.random_state,
            test_size=config.test_size,
            fit_intercept=config.fit_intercept,
        )
        print(f"Model:        {config.model_name}")
        print(f"Holdout MAE:   ${result.mae:,.2f}")
        print(f"Holdout MedAE: ${result.medae:,.2f}")
        print(f"Holdout RMSE:  ${result.rmse:,.2f}")
        print(f"Holdout R²:    {result.r2:.4f}")
        return

    if args.command == "evaluate-temporal":
        config = RunConfig(
            data_path=args.data,
            model_name=args.model_name,
            temporal_cutoff_year=args.cutoff_year,
            fit_intercept=args.fit_intercept,
        )
        training_data = load_training_data(config.data_path)
        result = evaluate_temporal_holdout(
            training_data,
            model_name=config.model_name,
            cutoff_year=config.temporal_cutoff_year,
            fit_intercept=config.fit_intercept,
        )
        print(f"Temporal Evaluation for '{config.model_name}' (Cutoff YrSold >= {config.temporal_cutoff_year}):")
        print(f"  Holdout MAE:   ${result.mae:,.2f}")
        print(f"  Holdout MedAE: ${result.medae:,.2f}")
        print(f"  Holdout RMSE:  ${result.rmse:,.2f}")
        print(f"  Holdout R²:    {result.r2:.4f}")
        return

    config = RunConfig(
        data_path=args.data,
        model_path=args.model,
        output_path=args.output,
    )
    input_data = pd.read_csv(config.data_path)
    validate_prediction_data(input_data)
    model = load_model(config.model_path)
    predictions = predict(input_data, model)
    output = pd.DataFrame()
    if IDENTIFIER_COLUMN in input_data.columns:
        output[IDENTIFIER_COLUMN] = input_data[IDENTIFIER_COLUMN]
    output["PredictedSalePrice"] = predictions
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(config.output_path, index=False)
    print(f"Wrote {len(output)} predictions to {config.output_path}")


if __name__ == "__main__":
    main()
