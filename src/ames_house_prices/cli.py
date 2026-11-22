"""Command-line interface for the local baseline and pipeline workflow."""

import argparse
from pathlib import Path

import pandas as pd

from ames_house_prices.config import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    IDENTIFIER_COLUMN,
    RunConfig,
)
from ames_house_prices.data import load_training_data
from ames_house_prices.modeling import (
    evaluate_holdout,
    fit_model,
    load_model,
    predict,
    save_model,
)
from ames_house_prices.validation import generate_data_audit, validate_prediction_data


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ames-housing",
        description="Train, evaluate, audit, or run batch predictions with the Ames model pipeline.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Fit the model pipeline and save it.")
    train.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    train.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    train.add_argument(
        "--fit-intercept",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether LinearRegression should fit an intercept (default: enabled).",
    )

    evaluate = commands.add_parser("evaluate", help="Evaluate the pipeline on a holdout split.")
    evaluate.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    evaluate.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    evaluate.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    evaluate.add_argument(
        "--fit-intercept",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether LinearRegression should fit an intercept (default: enabled).",
    )

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

    if args.command == "train":
        config = RunConfig(
            data_path=args.data,
            model_path=args.model,
            fit_intercept=args.fit_intercept,
        )
        training_data = load_training_data(config.data_path)
        pipeline = fit_model(training_data, fit_intercept=config.fit_intercept)
        model_path = save_model(pipeline, config.model_path)
        print(f"Trained on {len(training_data)} rows; model pipeline saved to {model_path}")
        return

    if args.command == "evaluate":
        config = RunConfig(
            data_path=args.data,
            random_state=args.random_state,
            test_size=args.test_size,
            fit_intercept=args.fit_intercept,
        )
        training_data = load_training_data(config.data_path)
        result = evaluate_holdout(
            training_data,
            random_state=config.random_state,
            test_size=config.test_size,
            fit_intercept=config.fit_intercept,
        )
        print(f"Holdout MAE:  ${result.mae:,.2f}")
        print(f"Holdout RMSE: ${result.rmse:,.2f}")
        print(f"Holdout R²:   {result.r2:.4f}")
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
