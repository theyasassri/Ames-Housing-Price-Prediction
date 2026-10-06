"""Command-line interface for the local baseline workflow."""

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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ames-housing",
        description="Train, evaluate, or run batch predictions with the Ames baseline model.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Fit the baseline model and save it.")
    train.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    train.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    train.add_argument(
        "--fit-intercept",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether LinearRegression should fit an intercept (default: enabled).",
    )

    evaluate = commands.add_parser("evaluate", help="Evaluate the baseline on a holdout split.")
    evaluate.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    evaluate.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    evaluate.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    evaluate.add_argument(
        "--fit-intercept",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether LinearRegression should fit an intercept (default: enabled).",
    )

    prediction = commands.add_parser("predict", help="Write predictions for an input CSV.")
    prediction.add_argument("--data", type=Path, required=True)
    prediction.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    prediction.add_argument("--output", type=Path, default=Path("predictions.csv"))
    return parser


def main() -> None:
    args = _parser().parse_args()

    if args.command == "train":
        config = RunConfig(
            data_path=args.data,
            model_path=args.model,
            fit_intercept=args.fit_intercept,
        )
        training_data = load_training_data(config.data_path)
        model = fit_model(training_data, fit_intercept=config.fit_intercept)
        model_path = save_model(model, config.model_path)
        print(f"Trained on {len(training_data)} rows; model saved to {model_path}")
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
        print(f"Holdout MAE: ${result.mae:,.2f}")
        return

    config = RunConfig(
        data_path=args.data,
        model_path=args.model,
        output_path=args.output,
    )
    input_data = pd.read_csv(config.data_path)
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
