import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sklearn.base import clone

from po_valley_methane_forecasting.validation import build_temporal_fold


def calculate_regression_metrics(
    y_true,
    y_pred,
) -> dict[str, float]:
    """
    Calculate regression metrics and prediction diagnostics.
    """
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)

    if y_true_array.shape != y_pred_array.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape."
        )

    if not np.isfinite(y_true_array).all():
        raise ValueError("y_true contains non-finite values.")

    if not np.isfinite(y_pred_array).all():
        raise ValueError("y_pred contains non-finite values.")

    errors = y_true_array - y_pred_array

    if (
        np.std(y_true_array) > 0
        and np.std(y_pred_array) > 0
    ):
        correlation = np.corrcoef(
            y_true_array,
            y_pred_array,
        )[0, 1]
    else:
        correlation = np.nan

    return {
        "mean_error": float(errors.mean()),
        "MAE": float(
            mean_absolute_error(
                y_true_array,
                y_pred_array,
            )
        ),
        "RMSE": float(
            np.sqrt(
                mean_squared_error(
                    y_true_array,
                    y_pred_array,
                )
            )
        ),
        "R2": float(
            r2_score(
                y_true_array,
                y_pred_array,
            )
        ),
        "target_mean": float(y_true_array.mean()),
        "prediction_mean": float(
            y_pred_array.mean()
        ),
        "target_std": float(y_true_array.std()),
        "prediction_std": float(
            y_pred_array.std()
        ),
        "correlation": float(correlation),
    }


def evaluate_persistence(
    evaluation_df: pd.DataFrame,
    target_column: str,
    persistence_column: str = "CH4",
) -> tuple[dict[str, float], np.ndarray]:
    """
    Evaluate a persistence baseline on a given dataset.

    The current value in ``persistence_column`` is used as
    the prediction of the future target.
    """
    required_columns = {
        target_column,
        persistence_column,
    }

    missing_columns = (
        required_columns
        - set(evaluation_df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if evaluation_df.empty:
        raise ValueError(
            "The evaluation DataFrame is empty."
        )

    y_true = evaluation_df[
        target_column
    ].to_numpy()

    predictions = evaluation_df[
        persistence_column
    ].to_numpy()

    metrics = calculate_regression_metrics(
        y_true,
        predictions,
    )

    return metrics, predictions


def evaluate_estimator(
    base_model,
    train_df: pd.DataFrame,
    evaluation_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> dict:
    """
    Fit and evaluate a scikit-learn regression estimator.
    """
    if train_df.empty:
        raise ValueError(
            "The training DataFrame is empty."
        )

    if evaluation_df.empty:
        raise ValueError(
            "The evaluation DataFrame is empty."
        )

    required_columns = set(
        feature_columns + [target_column]
    )

    missing_train_columns = (
        required_columns
        - set(train_df.columns)
    )

    missing_evaluation_columns = (
        required_columns
        - set(evaluation_df.columns)
    )

    if missing_train_columns:
        raise ValueError(
            "Missing training columns: "
            f"{sorted(missing_train_columns)}"
        )

    if missing_evaluation_columns:
        raise ValueError(
            "Missing evaluation columns: "
            f"{sorted(missing_evaluation_columns)}"
        )

    model = clone(base_model)

    X_train = train_df[feature_columns]
    y_train = train_df[target_column]

    X_evaluation = evaluation_df[
        feature_columns
    ]
    y_evaluation = evaluation_df[
        target_column
    ]

    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    evaluation_predictions = model.predict(
        X_evaluation
    )

    return {
        "model": model,
        "train_metrics": (
            calculate_regression_metrics(
                y_train,
                train_predictions,
            )
        ),
        "evaluation_metrics": (
            calculate_regression_metrics(
                y_evaluation,
                evaluation_predictions,
            )
        ),
        "train_predictions": train_predictions,
        "evaluation_predictions": (
            evaluation_predictions
        ),
    }


def evaluate_models_over_folds(
    dataframe: pd.DataFrame,
    validation_folds: list[dict],
    candidate_models: dict,
    feature_columns: list[str],
    target_column: str,
    persistence_column: str = "CH4",
) -> pd.DataFrame:
    """
    Evaluate persistence and scikit-learn models over temporal folds.

    Parameters
    ----------
    dataframe
        Complete model-ready dataset.
    validation_folds
        Fold specifications containing name, train_end_year,
        and validation_year.
    candidate_models
        Mapping from model names to unfitted scikit-learn estimators.
    feature_columns
        Columns used as model inputs.
    target_column
        Regression target column.
    persistence_column
        Current-value column used by the persistence baseline.

    Returns
    -------
    pd.DataFrame
        One row for each model and validation fold.
    """
    if dataframe.empty:
        raise ValueError("The input DataFrame is empty.")

    if not validation_folds:
        raise ValueError("No validation folds were provided.")

    if not candidate_models:
        raise ValueError("No candidate models were provided.")

    fold_results = []

    for fold in validation_folds:
        fold_name = fold["name"]

        train_df, validation_df = build_temporal_fold(
            dataframe=dataframe,
            train_end_year=fold["train_end_year"],
            validation_year=fold["validation_year"],
        )

        # Persistence baseline
        persistence_metrics, _ = evaluate_persistence(
            evaluation_df=validation_df,
            target_column=target_column,
            persistence_column=persistence_column,
        )

        fold_results.append({
            "fold": fold_name,
            "validation_year": fold["validation_year"],
            "model": "Persistence",
            "train_rows": len(train_df),
            "validation_rows": len(validation_df),
            **{
                f"validation_{metric_name}": value
                for metric_name, value
                in persistence_metrics.items()
            },
        })

        # Scikit-learn estimators
        for model_name, base_model in candidate_models.items():
            evaluation = evaluate_estimator(
                base_model=base_model,
                train_df=train_df,
                evaluation_df=validation_df,
                feature_columns=feature_columns,
                target_column=target_column,
            )

            train_metrics = evaluation["train_metrics"]
            validation_metrics = evaluation[
                "evaluation_metrics"
            ]

            fold_results.append({
                "fold": fold_name,
                "validation_year": fold["validation_year"],
                "model": model_name,
                "train_rows": len(train_df),
                "validation_rows": len(validation_df),
                **{
                    f"train_{metric_name}": value
                    for metric_name, value
                    in train_metrics.items()
                },
                **{
                    f"validation_{metric_name}": value
                    for metric_name, value
                    in validation_metrics.items()
                },
            })

    return pd.DataFrame(fold_results)


def comparison_with_baseline(
    fold_results_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add improvements over persistence to the fold results.
    """
    if fold_results_df.empty:
        raise ValueError(
            "The input DataFrame is empty."
        )

    required_columns = {
        "fold",
        "model",
        "validation_MAE",
        "validation_RMSE",
    }

    missing_columns = (
        required_columns
        - set(fold_results_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    persistence_by_fold = (
        fold_results_df.loc[
            fold_results_df["model"]
            == "Persistence",
            [
                "fold",
                "validation_MAE",
                "validation_RMSE",
            ],
        ]
        .rename(columns={
            "validation_MAE": "persistence_MAE",
            "validation_RMSE": "persistence_RMSE",
        })
    )

    if persistence_by_fold.empty:
        raise ValueError(
            "No persistence results were found."
        )

    if persistence_by_fold["fold"].duplicated().any():
        raise ValueError(
            "More than one persistence result "
            "was found for the same fold."
        )

    comparison_df = fold_results_df.merge(
        persistence_by_fold,
        on="fold",
        how="left",
        validate="many_to_one",
    )

    comparison_df[
        "MAE_improvement_over_persistence_pct"
    ] = (
        (
            comparison_df["persistence_MAE"]
            - comparison_df["validation_MAE"]
        )
        / comparison_df["persistence_MAE"]
        * 100
    )

    comparison_df[
        "RMSE_improvement_over_persistence_pct"
    ] = (
        (
            comparison_df["persistence_RMSE"]
            - comparison_df["validation_RMSE"]
        )
        / comparison_df["persistence_RMSE"]
        * 100
    )

    return (
        comparison_df
        .sort_values(["fold", "validation_MAE"])
        .reset_index(drop=True)
    )


def summarize_validation_results(
    comparison_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize comparison with persistence by taking mean values over temporal folds
    """

    if comparison_df.empty:
        raise ValueError("The input DataFrame is empty.")

    validation_summary_df = (
        comparison_df
        .groupby("model")
        .agg(
            mean_validation_MAE=(
                "validation_MAE",
                "mean",
            ),
            std_validation_MAE=(
                "validation_MAE",
                "std",
            ),
            worst_validation_MAE=(
                "validation_MAE",
                "max",
            ),
            mean_validation_RMSE=(
                "validation_RMSE",
                "mean",
            ),
            std_validation_RMSE=(
                "validation_RMSE",
                "std",
            ),
            mean_validation_R2=(
                "validation_R2",
                "mean",
            ),
            mean_MAE_improvement_pct=(
                "MAE_improvement_over_persistence_pct",
                "mean",
            ),
            mean_prediction_correlation=(
                "validation_correlation",
                "mean",
            ),
        )
        .sort_values("mean_validation_MAE")
        .reset_index()
    )

    return validation_summary_df