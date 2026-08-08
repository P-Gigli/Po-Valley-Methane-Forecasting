import pandas as pd


def build_temporal_fold(
    dataframe: pd.DataFrame,
    train_end_year: int,
    validation_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build an expanding-window temporal training-validation split.

    Training rows have target years up to and including
    ``train_end_year``. Validation rows belong to
    ``validation_year``.

    Parameters
    ----------
    dataframe
        Dataset containing ``target_year`` and ``target_week``.
    train_end_year
        Last target year included in training.
    validation_year
        Target year used for validation.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Training and validation DataFrames.
    """
    required_columns = {"target_year", "target_week"}
    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if validation_year <= train_end_year:
        raise ValueError(
            "validation_year must be greater than train_end_year."
        )

    train_df = dataframe.loc[
        dataframe["target_year"] <= train_end_year
    ].copy()

    validation_df = dataframe.loc[
        dataframe["target_year"] == validation_year
    ].copy()

    if train_df.empty:
        raise ValueError("The training set is empty.")

    if validation_df.empty:
        raise ValueError("The validation set is empty.")

    if (
        train_df["target_week"].max()
        >= validation_df["target_week"].min()
    ):
        raise ValueError(
            "Training timestamps must precede validation timestamps."
        )

    return train_df, validation_df


def build_final_train_test_split(
    dataframe: pd.DataFrame,
    test_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the final temporal training-test split.

    Training rows have target years earlier than ``test_year``.
    Test rows belong to ``test_year``.

    Parameters
    ----------
    dataframe
        Dataset containing ``target_year`` and ``target_week``.
    test_year
        Target year reserved for final testing.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Final training and test DataFrames.
    """
    required_columns = {"target_year", "target_week"}
    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    final_train_df = dataframe.loc[
        dataframe["target_year"] < test_year
    ].copy()

    test_df = dataframe.loc[
        dataframe["target_year"] == test_year
    ].copy()

    if final_train_df.empty:
        raise ValueError("The final training set is empty.")

    if test_df.empty:
        raise ValueError("The test set is empty.")

    if (
        final_train_df["target_week"].max()
        >= test_df["target_week"].min()
    ):
        raise ValueError(
            "Training timestamps must precede test timestamps."
        )

    return final_train_df, test_df