import pandas as pd
import xarray as xr
import numpy as np


def construct_df(
    candidate_ds: xr.Dataset,
) -> pd.DataFrame:
    """
    Convert the candidate xarray dataset into a tabular CH4 dataset.
    """
    required_variables = {
        "CH4",
        "candidate_mask",
    }

    missing_variables = (
        required_variables
        - set(candidate_ds.data_vars)
    )

    if missing_variables:
        raise ValueError(
            "Missing required variables: "
            f"{sorted(missing_variables)}"
        )

    candidate_mask = (
        candidate_ds["candidate_mask"]
        .astype(bool)
    )

    ch4_stacked = (
        candidate_ds["CH4"]
        .where(candidate_mask)
        .stack(cell=("y", "x"))
        .dropna(dim="cell", how="all")
    )

    ch4_df = (
        ch4_stacked
        .to_series()
        .rename("CH4")
        .reset_index()
        .rename(columns={"t": "week"})
        .sort_values(["y", "x", "week"])
        .reset_index(drop=True)
    )

    ch4_df["cell_id"] = (
        ch4_df
        .groupby(["y", "x"], sort=True)
        .ngroup()
    )

    return (
        ch4_df[
            ["week", "cell_id", "y", "x", "CH4"]
        ]
        .sort_values(["cell_id", "week"])
        .reset_index(drop=True)
    )


def add_target_columns(
    ch4_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add the next-week CH4 target and its timestamp.
    """
    target_df = (
        ch4_df
        .sort_values(["cell_id", "week"])
        .reset_index(drop=True)
        .copy()
    )

    cell_groups = target_df.groupby(
        "cell_id",
        sort=False,
    )

    target_df["target_next_week"] = (
        cell_groups["CH4"].shift(-1)
    )

    target_df["target_week"] = (
        cell_groups["week"].shift(-1)
    )

    target_df["forecast_horizon_days"] = (
        target_df["target_week"]
        - target_df["week"]
    ).dt.days

    return target_df


def create_analytical_features(
    ch4_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add lagged values and recent weekly change.
    """
    feature_df = (
        ch4_df
        .sort_values(["cell_id", "week"])
        .reset_index(drop=True)
        .copy()
    )

    grouped_ch4 = (
        feature_df
        .groupby("cell_id", sort=False)["CH4"]
    )

    feature_df["ch4_prev_1w"] = (
        grouped_ch4.shift(1)
    )

    feature_df["ch4_prev_2w"] = (
        grouped_ch4.shift(2)
    )

    feature_df["ch4_change_1w"] = (
        feature_df["CH4"]
        - feature_df["ch4_prev_1w"]
    )

    return feature_df


def create_statistical_features(
        feature_df:pd.DataFrame
) -> pd.DataFrame:
    """
    Add weekly mean values and standard deviations over the last weeks as features
    """

    feature_df_1=feature_df.copy()

    grouped_ch4 = feature_df.groupby("cell_id")["CH4"]

    feature_df_1["ch4_mean_last_3w"] = (
        grouped_ch4.transform(
            lambda series: series.rolling(
                window=3,
                min_periods=3,
            ).mean()
        )
    )

    feature_df_1["ch4_std_last_3w"] = (
        grouped_ch4.transform(
            lambda series: series.rolling(
                window=3,
                min_periods=3,
            ).std()
        )
    )

    return feature_df_1


def encode_time_of_the_year(
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Encode the seasonal position using cyclical features.
    """
    encoded_df = feature_df.copy()

    week_of_year = (
        encoded_df["week"]
        .dt
        .isocalendar()
        .week
        .astype(int)
    )

    encoded_df["season_sin"] = np.sin(
        2 * np.pi * week_of_year / 52
    )

    encoded_df["season_cos"] = np.cos(
        2 * np.pi * week_of_year / 52
    )

    return encoded_df


def build_model_ready_dataframe(
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep rows with complete features and an exact one-week target.
    """
    feature_columns = [
        "CH4",
        "ch4_prev_1w",
        "ch4_prev_2w",
        "ch4_change_1w",
        "ch4_mean_last_3w",
        "ch4_std_last_3w",
        "x",
        "y",
        "season_sin",
        "season_cos",
    ]

    required_columns = set(
        feature_columns
        + [
            "target_next_week",
            "forecast_horizon_days",
        ]
    )

    missing_columns = (
        required_columns
        - set(feature_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return (
        feature_df.loc[
            feature_df["target_next_week"].notna()
            & feature_df[
                "forecast_horizon_days"
            ].eq(7)
            & feature_df[
                feature_columns
            ].notna().all(axis=1)
        ]
        .copy()
        .reset_index(drop=True)
    )





