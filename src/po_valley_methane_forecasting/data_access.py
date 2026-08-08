from pathlib import Path

import openeo
import pandas as pd
import xarray as xr

from po_valley_methane_forecasting.paths import find_project_root


OPENEO_URL = "openeo.dataspace.copernicus.eu"
COLLECTION_ID = "SENTINEL_5P_L2"
CH4_BAND = "CH4"


def connect_to_copernicus():
    """
    Connect and authenticate to the Copernicus Data Space
    openEO backend.
    """
    return (
        openeo
        .connect(OPENEO_URL)
        .authenticate_oidc()
    )


def build_ch4_cube(
    connection,
    spatial_extent: dict,
    temporal_extent: tuple[str, str],
) -> openeo.DataCube:
    """
    Build a Sentinel-5P XCH4 data cube.
    """
    return connection.load_collection(
        COLLECTION_ID,
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        bands=[CH4_BAND],
    )


def build_weekly_ch4_cube(
    connection,
    spatial_extent: dict,
    temporal_extent: tuple[str, str],
) -> openeo.DataCube:
    """
    Build a weekly mean Sentinel-5P XCH4 data cube.
    """
    cube = build_ch4_cube(
        connection=connection,
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
    )

    return cube.aggregate_temporal_period(
        period="week",
        reducer="mean",
    )


def download_weekly_ch4(
    connection,
    spatial_extent: dict,
    temporal_extent: tuple[str, str],
    output_path: Path,
    title: str,
):
    """
    Execute a weekly XCH4 batch job and save it as NetCDF.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_path.exists():
        print(
            f"File already available: "
            f"{output_path.name} — skipping download."
        )
        return None

    weekly_cube = build_weekly_ch4_cube(
        connection=connection,
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
    )

    job = weekly_cube.execute_batch(
        outputfile=output_path,
        out_format="NetCDF",
        title=title,
    )

    print(f"Saved to: {output_path}")

    return job


def get_annual_output_path(
    year: int,
) -> Path:
    """
    Return the default output path for one annual dataset.
    """
    project_root = find_project_root()

    return (
        project_root
        / "data"
        / "interim"
        / f"sentinel5p_ch4_po_valley_weekly_{year}.nc"
    )


def download_ch4_year(
    connection,
    year: int,
    spatial_extent: dict,
):
    """
    Download one year of weekly Sentinel-5P XCH4 data.
    """
    temporal_extent = (
        f"{year}-01-01",
        f"{year + 1}-01-01",
    )

    output_path = get_annual_output_path(year)

    return download_weekly_ch4(
        connection=connection,
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        output_path=output_path,
        title=(
            f"Sentinel-5P weekly XCH4 — "
            f"Po Valley {year}"
        ),
    )


def download_ch4_multiyear(
    connection,
    first_year: int,
    last_year: int,
    spatial_extent: dict,
    output_path: Path,
):
    """
    Download a continuous multiyear weekly Sentinel-5P
    XCH4 dataset.
    """
    if last_year < first_year:
        raise ValueError(
            "last_year must be greater than or equal "
            "to first_year."
        )

    temporal_extent = (
        f"{first_year}-01-01",
        f"{last_year + 1}-01-01",
    )

    return download_weekly_ch4(
        connection=connection,
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        output_path=Path(output_path),
        title=(
            f"Sentinel-5P weekly XCH4 — "
            f"Po Valley {first_year}–{last_year}"
        ),
    )


def load_xarray_dataset(
    dataset_path: Path,
) -> xr.Dataset:
    """
    Load a NetCDF dataset into memory.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    return xr.load_dataset(dataset_path)


def save_xarray_dataset(
    dataset: xr.Dataset,
    output_path: Path,
) -> None:
    """
    Save an xarray Dataset as NetCDF.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.to_netcdf(
        output_path,
        mode="w",
    )


def save_feature_table(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Save the model-ready feature table in Parquet format.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_parquet(
        output_path,
        index=False,
    )


def load_feature_table(
    input_path: Path,
) -> pd.DataFrame:
    """
    Load a model-ready Parquet feature table.
    """
    if not input_path.exists():
        raise FileNotFoundError(
            f"Feature table not found: {input_path}"
        )

    return pd.read_parquet(input_path)