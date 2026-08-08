import xarray as xr
import geopandas as gpd
import regionmask

from po_valley_methane_forecasting.paths import find_project_root


def load_po_valley_roi() -> gpd.GeoDataFrame:
    """
    Load the provisional Po Valley ROI.
    """
    project_root = find_project_root()

    roi_path = (
        project_root
        / "data"
        / "regions"
        / "po_valley_provisional_envelope_roi.geojson"
    )

    if not roi_path.exists():
        raise FileNotFoundError(
            f"ROI file not found: {roi_path}"
        )

    roi = gpd.read_file(roi_path)

    if roi.crs is None:
        raise ValueError(
            "The ROI has no defined CRS."
        )

    return roi.to_crs("EPSG:4326")


def create_roi_mask(
    dataset: xr.Dataset,
    roi: gpd.GeoDataFrame,
) -> xr.DataArray:
    """
    Create a boolean spatial mask for cells inside the ROI.
    """
    ch4 = dataset["CH4"]

    roi = roi.to_crs("EPSG:4326")

    roi_mask = regionmask.mask_geopandas(
        roi,
        ch4["x"],
        ch4["y"],
    )

    return roi_mask.notnull()


def create_coverage_mask(
    dataset: xr.Dataset,
    coverage_threshold: float,
    start_date: str,
    end_date: str,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Compute temporal coverage over a reference period
    and return coverage values and the corresponding mask.
    """
    if not 0 <= coverage_threshold <= 1:
        raise ValueError(
            "coverage_threshold must be between 0 and 1."
        )

    ch4 = dataset["CH4"].sel(
        t=slice(start_date, end_date)
    )

    temporal_coverage = (
        ch4.notnull().mean(dim="t")
    )

    coverage_mask = (
        temporal_coverage >= coverage_threshold
    )

    return temporal_coverage, coverage_mask


def build_candidate_dataset(
    dataset: xr.Dataset,
    roi: gpd.GeoDataFrame,
    coverage_threshold: float,
    coverage_start: str,
    coverage_end: str,
) -> xr.Dataset:
    """
    Build the spatially filtered candidate CH4 dataset.
    """
    roi_mask = create_roi_mask(
        dataset,
        roi,
    )

    temporal_coverage, coverage_mask = (
        create_coverage_mask(
            dataset,
            coverage_threshold,
            coverage_start,
            coverage_end,
        )
    )

    candidate_mask = (
        roi_mask & coverage_mask
    )

    candidate_ch4 = (
        dataset["CH4"]
        .where(candidate_mask)
    )

    return xr.Dataset({
        "CH4": candidate_ch4,
        "roi_mask": roi_mask.astype("int8"),
        "candidate_mask": candidate_mask.astype("int8"),
        "training_temporal_coverage": temporal_coverage,
    })