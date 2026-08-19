"""
predict_rf_uncertainty.py

Generates an RF uncertainty map based on the standard deviation of
individual tree probabilities.

Uncertainty is computed by:
  1. Re-predicting on the full extent grid
  2. Capturing predictions from each tree in the ensemble
  3. Computing per-pixel standard deviation
"""

import sys
from pathlib import Path

# Add pipeline directory to path so we can import config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import geopandas as gpd
from shapely.geometry import box

from config import (
    RF_MODEL_PATH,
    RF_PROSPECTIVITY_TIFF,
    OUTPUTS_DIR,
    BMC_BBOX_WGS84,
    CRS_SOURCE,
    CRS_TARGET,
    TARGET_RESOLUTION_M,
    PROCESSED_DIR,
    MASTER_RASTER_PATH,
)

import pandas as pd

UNCERTAINTY_TIF = OUTPUTS_DIR / "rf_uncertainty_map.tif"
RASTERS_DIR = PROCESSED_DIR / "rasters_reprojected"
DATASET_DIR = PROCESSED_DIR / "training_dataset"


def build_prediction_grid():
    """Create a prediction grid matching the master BMC raster exactly."""
    if not MASTER_RASTER_PATH.exists():
        raise FileNotFoundError(
            f"Master raster not found: {MASTER_RASTER_PATH}. "
            "Run pipeline/02_preprocessing/reproject_grids.py first."
        )

    with rasterio.open(MASTER_RASTER_PATH) as master:
        if master.crs != CRS_TARGET:
            raise ValueError(f"Master raster CRS must be {CRS_TARGET}, got {master.crs}")
        minx, miny, maxx, maxy = master.bounds
        width, height = master.width, master.height

    xs = minx + (np.arange(width) + 0.5) * TARGET_RESOLUTION_M
    ys = maxy - (np.arange(height) + 0.5) * TARGET_RESOLUTION_M
    xx, yy = np.meshgrid(xs, ys)

    coords = np.column_stack([xx.ravel(), yy.ravel()])
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    print(f"[Grid] {height} rows x {width} cols = {height*width:,} cells")
    return coords, transform, (height, width)


def sample_rasters(transform, shape):
    """Sample reprojected rasters at grid points to build feature matrix."""
    from rasterio.warp import reproject, Resampling
    import pandas as pd
    
    raster_paths = sorted(RASTERS_DIR.glob("*.tif"))
    if not raster_paths:
        raise FileNotFoundError(f"No rasters found in {RASTERS_DIR}")

    results = {}
    height, width = shape
    print(f"[Sampling] {len(raster_paths)} rasters at {height*width:,} points ...")

    for rp in raster_paths:
        col = rp.stem.replace("_epsg2953", "")
        dest = np.empty((height, width), dtype=np.float32)
        with rasterio.open(rp) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=dest,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=CRS_TARGET,
                resampling=Resampling.nearest,
                dst_nodata=np.nan
            )
        results[col] = dest.ravel()

    return pd.DataFrame(results)


def main():

    rf = joblib.load(RF_MODEL_PATH)
    print(f"[Model] Loaded RF with {len(rf.estimators_)} trees")

    # Build prediction grid
    coords, transform, shape = build_prediction_grid()

    # Sample rasters to build feature matrix
    grid_df = sample_rasters(transform, shape)
    print(f"[Features] Extracted {grid_df.shape[1]} features from {grid_df.shape[0]:,} cells")

    # Load feature names used during training
    feature_names = pd.read_csv(
        DATASET_DIR / "feature_names.csv",
        header=None
    ).iloc[:, 0].tolist()
    print(f"[Features] Using {len(feature_names)} trained features")

    # Ensure all features used by the model are present
    for col in feature_names:
        if col not in grid_df.columns:
            print(f"  Warning: Missing feature {col}, filling with NaN")
            grid_df[col] = np.nan

    X_grid = grid_df[feature_names].values.astype(np.float32)

    # Compute predictions from each tree
    print(f"[Uncertainty] Computing predictions from {len(rf.estimators_)} trees ...")
    tree_probs = np.zeros(
        (len(rf.estimators_), X_grid.shape[0]),
        dtype=np.float32
    )

    for i, tree in enumerate(rf.estimators_):
        tree_probs[i] = tree.predict_proba(X_grid)[:, 1]
        if (i + 1) % 10 == 0:
            print(f"  {i + 1} / {len(rf.estimators_)} trees processed")

    # Compute standard deviation across trees (uncertainty)
    uncertainty = np.std(tree_probs, axis=0)
    print(f"[Uncertainty] Range: {np.nanmin(uncertainty):.4f} - {np.nanmax(uncertainty):.4f}")

    # Load profile from existing prospectivity map
    with rasterio.open(RF_PROSPECTIVITY_TIFF) as src:
        profile = src.profile
        probs = src.read(1)

    uncertainty_grid = uncertainty.reshape(probs.shape)

    profile.update(
        dtype="float32",
        compress="lzw"
    )

    with rasterio.open(
        UNCERTAINTY_TIF,
        "w",
        **profile
    ) as dst:
        dst.write(
            uncertainty_grid.astype(np.float32),
            1
        )

    print(
        f"Saved: {UNCERTAINTY_TIF}"
    )


if __name__ == "__main__":
    main()