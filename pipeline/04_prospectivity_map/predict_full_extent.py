"""
predict_full_extent.py
───────────────────────
Runs the trained model across the full ~3,800 km² Bathurst Mining Camp
extent to generate a continuous prospectivity map.

Process:
  1. Load best model (RF or XGBoost, whichever has higher CV AUC)
  2. Create a 100m resolution prediction grid over the BMC extent
  3. Sample all geophysical rasters at each grid cell centroid
  4. Apply engineered features (same as training)
  5. Predict VMS probability at each grid cell
  6. Write output as GeoTIFF prospectivity map

The output GeoTIFF is the primary scientific deliverable — it can be
opened in QGIS, ArcGIS, or any GIS platform for drill target generation.

Usage:
    python pipeline/04_prospectivity_map/predict_full_extent.py
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import geopandas as gpd
from shapely.geometry import box

PIPELINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_DIR))
from config import (
    PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR,
    RF_MODEL_PATH, XGB_MODEL_PATH,
    BMC_BBOX_WGS84, CRS_SOURCE, CRS_TARGET,
    TARGET_RESOLUTION_M, NODATA_VALUE,
    PROSPECTIVITY_TIFF
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

DATASET_DIR     = PROCESSED_DIR / "training_dataset"
RASTERS_DIR     = PROCESSED_DIR / "rasters_reprojected"


def select_best_model() -> tuple:
    """Load both models and select the one with higher CV AUC."""
    rf_metrics  = pd.read_csv(MODELS_DIR / "rf_cv_metrics.csv")
    xgb_metrics = pd.read_csv(MODELS_DIR / "xgb_cv_metrics.csv")

    rf_auc  = rf_metrics["roc_auc_mean"].values[0]
    xgb_auc = xgb_metrics["roc_auc_mean"].values[0]

    if rf_auc >= xgb_auc:
        log.info(f"  Selected: Random Forest (AUC={rf_auc:.4f} vs XGB={xgb_auc:.4f})")
        return joblib.load(RF_MODEL_PATH), "RandomForest"
    else:
        log.info(f"  Selected: XGBoost (AUC={xgb_auc:.4f} vs RF={rf_auc:.4f})")
        return joblib.load(XGB_MODEL_PATH), "XGBoost"


def build_prediction_grid() -> tuple[np.ndarray, rasterio.transform.Affine, tuple]:
    """
    Create a regular grid of points covering the BMC extent in EPSG:2953.
    Returns grid coordinates (N, 2), affine transform, and (height, width).
    """
    # Reproject BMC bbox from WGS84 → EPSG:2953
    lon_min, lat_min, lon_max, lat_max = BMC_BBOX_WGS84
    bbox_gdf = gpd.GeoDataFrame(
        geometry=[box(lon_min, lat_min, lon_max, lat_max)],
        crs=CRS_SOURCE
    ).to_crs(CRS_TARGET)
    minx, miny, maxx, maxy = bbox_gdf.total_bounds

    # Build grid at 100m resolution
    xs = np.arange(minx, maxx, TARGET_RESOLUTION_M)
    ys = np.arange(maxy, miny, -TARGET_RESOLUTION_M)  # Top-down
    xx, yy = np.meshgrid(xs, ys)
    height, width = xx.shape

    coords = np.column_stack([xx.ravel(), yy.ravel()])
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    log.info(f"  Grid: {height} rows × {width} cols = {height*width:,} cells")
    log.info(f"  Extent: ({minx:.0f}, {miny:.0f}) → ({maxx:.0f}, {maxy:.0f}) [EPSG:2953]")
    return coords, transform, (height, width)


def sample_all_rasters(coords: np.ndarray) -> pd.DataFrame:
    """Sample all available rasters at grid point coordinates."""
    from rasterio.sample import sample_gen
    raster_paths = sorted(RASTERS_DIR.glob("*.tif"))

    if not raster_paths:
        log.warning(
            "  No rasters found in reprojected dir. "
            "Prediction will use NaN-filled columns for raster features."
        )
        return pd.DataFrame()

    results = {}
    log.info(f"  Sampling {len(raster_paths)} raster(s) at {len(coords):,} grid points ...")

    for rp in raster_paths:
        col = rp.stem.replace("_epsg2953", "")
        with rasterio.open(rp) as src:
            vals = [
                float(v[0]) if v[0] != src.nodata else np.nan
                for v in sample_gen(src, coords)
            ]
        results[col] = vals
        valid = sum(1 for v in vals if not np.isnan(float(v) if v is not None else np.nan))
        log.info(f"    {col:30s}: {valid:,} / {len(vals):,} valid")

    return pd.DataFrame(results)


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply same derived features as engineer_features.py (inline)."""
    EPSILON = 1e-6
    tmi_col = "mag_tmi_nb_2013"
    fvd_col = "mag_fvd_nb_2013"

    if tmi_col in df.columns and fvd_col in df.columns:
        tmi = df[tmi_col].fillna(0).values
        fvd = df[fvd_col].fillna(0).values
        df["mag_hgm"] = np.sqrt(np.gradient(tmi)**2 + fvd**2 + EPSILON)
        df["mag_as"]  = np.sqrt(2 * np.gradient(tmi)**2 + fvd**2 + EPSILON)
    else:
        df["mag_hgm"] = np.nan
        df["mag_as"]  = np.nan

    for col in ["rad_k", "rad_th", "rad_u"]:
        if col not in df.columns:
            df[col] = np.nan

    k  = df.get("rad_k",  pd.Series(EPSILON, index=df.index)).fillna(EPSILON).values
    th = df.get("rad_th", pd.Series(EPSILON, index=df.index)).fillna(EPSILON).values
    u  = df.get("rad_u",  pd.Series(EPSILON, index=df.index)).fillna(EPSILON).values
    df["rad_k_th"] = k / np.where(th < EPSILON, EPSILON, th)
    df["rad_u_th"] = u / np.where(th < EPSILON, EPSILON, th)
    df["rad_th_k"] = th / np.where(k  < EPSILON, EPSILON, k)

    for col in ["zn_ppm", "pb_ppm", "cu_ppm", "ag_ppm", "au_ppb", "as_ppm"]:
        if col not in df.columns:
            df[col] = np.nan
        df[f"log_{col}"] = np.where(
            df[col].isna(), np.nan, np.log10(np.clip(df[col], EPSILON, None))
        )

    return df


def predict_and_write(
    model,
    coords: np.ndarray,
    transform,
    shape: tuple,
    feature_names: list,
    model_name: str
) -> Path:
    """Run prediction and write GeoTIFF."""
    log.info(f"\n[Prediction — {model_name}]")

    # Sample rasters
    grid_df = sample_all_rasters(coords)
    grid_df = apply_feature_engineering(grid_df)

    # Align to training features
    imputer = joblib.load(DATASET_DIR / "imputer.joblib")
    for col in feature_names:
        if col not in grid_df.columns:
            grid_df[col] = np.nan

    X_grid = imputer.transform(grid_df[feature_names].values.astype(np.float32))

    # Batch prediction (avoids RAM issues on large grids)
    BATCH = 50_000
    n     = len(X_grid)
    probs = np.full(n, np.nan, dtype=np.float32)

    log.info(f"  Predicting {n:,} cells in batches of {BATCH:,} ...")
    for i in range(0, n, BATCH):
        batch     = X_grid[i:i+BATCH]
        probs[i:i+BATCH] = model.predict_proba(batch)[:, 1]

    # Reshape to raster
    prob_grid = probs.reshape(shape).astype(np.float32)

    # Write GeoTIFF
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROSPECTIVITY_TIFF
    with rasterio.open(
        out_path, "w",
        driver="GTiff",
        height=shape[0], width=shape[1],
        count=1,
        dtype="float32",
        crs=CRS_TARGET,
        transform=transform,
        nodata=NODATA_VALUE,
        compress="lzw",
        tiled=True,
        blockxsize=256, blockysize=256
    ) as dst:
        dst.write(prob_grid, 1)
        dst.update_tags(
            model=model_name,
            description="VMS Prospectivity — Bathurst Mining Camp",
            units="Probability (0-1)",
            crs=str(CRS_TARGET),
            resolution_m=str(TARGET_RESOLUTION_M)
        )

    log.info(f"  ✅ Prospectivity map saved → {out_path}")
    log.info(f"  Probability range: {np.nanmin(probs):.4f} – {np.nanmax(probs):.4f}")
    log.info(f"  High-probability cells (>0.7): {(probs > 0.7).sum():,}")
    return out_path


def main():
    log.info("═══ Full-Extent Prospectivity Prediction ═══")

    # Load best model + feature list
    model, model_name = select_best_model()
    feature_names = pd.read_csv(
        DATASET_DIR / "feature_names.csv", header=None
    ).squeeze().tolist()

    # Build prediction grid
    log.info("\n[Building 100m prediction grid over BMC extent]")
    coords, transform, shape = build_prediction_grid()

    # Predict and write
    out_tiff = predict_and_write(
        model, coords, transform, shape, feature_names, model_name
    )

    log.info(
        f"\n✅ Complete. Open {out_tiff} in QGIS/ArcGIS for drill target review.\n"
        "   Run next: python pipeline/05_explainability/shap_analysis.py\n"
        "   Run next: python pipeline/04_prospectivity_map/export_map.py"
    )


if __name__ == "__main__":
    main()
