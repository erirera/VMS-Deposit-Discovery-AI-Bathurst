"""
predict_full_extent.py
───────────────────────
Runs BOTH trained models (RF and XGBoost) across the full ~3,800 km² Bathurst
Mining Camp extent to generate two continuous prospectivity maps.

Outputs:
  outputs/rf_prospectivity_map.tif   ← Random Forest
  outputs/xgb_prospectivity_map.tif  ← XGBoost

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
from scipy.ndimage import gaussian_filter

PIPELINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_DIR))
from config import (
    PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR,
    RF_MODEL_PATH, XGB_MODEL_PATH,
    RF_PROSPECTIVITY_TIFF, XGB_PROSPECTIVITY_TIFF,
    BMC_BBOX_WGS84, CRS_SOURCE, CRS_TARGET,
    TARGET_RESOLUTION_M, NODATA_VALUE, MASTER_RASTER_PATH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

DATASET_DIR = PROCESSED_DIR / "training_dataset"
RASTERS_DIR = PROCESSED_DIR / "rasters_reprojected"

# Both models and their output paths
MODELS_TO_RUN = [
    ("Random Forest", RF_MODEL_PATH,  RF_PROSPECTIVITY_TIFF),
    ("XGBoost",       XGB_MODEL_PATH, XGB_PROSPECTIVITY_TIFF),
]


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

    log.info(f"  Grid: {height} rows x {width} cols = {height*width:,} cells")
    log.info(f"  Extent: ({minx:.0f}, {miny:.0f}) -> ({maxx:.0f}, {maxy:.0f}) [EPSG:2953]")
    return coords, transform, (height, width)


def sample_all_rasters(coords, transform, shape):
    """Sample all reprojected rasters at grid point coordinates."""
    from rasterio.warp import reproject, Resampling
    raster_paths = sorted(RASTERS_DIR.glob("*.tif"))

    if not raster_paths:
        log.warning("  No rasters found -- prediction will use NaN-filled raster features.")
        return pd.DataFrame()

    results = {}
    height, width = shape
    log.info(f"  Sampling {len(raster_paths)} rasters at {len(coords):,} grid points ...")

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
        valid = np.isfinite(results[col]).sum()
        log.info(f"    {col:40s}: {valid:,} / {len(results[col]):,} valid")

    return pd.DataFrame(results)


def apply_feature_engineering(df):
    """Apply same derived features as engineer_features.py."""
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
    df["rad_k_th"] = k  / np.where(th < EPSILON, EPSILON, th)
    df["rad_u_th"] = u  / np.where(th < EPSILON, EPSILON, th)
    df["rad_th_k"] = th / np.where(k  < EPSILON, EPSILON, k)

    for col in ["zn_ppm", "pb_ppm", "cu_ppm", "ag_ppm", "au_ppb", "as_ppm"]:
        if col not in df.columns:
            df[col] = np.nan
        df[f"log_{col}"] = np.where(
            df[col].isna(), np.nan, np.log10(np.clip(df[col], EPSILON, None))
        )
    return df


def smooth_probability_grid(prob_grid, sigma=(1.0, 1.0)):
    """Apply a lightweight Gaussian smoothing to reduce striping artifacts."""
    valid = np.isfinite(prob_grid)
    if not valid.any():
        return prob_grid.astype(np.float32)

    filled = np.where(valid, prob_grid, 0.0).astype(np.float32)
    filtered = gaussian_filter(filled, sigma=sigma, mode="nearest")
    filtered = np.where(valid, filtered, np.nan)
    filtered = np.clip(filtered, 0.0, 1.0).astype(np.float32)
    return filtered


def predict_and_write(model, grid_df, feature_names, imputer,
                      model_name, out_path, shape, transform):
    """Run prediction and write GeoTIFF."""
    log.info(f"\n[Prediction -- {model_name}]")

    for col in feature_names:
        if col not in grid_df.columns:
            grid_df[col] = np.nan

    missing = [
        col for col in feature_names
        if col not in grid_df.columns
    ]
    print(f"Missing features: {len(missing)}")

    for m in missing:
        print(m)

    X_grid = imputer.transform(grid_df[feature_names].values.astype(np.float32))

    BATCH = 50_000
    n     = len(X_grid)
    probs = np.full(n, np.nan, dtype=np.float32)

    log.info(f"  Predicting {n:,} cells in batches of {BATCH:,} ...")
    for i in range(0, n, BATCH):
        probs[i:i+BATCH] = model.predict_proba(X_grid[i:i+BATCH])[:, 1]

    prob_grid = probs.reshape(shape).astype(np.float32)
    prob_grid = smooth_probability_grid(prob_grid, sigma=(1.0, 1.0))
    log.info(f"  Applied Gaussian smoothing: sigma=(1.0, 1.0)")
    out_path.parent.mkdir(parents=True, exist_ok=True)

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
            description="VMS Prospectivity -- Bathurst Mining Camp",
            units="Probability (0-1)",
            crs=str(CRS_TARGET),
            resolution_m=str(TARGET_RESOLUTION_M)
        )

    log.info(f"  Saved -> {out_path.name}")
    log.info(f"  Probability range  : {np.nanmin(probs):.4f} - {np.nanmax(probs):.4f}")
    log.info(f"  Median probability : {np.nanmedian(probs):.4f}")
    log.info(f"  High-prob (>0.7)   : {(probs > 0.7).sum():,} cells")
    log.info(f"  Mod-prob  (>0.5)   : {(probs > 0.5).sum():,} cells  ({100*(probs > 0.5).mean():.1f}%)")
    return out_path


def main():
    log.info("=== Full-Extent Prospectivity Prediction -- RF and XGBoost ===")

    feature_names = pd.read_csv(
        DATASET_DIR / "feature_names.csv", header=None
    ).squeeze().tolist()
    imputer = joblib.load(DATASET_DIR / "imputer.joblib")
    log.info(f"  Feature columns: {len(feature_names)}")

    log.info("\n[Building 100 m prediction grid over BMC extent]")
    coords, transform, shape = build_prediction_grid()

    log.info("\n[Sampling geophysical rasters (once, shared across models)]")
    grid_df_base = sample_all_rasters(coords, transform, shape)
    grid_df_base = apply_feature_engineering(grid_df_base)

    outputs = {}
    for model_name, model_path, out_tiff in MODELS_TO_RUN:
        if not model_path.exists():
            log.warning(f"  Model not found: {model_path} -- skipping {model_name}")
            continue
        model = joblib.load(model_path)
        log.info(f"  Loaded {model_name} from {model_path.name}")
        tiff = predict_and_write(
            model=model,
            grid_df=grid_df_base.copy(),
            feature_names=feature_names,
            imputer=imputer,
            model_name=model_name,
            out_path=out_tiff,
            shape=shape,
            transform=transform,
        )
        outputs[model_name] = tiff

    log.info("\n=== Output Summary ===")
    for name, path in outputs.items():
        log.info(f"  {name:15s}: {path.name}")

    log.info(
        "\nNext steps:\n"
        "  python pipeline/03_training/success_rate_curve.py\n"
        "  python pipeline/04_prospectivity_map/export_map.py\n"
        "  python pipeline/05_explainability/shap_analysis.py"
    )


if __name__ == "__main__":
    main()
