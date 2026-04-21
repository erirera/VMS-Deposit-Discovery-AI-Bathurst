"""
extract_features.py
────────────────────
Samples all reprojected geophysical raster grids at the locations of the
till geochemistry samples and training labels, building the feature matrix
used for model training.

Strategy:
  For each point (till sample or label location), extract the pixel values
  from all available reprojected rasters. The result is merged with the
  till geochemistry pathfinder data to form a combined feature vector.

Output:
  data/processed/feature_matrix.parquet
  Columns: [point_id, label, geometry_wkt, <raster features>, <geochem features>]

Usage:
    python pipeline/02_preprocessing/extract_features.py
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.sample import sample_gen
from tqdm import tqdm

PIPELINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_DIR))
from config import (
    PROCESSED_DIR, GEOCHEMISTRY_GPKG, VMS_LABELS_GPKG, BARREN_LABELS_GPKG,
    FEATURE_MATRIX_PQ, CRS_TARGET, NODATA_VALUE
)

PATHFINDER_ELEMENTS = ["zn_ppm", "pb_ppm", "cu_ppm", "ag_ppm", "au_ppb", "as_ppm"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

REPROJECTED_RASTERS_DIR = PROCESSED_DIR / "rasters_reprojected"


def load_label_points() -> gpd.GeoDataFrame:
    """Load VMS deposits and barren holes into a unified GeoDataFrame."""
    gdfs = []

    if VMS_LABELS_GPKG.exists():
        vms = gpd.read_file(VMS_LABELS_GPKG, layer="vms_deposits")
        vms["point_id"] = "VMS_" + vms.index.astype(str)
        vms["source"] = "vms_deposit"
        gdfs.append(vms[["point_id", "label", "source", "geometry"]])
        log.info(f"  Loaded {len(vms)} VMS deposit points")
    else:
        log.warning(f"  VMS labels not found: {VMS_LABELS_GPKG}")

    if BARREN_LABELS_GPKG.exists():
        barren = gpd.read_file(BARREN_LABELS_GPKG, layer="barren_holes")
        barren["point_id"] = "BARREN_" + barren.index.astype(str)
        barren["source"] = "barren_hole"
        gdfs.append(barren[["point_id", "label", "source", "geometry"]])
        log.info(f"  Loaded {len(barren)} barren drill points")
    else:
        log.warning(f"  Barren labels not found: {BARREN_LABELS_GPKG}")

    if not gdfs:
        raise FileNotFoundError(
            "No label files found. Run download_vms_labels.py first."
        )

    gdf = pd.concat(gdfs, ignore_index=True)
    if gdf.crs != CRS_TARGET:
        gdf = gdf.to_crs(CRS_TARGET)
    return gdf


def load_geochem_points() -> gpd.GeoDataFrame:
    """Load till geochemistry samples (no labels — used as prediction points)."""
    if not GEOCHEMISTRY_GPKG.exists():
        log.warning(f"Geochemistry file not found: {GEOCHEMISTRY_GPKG}")
        return None
    gdf = gpd.read_file(GEOCHEMISTRY_GPKG, layer="till_geochemistry")
    gdf["point_id"] = "GEOCHEM_" + gdf.index.astype(str)
    gdf["label"] = np.nan   # Unknown labels for spatial prediction
    gdf["source"] = "till_geochem"
    log.info(f"  Loaded {len(gdf)} till geochemistry sample points")
    return gdf


def sample_rasters_at_points(
    gdf: gpd.GeoDataFrame,
    raster_dir: Path
) -> pd.DataFrame:
    """
    Sample all rasters in raster_dir at point locations.
    Returns a DataFrame with one column per raster band.
    """
    raster_paths = sorted(raster_dir.glob("*.tif"))
    if not raster_paths:
        log.warning(f"  No reprojected rasters found in {raster_dir}")
        return pd.DataFrame(index=gdf.index)

    log.info(f"  Sampling {len(raster_paths)} raster(s) at {len(gdf)} points ...")

    # Coordinates as list of (x, y) tuples in target CRS
    coords = [(geom.x, geom.y) for geom in gdf.geometry]
    results = {}

    for raster_path in tqdm(raster_paths, desc="Raster sampling"):
        col_name = raster_path.stem.replace("_epsg2953", "")
        with rasterio.open(raster_path) as src:
            sampled = list(sample_gen(src, coords))
            values = [
                float(v[0]) if v[0] != src.nodata and not np.isnan(v[0])
                else np.nan
                for v in sampled
            ]
        results[col_name] = values
        log.info(
            f"    {col_name:30s}: "
            f"{sum(1 for v in values if not np.isnan(v)):5d} valid / {len(values)} total"
        )

    return pd.DataFrame(results, index=gdf.index)


def merge_geochem_features(
    points_df: pd.DataFrame,
    geochem_gdf: gpd.GeoDataFrame
) -> pd.DataFrame:
    """
    Merge geochemistry pathfinder element values onto label points
    via spatial nearest-neighbour join (within 1km search radius).
    """
    if geochem_gdf is None:
        log.warning("  No geochemistry data — skipping geochem feature merge")
        for col in ["zn_ppm", "pb_ppm", "cu_ppm", "ag_ppm", "au_ppb", "as_ppm"]:
            points_df[col] = np.nan
        return points_df

    log.info("  Performing spatial nearest-join for geochemistry features ...")

    # Convert to GeoDataFrame for nearest join
    from shapely.wkt import loads
    points_gdf_tmp = gpd.GeoDataFrame(points_df, crs=CRS_TARGET)

    # Nearest join: for each label point, find the closest till sample
    geochem_sub = geochem_gdf[
        [c for c in PATHFINDER_ELEMENTS if c in geochem_gdf.columns] + ["geometry"]
    ].copy()

    joined = gpd.sjoin_nearest(
        points_gdf_tmp,
        geochem_sub,
        how="left",
        max_distance=1000   # 1km search radius
    )

    for col in PATHFINDER_ELEMENTS:
        if col in joined.columns:
            points_df[col] = joined[col].values
        else:
            points_df[col] = np.nan

    log.info(f"  Geochem merge — matched within 1km: "
             f"{joined['zn_ppm'].notna().sum() if 'zn_ppm' in joined.columns else 'N/A'} "
             f"/ {len(joined)}")
    return points_df


def log_feature_matrix_summary(df: pd.DataFrame):
    """Print a null-count and summary table for quality control."""
    log.info("\n─── Feature Matrix Quality Report ───")
    log.info(f"  Shape: {df.shape}")
    log.info(f"  Label distribution:\n{df['label'].value_counts(dropna=False).to_string()}")
    log.info("\n  Column null counts (top 20 by nulls):")
    nulls = df.isnull().sum().sort_values(ascending=False).head(20)
    for col, n in nulls.items():
        pct = 100 * n / len(df)
        flag = " ⚠️" if pct > 30 else ""
        log.info(f"    {col:35s}: {n:5d} ({pct:5.1f}%){flag}")


def main():
    log.info("═══ Feature Matrix Construction ═══")

    # ── Load points ──────────────────────────────────────────────────────────
    log.info("\n[1/4] Loading label points ...")
    labels_gdf = load_label_points()

    log.info("\n[2/4] Loading geochemistry sample points ...")
    geochem_gdf = load_geochem_points()

    # ── Sample geophysical rasters ───────────────────────────────────────────
    log.info("\n[3/4] Sampling geophysical rasters at label locations ...")
    raster_features_df = sample_rasters_at_points(labels_gdf, REPROJECTED_RASTERS_DIR)

    # Combine label info with raster features
    combined_df = pd.concat(
        [
            labels_gdf[["point_id", "label", "source"]].reset_index(drop=True),
            raster_features_df.reset_index(drop=True)
        ],
        axis=1
    )

    # Add geometry as WKT for reference (not used in training)
    combined_df["geometry_wkt"] = labels_gdf.geometry.to_wkt().values

    # ── Merge geochemistry features ──────────────────────────────────────────
    log.info("\n[4/4] Merging till geochemistry features ...")
    combined_df = merge_geochem_features(combined_df, geochem_gdf)

    # ── Quality report ───────────────────────────────────────────────────────
    log_feature_matrix_summary(combined_df)

    # ── Save ─────────────────────────────────────────────────────────────────
    FEATURE_MATRIX_PQ.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_parquet(FEATURE_MATRIX_PQ, index=False)
    log.info(f"\n✅ Feature matrix saved → {FEATURE_MATRIX_PQ}")
    log.info("   Run next: python pipeline/02_preprocessing/engineer_features.py")


if __name__ == "__main__":
    main()
