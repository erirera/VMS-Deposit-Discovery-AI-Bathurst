"""
download_nb_geochemistry.py
───────────────────────────
Downloads NB mineral occurrence and drill-hole point data from the live
GeoNB ArcGIS REST service and saves them as GeoPackages for use in the
VMS prospectivity model.

Live endpoint confirmed 2026-04-28:
  https://geonb.snb.ca/arcgis/rest/services/GeoNB_DNR_MineralOccurrences/MapServer
    Layer 0 — Mineral occurrences (point features, EPSG:2953)
    Layer 1 — Drill Holes      (point features, EPSG:2953)

NOTE on till geochemistry (ICP-MS):
  The NB GSB till geochemistry surveys are distributed as individual Open File
  data files via the 1:250,000 index search at:
    https://www1.gnb.ca/0078/GeoscienceDatabase/Till_GeoChem/TillIndex-e.asp
  These are per-quadrangle downloads and require the DATA_ACQUISITION_GUIDE.md
  manual steps.  The mineral occurrence and drill-hole layers fetched here provide
  the spatial label data needed for Phase 1 training.

Usage:
    python pipeline/01_data_download/download_nb_geochemistry.py
"""

import sys
import logging
import math
from pathlib import Path

import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm

# ── Path Setup ────────────────────────────────────────────────────────────────
PIPELINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_DIR))
from config import (
    RAW_DIR, GEOCHEMISTRY_GPKG, BMC_BBOX_WGS84, CRS_SOURCE, CRS_TARGET
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ── GeoNB REST Endpoints (confirmed live 2026-04-28) ─────────────────────────
GEONB_BASE = (
    "https://geonb.snb.ca/arcgis/rest/services/"
    "GeoNB_DNR_MineralOccurrences/MapServer"
)
MINERAL_LAYER   = 0   # Mineral occurrences (all NB)
DRILLHOLE_LAYER = 1   # Drill holes (all NB)

# ArcGIS REST page size limit
PAGE_SIZE = 1000

# BMC bounding box in the service's native CRS (EPSG:2953 / NAD83 CSRS)
# Converted from WGS84: lon_min=-67.5, lat_min=46.5, lon_max=-64.5, lat_max=48.5
# Using a generous envelope that captures the full Bathurst camp + dispersal train
BMC_ENVELOPE_2953 = {
    "xmin": 2310000,
    "ymin": 7410000,
    "xmax": 2740000,
    "ymax": 7660000,
}


def fetch_layer_page(layer_id: int, offset: int, envelope: dict) -> dict:
    """Fetch one page of features from the ArcGIS REST endpoint."""
    params = {
        "where": "1=1",
        "geometry": (
            f"{envelope['xmin']},{envelope['ymin']},"
            f"{envelope['xmax']},{envelope['ymax']}"
        ),
        "geometryType": "esriGeometryEnvelope",
        "inSR": "2953",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "true",
        "outSR": "4326",          # Request in WGS84 for easy GeoDataFrame construction
        "f": "json",
        "resultOffset": offset,
        "resultRecordCount": PAGE_SIZE,
    }
    url = f"{GEONB_BASE}/{layer_id}/query"
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def fetch_all_features(layer_id: int, layer_name: str) -> gpd.GeoDataFrame:
    """Paginate through all features in a GeoNB layer within the BMC envelope."""
    log.info(f"  Fetching layer {layer_id} ({layer_name}) ...")

    all_records = []
    offset = 0

    # First call to get total count
    count_resp = requests.get(
        f"{GEONB_BASE}/{layer_id}/query",
        params={
            "where": "1=1",
            "geometry": (
                f"{BMC_ENVELOPE_2953['xmin']},{BMC_ENVELOPE_2953['ymin']},"
                f"{BMC_ENVELOPE_2953['xmax']},{BMC_ENVELOPE_2953['ymax']}"
            ),
            "geometryType": "esriGeometryEnvelope",
            "inSR": "2953",
            "spatialRel": "esriSpatialRelIntersects",
            "returnCountOnly": "true",
            "f": "json",
        },
        timeout=60,
    )
    count_resp.raise_for_status()
    total = count_resp.json().get("count", 0)
    log.info(f"    Total features in BMC envelope: {total}")

    pages = math.ceil(total / PAGE_SIZE) if total > 0 else 1
    with tqdm(total=total, unit="features", desc=f"  Layer {layer_id}") as bar:
        for _ in range(pages):
            data = fetch_layer_page(layer_id, offset, BMC_ENVELOPE_2953)
            features = data.get("features", [])
            if not features:
                break
            for f in features:
                attrs = f.get("attributes", {})
                geom  = f.get("geometry", {})
                attrs["longitude"] = geom.get("x")
                attrs["latitude"]  = geom.get("y")
                all_records.append(attrs)
            bar.update(len(features))
            offset += PAGE_SIZE
            if len(features) < PAGE_SIZE:
                break

    if not all_records:
        log.warning(f"    No features returned for layer {layer_id}")
        return gpd.GeoDataFrame()

    df = pd.DataFrame(all_records)
    # Standardise column names to lowercase
    df.columns = [c.lower().strip() for c in df.columns]

    # Build geometry
    geometry = [
        Point(row["longitude"], row["latitude"])
        for _, row in df.iterrows()
        if pd.notna(row.get("longitude")) and pd.notna(row.get("latitude"))
    ]
    df = df[df["longitude"].notna() & df["latitude"].notna()].reset_index(drop=True)
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=CRS_SOURCE)  # WGS84 from outSR=4326
    gdf = gdf.to_crs(CRS_TARGET)  # → EPSG:2953
    log.info(f"    Loaded {len(gdf)} features, reprojected to {CRS_TARGET}")
    return gdf


def main():
    log.info("═══ GeoNB Mineral Occurrences & Drill Holes Download ═══")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = RAW_DIR / "geonb"
    out_dir.mkdir(exist_ok=True)

    # ── Layer 0: Mineral Occurrences ─────────────────────────────────────────
    log.info("\n[1/2] Mineral Occurrences (Layer 0) ...")
    mineral_gpkg = out_dir / "nb_mineral_occurrences.gpkg"
    if mineral_gpkg.exists():
        log.info(f"  Already exists — skipping: {mineral_gpkg}")
        mineral_gdf = gpd.read_file(mineral_gpkg)
    else:
        mineral_gdf = fetch_all_features(MINERAL_LAYER, "Mineral Occurrences")
        if not mineral_gdf.empty:
            mineral_gdf.to_file(mineral_gpkg, driver="GPKG", layer="mineral_occurrences")
            log.info(f"  ✅ Saved → {mineral_gpkg}")

    # ── Layer 1: Drill Holes ─────────────────────────────────────────────────
    log.info("\n[2/2] Drill Holes (Layer 1) ...")
    drillhole_gpkg = out_dir / "nb_drill_holes.gpkg"
    if drillhole_gpkg.exists():
        log.info(f"  Already exists — skipping: {drillhole_gpkg}")
        drill_gdf = gpd.read_file(drillhole_gpkg)
    else:
        drill_gdf = fetch_all_features(DRILLHOLE_LAYER, "Drill Holes")
        if not drill_gdf.empty:
            drill_gdf.to_file(drillhole_gpkg, driver="GPKG", layer="drill_holes")
            log.info(f"  ✅ Saved → {drillhole_gpkg}")

    # ── Summary ──────────────────────────────────────────────────────────────
    log.info("\n─── Download Summary ───")
    log.info(f"  Mineral occurrences : {len(mineral_gdf):5d} points")
    log.info(f"  Drill holes         : {len(drill_gdf):5d} points")
    log.info(f"  Output directory    : {out_dir}")
    log.info("\nNOTE: Till ICP-MS geochemistry requires manual download.")
    log.info("      See pipeline/DATA_ACQUISITION_GUIDE.md for instructions.")
    log.info("\nRun next: python pipeline/01_data_download/download_vms_labels.py")


if __name__ == "__main__":
    main()
