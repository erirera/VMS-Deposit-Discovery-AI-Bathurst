"""
download_nb_geochemistry.py
───────────────────────────
Downloads NB Geological Survey till geochemistry data and saves it as a
GeoPackage for use in the VMS prospectivity model.

Data source:
  NB Open Data — Surficial Geochemistry (Till) Program
  https://www2.gnb.ca/content/gnb/en/departments/erd/energy/content/minerals/content/geology_data.html

The till geochemistry dataset records heavy mineral and multi-element
(ICP-MS) analyses of glacial till samples across New Brunswick.
Key pathfinder elements for VMS prospecting: Zn, Pb, Cu, Ag, Au, As.

Usage:
    python pipeline/01_data_download/download_nb_geochemistry.py
"""

import sys
import logging
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

# ── NB Open Data — Till Geochemistry ─────────────────────────────────────────
# The NB Geological Survey provides till geochemistry as a CSV download via
# their Open Data portal. The URL below fetches the current published dataset.
# If this URL changes, find the latest at: https://data-donnees.az.ec.gc.ca/
NB_GEOCHEM_URL = (
    "https://data-donnees.az.ec.gc.ca/api/file?"
    "path=/ess-sst/3ac09dba-e7e3-4b72-b891-a49abf36cf48/"
    "nb-till-geochemistry.csv"
)

# Fallback — NB Open Data GeoJSON endpoint (WFS)
NB_GEOCHEM_WFS = (
    "https://geonb.snb.ca/arcgis/rest/services/GeoNB_EN_Geology/"
    "MapServer/5/query?where=1%3D1&outFields=*&f=geojson"
)

# Column mappings from raw NB GSB till geochemistry export
COLUMN_MAP = {
    # Raw name              : Standardised name
    "LONGITUDE":              "longitude",
    "LATITUDE":               "latitude",
    "ZN_PPM":                 "zn_ppm",
    "PB_PPM":                 "pb_ppm",
    "CU_PPM":                 "cu_ppm",
    "AG_PPM":                 "ag_ppm",
    "AU_PPB":                 "au_ppb",
    "AS_PPM":                 "as_ppm",
    "SAMPLE_ID":              "sample_id",
    "YEAR":                   "year",
    "DEPTH_CM":               "depth_cm",
    # Alternative naming convention
    "Longitude":              "longitude",
    "Latitude":               "latitude",
    "Zn_ppm":                 "zn_ppm",
    "Pb_ppm":                 "pb_ppm",
    "Cu_ppm":                 "cu_ppm",
    "Ag_ppm":                 "ag_ppm",
    "Au_ppb":                 "au_ppb",
    "As_ppm":                 "as_ppm",
}

REQUIRED_COLUMNS = ["longitude", "latitude", "zn_ppm", "pb_ppm", "cu_ppm"]
PATHFINDER_ELEMENTS = ["zn_ppm", "pb_ppm", "cu_ppm", "ag_ppm", "au_ppb", "as_ppm"]


def download_csv(url: str, dest_path: Path) -> Path:
    """Download a file from URL with a progress bar. Returns local path."""
    log.info(f"Downloading: {url}")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(dest_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest_path.name
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    log.info(f"Saved to: {dest_path}")
    return dest_path


def load_and_clean(csv_path: Path) -> pd.DataFrame:
    """Load the raw CSV, standardise columns, and remove invalid rows."""
    log.info(f"Loading {csv_path.name} ...")
    df = pd.read_csv(csv_path, low_memory=False)
    log.info(f"  Raw shape: {df.shape}")

    # Standardise column names
    df.rename(columns=COLUMN_MAP, inplace=True)
    df.columns = [c.lower().strip() for c in df.columns]

    # Check required columns are present
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Required columns missing from downloaded data: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Coerce numeric pathfinder columns
    for col in PATHFINDER_ELEMENTS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with no coordinates
    df.dropna(subset=["longitude", "latitude"], inplace=True)

    # Enforce valid coordinate ranges
    df = df[
        (df["longitude"].between(-180, 180)) &
        (df["latitude"].between(-90, 90))
    ]

    log.info(f"  Cleaned shape: {df.shape}")
    return df


def clip_to_bmc(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Clip to Bathurst Mining Camp bounding box + 50km buffer."""
    lon_min, lat_min, lon_max, lat_max = BMC_BBOX_WGS84
    # Add a generous buffer so we capture till samples in the dispersal train
    buffer = 0.5  # ~50km in degrees at this latitude
    clipped = gdf.cx[
        lon_min - buffer : lon_max + buffer,
        lat_min - buffer : lat_max + buffer
    ]
    log.info(f"  BMC clip: {len(gdf)} → {len(clipped)} samples retained")
    return clipped.copy()


def build_geodataframe(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Convert DataFrame to GeoDataFrame in WGS84, reprojects to EPSG:2953."""
    geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=CRS_SOURCE)
    gdf = gdf.to_crs(CRS_TARGET)
    log.info(f"  Reprojected to {CRS_TARGET}")
    return gdf


def summarise(gdf: gpd.GeoDataFrame):
    """Print a summary of the pathfinder element distributions."""
    log.info("\n─── Pathfinder Summary (BMC Till Samples) ───")
    for col in PATHFINDER_ELEMENTS:
        if col in gdf.columns and gdf[col].notna().any():
            log.info(
                f"  {col:12s}: n={gdf[col].notna().sum():5d}  "
                f"median={gdf[col].median():.2f}  "
                f"max={gdf[col].max():.2f}"
            )


def main():
    log.info("═══ NB Till Geochemistry Download ═══")

    # ── Step 1: Download raw CSV ─────────────────────────────────────────────
    raw_csv = RAW_DIR / "nb_till_geochemistry_raw.csv"

    if raw_csv.exists():
        log.info(f"Raw CSV already exists — skipping download: {raw_csv}")
    else:
        try:
            download_csv(NB_GEOCHEM_URL, raw_csv)
        except Exception as e:
            log.warning(f"Primary URL failed ({e}). Trying WFS endpoint ...")
            # Fall back: fetch GeoJSON from GeoNB WFS
            resp = requests.get(NB_GEOCHEM_WFS, timeout=120)
            resp.raise_for_status()
            gdf_raw = gpd.read_file(resp.text)
            gdf_raw.to_csv(raw_csv, index=False)
            log.info(f"WFS data saved to: {raw_csv}")

    # ── Step 2: Clean & structure ────────────────────────────────────────────
    df = load_and_clean(raw_csv)

    # ── Step 3: Build GeoDataFrame & clip to BMC ────────────────────────────
    gdf = build_geodataframe(df)
    gdf = clip_to_bmc(gdf)

    # ── Step 4: Summary statistics ───────────────────────────────────────────
    summarise(gdf)

    # ── Step 5: Save output GeoPackage ──────────────────────────────────────
    GEOCHEMISTRY_GPKG.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(GEOCHEMISTRY_GPKG, driver="GPKG", layer="till_geochemistry")
    log.info(f"\n✅ Saved {len(gdf)} samples → {GEOCHEMISTRY_GPKG}")
    log.info("   Run next: python pipeline/01_data_download/download_vms_labels.py")


if __name__ == "__main__":
    main()
