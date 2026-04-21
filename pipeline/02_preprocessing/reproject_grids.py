"""
reproject_grids.py
──────────────────
Reprojects all geophysical raster grids found in data/raw/rasters/ to the
project target CRS (EPSG:2953 — NAD83 / NB Double Stereographic) at a
consistent 100m pixel resolution.

This step ensures all raster layers share the same spatial reference before
feature extraction at sample point locations.

Usage:
    python pipeline/02_preprocessing/reproject_grids.py
"""

import sys
import logging
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling
)
from tqdm import tqdm

PIPELINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_DIR))
from config import (
    RASTERS_DIR, PROCESSED_DIR, CRS_TARGET,
    TARGET_RESOLUTION_M, NODATA_VALUE
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

REPROJECTED_DIR = PROCESSED_DIR / "rasters_reprojected"
SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".asc", ".grd"}


def reproject_raster(src_path: Path, dst_path: Path) -> None:
    """Reproject a single raster to target CRS at target resolution."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        log.info(f"  Source CRS   : {src.crs}")
        log.info(f"  Source shape : {src.height} × {src.width}")
        log.info(f"  Source dtype : {src.dtypes[0]}")

        # Calculate transform for target CRS at fixed resolution
        transform, width, height = calculate_default_transform(
            src.crs,
            CRS_TARGET,
            src.width,
            src.height,
            *src.bounds,
            resolution=TARGET_RESOLUTION_M
        )

        kwargs = src.meta.copy()
        kwargs.update({
            "crs": CRS_TARGET,
            "transform": transform,
            "width": width,
            "height": height,
            "nodata": NODATA_VALUE,
            "driver": "GTiff",
            "dtype": "float32",
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256
        })

        with rasterio.open(dst_path, "w", **kwargs) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=CRS_TARGET,
                    resampling=Resampling.bilinear
                )

    log.info(f"  Output shape : {height} × {width}")
    log.info(f"  ✅ Saved → {dst_path.name}")


def validate_raster(path: Path) -> bool:
    """Quick sanity check on a reprojected raster."""
    try:
        with rasterio.open(path) as r:
            assert str(r.crs.to_epsg()) == "2953", f"CRS mismatch: {r.crs}"
            data = r.read(1, masked=True)
            valid_pixels = data.count()
            assert valid_pixels > 0, "All pixels are nodata!"
            log.info(
                f"  Validation OK: {valid_pixels:,} valid pixels  "
                f"| min={float(data.min()):.2f}  max={float(data.max()):.2f}"
            )
        return True
    except Exception as e:
        log.error(f"  ✗ Validation failed: {e}")
        return False


def main():
    log.info("═══ Raster Reprojection Pipeline ═══")
    log.info(f"Target CRS        : {CRS_TARGET}")
    log.info(f"Target resolution : {TARGET_RESOLUTION_M}m")

    # Find all rasters in the raw rasters directory
    src_rasters = [
        p for p in RASTERS_DIR.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not src_rasters:
        log.warning(
            f"No raster files found in {RASTERS_DIR}\n"
            "  Run: python pipeline/01_data_download/download_nrcan_mag.py first,\n"
            "  or place downloaded GeoTIFF files in data/raw/rasters/"
        )
        return

    log.info(f"Found {len(src_rasters)} raster(s) to process")
    REPROJECTED_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for src_path in tqdm(src_rasters, desc="Reprojecting rasters"):
        dst_name = src_path.stem + "_epsg2953.tif"
        dst_path = REPROJECTED_DIR / dst_name

        if dst_path.exists():
            log.info(f"\n  Already reprojected — skipping: {dst_name}")
            results[src_path.name] = "skipped"
            continue

        log.info(f"\nProcessing: {src_path.name}")
        try:
            reproject_raster(src_path, dst_path)
            ok = validate_raster(dst_path)
            results[src_path.name] = "✅ ok" if ok else "⚠️ validation failed"
        except Exception as e:
            log.error(f"  ✗ Failed: {e}")
            results[src_path.name] = f"✗ {e}"

    # Summary
    log.info("\n─── Reprojection Summary ───")
    for name, status in results.items():
        log.info(f"  {name:40s} : {status}")

    log.info("\nRun next: python pipeline/02_preprocessing/extract_features.py")


if __name__ == "__main__":
    main()
