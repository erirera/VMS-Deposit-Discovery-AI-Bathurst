"""
interpolate_geochem.py  --  Till Geochemistry Point -> Raster Interpolation
============================================================================
Bathurst Mining Camp VMS Deposit Discovery AI Pipeline
Step 2b (optional enrichment): Interpolate BMC till geochemistry GPKG files
into continuous GeoTIFF raster surfaces.

Supported methods
-----------------
  idw      Inverse Distance Weighting (fast, no extra assumptions, default)
  kriging  Ordinary Kriging (geostatistical, assumes spatial correlation)
           Requires pykrige >=1.7  (already in environment)

Output location
---------------
  data/processed/rasters_reprojected/geochem_<element>_<method>.tif

These files are automatically discovered and sampled by extract_features.py
via its raster-sampling loop -- no changes to that script are required.

Usage
-----
  # IDW (default, fast)
  python pipeline/02_preprocessing/interpolate_geochem.py

  # Kriging (slower but geostatistically rigorous)
  python pipeline/02_preprocessing/interpolate_geochem.py --method kriging

  # Single element test run
  python pipeline/02_preprocessing/interpolate_geochem.py --element Cu

  # Overwrite existing outputs
  python pipeline/02_preprocessing/interpolate_geochem.py --overwrite

  # IDW with custom power and neighbours
  python pipeline/02_preprocessing/interpolate_geochem.py --idw-power 3 --idw-neighbours 16
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Resolve repo root and import shared config
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "pipeline"))
from config import (  # noqa: E402
    CRS_TARGET,
    NODATA_VALUE,
    RASTERS_DIR,
    PROCESSED_DIR,
)

REPROJECTED_RASTERS_DIR = PROCESSED_DIR / "rasters_reprojected"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("interpolate_geochem")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GEOCHEM_DIR = RASTERS_DIR           # data/raw/rasters/bmc_*.gpkg
OUT_DIR     = REPROJECTED_RASTERS_DIR
RESOLUTION  = 50                    # metres -- matches existing derivative rasters
IDW_POWER_DEFAULT      = 2
IDW_NEIGHBOURS_DEFAULT = 12

# GPKG stem -> output raster name stem
ELEMENT_MAP: dict[str, str] = {
    "bmc_Ag": "geochem_ag_ppm",
    "bmc_As": "geochem_as_ppm",
    "bmc_Ba": "geochem_ba_ppm",
    "bmc_Bi": "geochem_bi_ppm",
    "bmc_Cd": "geochem_cd_ppm",
    "bmc_Co": "geochem_co_ppm",
    "bmc_Cu": "geochem_cu_ppm",
    "bmc_Fe": "geochem_fe_ppm",
    "bmc_In": "geochem_in_ppm",
    "bmc_Mn": "geochem_mn_ppm",
    "bmc_Mo": "geochem_mo_ppm",
    "bmc_Ni": "geochem_ni_ppm",
    "bmc_Pb": "geochem_pb_ppm",
    "bmc_Sb": "geochem_sb_ppm",
    "bmc_Sn": "geochem_sn_ppm",
    "bmc_Tl": "geochem_tl_ppm",
    "bmc_Zn": "geochem_zn_ppm",
}


# ===========================================================================
# IDW Interpolation
# ===========================================================================

def idw_interpolate(
    pts: np.ndarray,
    values: np.ndarray,
    grid_xy: np.ndarray,
    power: float = 2.0,
    k: int = 12,
) -> np.ndarray:
    """Inverse Distance Weighting interpolation.

    Parameters
    ----------
    pts      : (N, 2) array of known point coordinates (x, y)
    values   : (N,)  array of known values at pts
    grid_xy  : (M, 2) array of query coordinates to estimate
    power    : IDW distance power (default 2 = classic)
    k        : Number of nearest neighbours to use

    Returns
    -------
    (M,) array of interpolated values at grid_xy
    """
    tree = cKDTree(pts)
    dists, idxs = tree.query(grid_xy, k=k, workers=-1)

    # Guard against exact coincident points (distance = 0)
    zero_mask = dists[:, 0] == 0
    weights = np.where(dists == 0, 0.0, 1.0 / (dists ** power))
    weight_sums = weights.sum(axis=1)
    result = (weights * values[idxs]).sum(axis=1) / weight_sums

    # For exact hits: use the coincident sample value directly
    result[zero_mask] = values[idxs[zero_mask, 0]]
    return result


def rasterise_idw(
    gdf: gpd.GeoDataFrame,
    out_path: Path,
    resolution: float = RESOLUTION,
    power: float = IDW_POWER_DEFAULT,
    k: int = IDW_NEIGHBOURS_DEFAULT,
    snap_bounds: tuple | None = None,
) -> None:
    """Interpolate a point GeoDataFrame using IDW and write a GeoTIFF.

    Parameters
    ----------
    snap_bounds : optional (xmin, ymin, xmax, ymax) tuple to force the output
                  grid extent, overriding the point-cloud bounding box.
                  Use this to ensure the raster covers the full study area.
    """
    coords = np.column_stack([gdf.geometry.x.values, gdf.geometry.y.values])
    values = gdf["VALUE"].values.astype(np.float64)

    valid = np.isfinite(values) & (values >= 0)
    if valid.sum() < 4:
        log.warning("  Only %d valid samples -- skipping IDW.", valid.sum())
        return
    coords, values = coords[valid], values[valid]

    if snap_bounds is not None:
        xmin, ymin, xmax, ymax = snap_bounds
        log.info("    Extent snapped to reference bounds.")
    else:
        xmin = coords[:, 0].min()
        ymin = coords[:, 1].min()
        xmax = coords[:, 0].max()
        ymax = coords[:, 1].max()
    ncols = int(np.ceil((xmax - xmin) / resolution)) + 1
    nrows = int(np.ceil((ymax - ymin) / resolution)) + 1

    xs = np.arange(ncols) * resolution + xmin
    ys = np.arange(nrows) * resolution + ymin
    gx, gy = np.meshgrid(xs, ys)
    grid_xy = np.column_stack([gx.ravel(), gy.ravel()])

    log.info("    Grid  : %d cols x %d rows  (%.0f m resolution)", ncols, nrows, resolution)
    log.info("    Points: %d valid samples", len(values))

    t0 = time.perf_counter()
    interp = idw_interpolate(coords, values, grid_xy, power=power, k=k)
    log.info("    IDW done in %.1f s", time.perf_counter() - t0)

    grid = interp.reshape(nrows, ncols).astype(np.float32)
    transform = from_bounds(
        xmin, ymin,
        xmin + ncols * resolution, ymin + nrows * resolution,
        ncols, nrows,
    )
    _write_tiff(grid, transform, out_path)


# ===========================================================================
# Ordinary Kriging
# ===========================================================================

def rasterise_kriging(
    gdf: gpd.GeoDataFrame,
    out_path: Path,
    resolution: float = RESOLUTION,
    variogram_model: str = "spherical",
    max_points: int = 500,
    chunk_size: int = 25_000,
) -> None:
    """Interpolate a point GeoDataFrame using Ordinary Kriging and write a GeoTIFF.

    Parameters
    ----------
    variogram_model : pykrige variogram model ('linear', 'power', 'gaussian',
                      'spherical', 'exponential', 'hole-effect')
    max_points      : Maximum sample size for variogram fitting. Kriging is
                      O(N^3) so large datasets must be subsampled.
    chunk_size      : Number of grid cells predicted per batch. Controls peak
                      RAM usage: peak ~ chunk_size * max_points * 8 bytes.
                      Default 25 000 -> ~100 MB per chunk at max_points=500.
    """
    try:
        from pykrige.ok import OrdinaryKriging
    except ImportError:
        log.error("pykrige is not installed. Run: pip install pykrige")
        raise

    coords = np.column_stack([gdf.geometry.x.values, gdf.geometry.y.values])
    values = gdf["VALUE"].values.astype(np.float64)

    valid = np.isfinite(values) & (values >= 0)
    if valid.sum() < 4:
        log.warning("  Only %d valid samples -- skipping Kriging.", valid.sum())
        return
    coords, values = coords[valid], values[valid]

    # Subsample if too many points (Kriging is O(N^3))
    if len(values) > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(values), size=max_points, replace=False)
        coords, values = coords[idx], values[idx]
        log.info("    Subsampled to %d points for Kriging.", max_points)

    xmin = coords[:, 0].min()
    ymin = coords[:, 1].min()
    xmax = coords[:, 0].max()
    ymax = coords[:, 1].max()
    ncols = int(np.ceil((xmax - xmin) / resolution)) + 1
    nrows = int(np.ceil((ymax - ymin) / resolution)) + 1

    xs = np.arange(ncols) * resolution + xmin
    ys = np.arange(nrows) * resolution + ymin

    log.info("    Grid     : %d cols x %d rows  (%d cells)", ncols, nrows, ncols * nrows)
    log.info("    Variogram: %s  (max_points=%d  chunk=%d)", variogram_model, max_points, chunk_size)

    t0 = time.perf_counter()
    ok = OrdinaryKriging(
        coords[:, 0], coords[:, 1], values,
        variogram_model=variogram_model,
        verbose=False,
        enable_plotting=False,
        nlags=12,
        coordinates_type="euclidean",
    )

    # --- Chunked prediction to avoid multi-GB allocations ---
    # Build flat grid of (x, y) query points
    gx, gy = np.meshgrid(xs, ys)          # shape (nrows, ncols) each
    flat_x = gx.ravel()                    # shape (nrows*ncols,)
    flat_y = gy.ravel()
    n_cells = len(flat_x)
    z_flat = np.empty(n_cells, dtype=np.float64)

    n_chunks = int(np.ceil(n_cells / chunk_size))
    for i in range(n_chunks):
        s = i * chunk_size
        e = min(s + chunk_size, n_cells)
        z_chunk, _ = ok.execute("points", flat_x[s:e], flat_y[s:e])
        z_flat[s:e] = z_chunk
        if (i + 1) % 20 == 0 or (i + 1) == n_chunks:
            log.info("      chunk %d/%d  (%.0f%%)", i + 1, n_chunks,
                     100.0 * (i + 1) / n_chunks)

    log.info("    Kriging done in %.1f s", time.perf_counter() - t0)

    # Reshape, clip negatives (Kriging can predict slightly below 0), flip
    grid = np.clip(z_flat.reshape(nrows, ncols), 0, None).astype(np.float32)
    grid = np.flipud(grid)
    transform = from_bounds(
        xmin, ymin,
        xmin + ncols * resolution, ymin + nrows * resolution,
        ncols, nrows,
    )
    _write_tiff(grid, transform, out_path)


# ===========================================================================
# Shared GeoTIFF writer
# ===========================================================================

def _write_tiff(grid: np.ndarray, transform, out_path: Path) -> None:
    """Write a 2-D float32 numpy array to a single-band GeoTIFF."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path, "w",
        driver="GTiff",
        height=grid.shape[0],
        width=grid.shape[1],
        count=1,
        dtype=np.float32,
        crs=CRS_TARGET,
        transform=transform,
        nodata=NODATA_VALUE,
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(grid, 1)
    log.info("    Saved  -> %s", out_path.name)


# ===========================================================================
# Entry point
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interpolate BMC till geochemistry GPKGs into GeoTIFF rasters."
    )
    p.add_argument("--method", choices=["idw", "kriging"], default="idw",
                   help="Interpolation method (default: idw)")
    p.add_argument("--element", default=None,
                   help="Process only this element symbol, e.g. Cu")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing output rasters")
    p.add_argument("--idw-power", type=float, default=IDW_POWER_DEFAULT,
                   metavar="P", help=f"IDW distance power (default: {IDW_POWER_DEFAULT})")
    p.add_argument("--idw-neighbours", type=int, default=IDW_NEIGHBOURS_DEFAULT,
                   metavar="K", help=f"IDW nearest neighbours (default: {IDW_NEIGHBOURS_DEFAULT})")
    p.add_argument("--kriging-variogram", default="spherical",
                   choices=["linear", "power", "gaussian", "spherical", "exponential", "hole-effect"],
                   help="pykrige variogram model (default: spherical)")
    p.add_argument("--kriging-max-points", type=int, default=500, metavar="N",
                   help="Max sample size for Kriging variogram fit (default: 500)")
    p.add_argument("--resolution", type=float, default=RESOLUTION,
                   help=f"Output raster resolution in metres (default: {RESOLUTION})")
    p.add_argument("--snap-to", default=None, metavar="RASTER",
                   help="Path to a reference GeoTIFF; output grid will match its extent.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.element:
        key = f"bmc_{args.element.capitalize()}"
        if key not in ELEMENT_MAP:
            log.error("Unknown element '%s'. Known: %s", args.element, list(ELEMENT_MAP))
            sys.exit(1)
        work_list = [(key, ELEMENT_MAP[key])]
    else:
        work_list = list(ELEMENT_MAP.items())

    log.info("=" * 60)
    log.info(" BMC Geochemistry Interpolation")
    log.info("   Method    : %s", args.method.upper())
    log.info("   Resolution: %.0f m", args.resolution)
    log.info("   Elements  : %d", len(work_list))
    log.info("   Output dir: %s", OUT_DIR)
    log.info("=" * 60)

    # Resolve snap bounds from reference raster (if provided)
    snap_bounds = None
    if args.snap_to:
        import rasterio as _rio
        with _rio.open(args.snap_to) as _ref:
            b = _ref.bounds
            snap_bounds = (b.left, b.bottom, b.right, b.top)
        log.info("Snap bounds from %s: %s", args.snap_to, snap_bounds)
    else:
        # Auto-detect: derive extent from all label GPKGs so IDW covers
        # every label point (IDW extrapolates gracefully outside data hull).
        import geopandas as _gpd
        _labels_dir = REPO_ROOT / "data" / "raw" / "labels"
        _label_gpkgs = list(_labels_dir.glob("*.gpkg"))
        if _label_gpkgs:
            _bounds = [_gpd.read_file(str(p), engine="pyogrio")
                       .to_crs(CRS_TARGET).total_bounds
                       for p in _label_gpkgs]
            _buf = 5_000  # 5 km buffer around label extent
            _xmin = min(b[0] for b in _bounds) - _buf
            _ymin = min(b[1] for b in _bounds) - _buf
            _xmax = max(b[2] for b in _bounds) + _buf
            _ymax = max(b[3] for b in _bounds) + _buf
            snap_bounds = (_xmin, _ymin, _xmax, _ymax)
            log.info("Auto-snapped extent to label union + 5 km buffer: "
                     "%.0f %.0f %.0f %.0f", *snap_bounds)
        else:
            log.warning("No label GPKGs found — using point-cloud extent.")

    skipped = processed = failed = 0

    for stem, out_stem in work_list:
        gpkg_path = GEOCHEM_DIR / f"{stem}.gpkg"
        out_path  = OUT_DIR / f"{out_stem}_{args.method}.tif"

        log.info("")
        log.info("-- %s  ->  %s", gpkg_path.name, out_path.name)

        if not gpkg_path.exists():
            log.warning("   GPKG not found: %s -- skipping.", gpkg_path)
            failed += 1
            continue

        if out_path.exists() and not args.overwrite:
            log.info("   Already exists -- skipping (use --overwrite to force).")
            skipped += 1
            continue

        gdf = gpd.read_file(gpkg_path, engine="pyogrio")
        if gdf.crs is None:
            gdf = gdf.set_crs(CRS_TARGET)
        elif str(gdf.crs) != CRS_TARGET:
            gdf = gdf.to_crs(CRS_TARGET)

        if "VALUE" not in gdf.columns:
            log.error("   No VALUE column in %s -- skipping.", gpkg_path.name)
            failed += 1
            continue

        try:
            if args.method == "idw":
                rasterise_idw(gdf, out_path,
                              resolution=args.resolution,
                              power=args.idw_power,
                              k=args.idw_neighbours,
                              snap_bounds=snap_bounds)
            else:
                rasterise_kriging(gdf, out_path,
                                  resolution=args.resolution,
                                  variogram_model=args.kriging_variogram,
                                  max_points=args.kriging_max_points)
            processed += 1
        except Exception as exc:
            log.exception("   ERROR processing %s: %s", stem, exc)
            failed += 1

    log.info("")
    log.info("=" * 60)
    log.info(" Done: %d processed  |  %d skipped  |  %d failed",
             processed, skipped, failed)
    if processed:
        log.info(" Run next: python pipeline/02_preprocessing/extract_features.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
