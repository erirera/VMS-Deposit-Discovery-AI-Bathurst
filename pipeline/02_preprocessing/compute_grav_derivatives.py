"""
compute_grav_derivatives.py
---------------------------
Computes seven standard geophysical derivative grids from any Bouguer gravity
raster for the Bathurst Mining Camp.

Derivatives computed
--------------------
1. HGM   - Horizontal Gradient Magnitude    : delineates density contacts / faults
2. TDR   - Tilt Derivative                  : self-normalising edge detector
3. FVD   - First Vertical Derivative        : enhances shallow anomalies (FFT)
4. SVD   - Second Vertical Derivative       : further suppresses deep/regional signal
5. AS    - Analytic Signal Amplitude        : sqrt(HGM^2 + FVD^2), always positive
6. UC500 - Upward Continued (500 m)         : regional field, strips shallow noise
7. RES   - Residual Bouguer                 : Bouguer - UC500, local anomalies only

All outputs are written to:
  data/processed/grav_derivatives/<source_stem>/

Usage
-----
  python pipeline/02_preprocessing/compute_grav_derivatives.py
  python pipeline/02_preprocessing/compute_grav_derivatives.py --source gra_ggr_bmc_combined.tif
  python pipeline/02_preprocessing/compute_grav_derivatives.py --source /abs/path/to/file.tif
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -- Pipeline config -----------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    RASTERS_DIR,
    PROCESSED_DIR,
    CRS_TARGET,
    NODATA_VALUE,
)

# -- Module-level constants ----------------------------------------------------
DEFAULT_SOURCE               = "gra_ggr_bmc_combined.tif"
GRAV_DERIVATIVE_RESOLUTION_M = 50        # target pixel size (metres)
UPWARD_CONTINUATION_HEIGHT_M = 500       # upward continuation height (metres)
GRAV_DERIVATIVES_DIR         = PROCESSED_DIR / "grav_derivatives"

# Ordered dict: output stem -> display title
DERIVATIVES = {
    "gra_ggr_hgm_bmc":   "Horizontal Gradient Magnitude (HGM)",
    "gra_ggr_tdr_bmc":   "Tilt Derivative (TDR)",
    "gra_ggr_fvd_bmc":   "First Vertical Derivative (FVD)",
    "gra_ggr_as_bmc":    "Analytic Signal Amplitude (AS)",
    "gra_ggr_uc500_bmc": f"Upward Continued {UPWARD_CONTINUATION_HEIGHT_M} m (UC500)",
    "gra_ggr_res_bmc":   f"Residual Bouguer (Bouguer - UC{UPWARD_CONTINUATION_HEIGHT_M})",
}

# -- Logging -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# -- Step 1 - Reprojection -----------------------------------------------------

def reproject_and_save_array(
    arr: np.ndarray,
    src_profile: dict,
    src_bounds,
    dst_path: Path,
    dst_resolution: float,
) -> np.ndarray:
    """
    Reproject a native-grid array to target CRS (EPSG:2953) at dst_resolution,
    save it as a single-band GeoTIFF, and return the reprojected array.
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    src_height, src_width = arr.shape

    # Calculate transform for target CRS and resolution
    transform, width, height = calculate_default_transform(
        src_profile["crs"],
        CRS_TARGET,
        src_width,
        src_height,
        *src_bounds,
        resolution=dst_resolution,
    )

    profile = src_profile.copy()
    profile.update({
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
        "blockysize": 256,
        "count": 1,
    })

    # Prepare array: fill NaNs with nodata for projection
    src_filled = np.where(np.isnan(arr), NODATA_VALUE, arr).astype(np.float32)
    dst_arr = np.empty((height, width), dtype=np.float32)

    reproject(
        source=src_filled,
        destination=dst_arr,
        src_transform=src_profile["transform"],
        src_crs=src_profile["crs"],
        dst_transform=transform,
        dst_crs=CRS_TARGET,
        resampling=Resampling.bilinear,
        src_nodata=NODATA_VALUE,
        dst_nodata=NODATA_VALUE,
    )

    # Write to file
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(dst_arr, 1)

    # Return array with NaNs restored
    dst_arr_nan = np.where(dst_arr == NODATA_VALUE, np.nan, dst_arr).astype(np.float64)
    return dst_arr_nan


def _fill_nan(grid: np.ndarray) -> np.ndarray:
    """Replace NaNs with the grid mean for FFT stability."""
    filled = grid.copy()
    filled[np.isnan(grid)] = np.nanmean(grid)
    return filled


def _wavenumbers(ny: int, nx: int, dx: float):
    """Return 2-D wavenumber grids (kx, ky) and radial |k| in rad/metre."""
    ky = np.fft.fftfreq(ny, d=dx) * 2 * np.pi
    kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    return KX, KY, K


def _vertical_derivative(G_fft: np.ndarray, K: np.ndarray, order: int = 1) -> np.ndarray:
    """FFT-domain vertical derivative of order `order`.
    Multiplies the spectrum by |k|^order -- industry standard."""
    filtered = G_fft * (K ** order)
    return np.real(np.fft.ifft2(filtered))


def _upward_continue(G_fft: np.ndarray, K: np.ndarray, height_m: float) -> np.ndarray:
    """FFT-domain upward continuation: applies exp(-|k|*h) filter."""
    uc_filter = np.exp(-K * height_m)
    return np.real(np.fft.ifft2(G_fft * uc_filter))


# -- Step 3 - Derivative computation -------------------------------------------

def compute_all_derivatives(grid: np.ndarray, dx: float) -> dict:
    """Compute all 7 gravity derivatives. Returns {stem: 2-D array}."""
    ny, nx = grid.shape
    filled = _fill_nan(grid)
    mask   = np.isnan(grid)

    KX, KY, K = _wavenumbers(ny, nx, dx)
    G_fft = np.fft.fft2(filled)

    # FVD
    log.info("  Computing FVD  (FFT order-1 vertical derivative) ...")
    fvd = _vertical_derivative(G_fft, K, order=1)

    # HGM
    log.info("  Computing HGM  (horizontal gradient magnitude) ...")
    dg_dx = np.gradient(grid, dx, axis=1)
    dg_dy = np.gradient(grid, dx, axis=0)
    hgm   = np.sqrt(dg_dx**2 + dg_dy**2)

    # AS
    log.info("  Computing AS   (analytic signal amplitude) ...")
    as_   = np.sqrt(hgm**2 + fvd**2)

    # TDR
    log.info("  Computing TDR  (tilt derivative) ...")
    eps = 1e-10
    tdr = np.arctan2(fvd, hgm + eps)

    # UC500
    log.info(f"  Computing UC{UPWARD_CONTINUATION_HEIGHT_M}  (upward continuation {UPWARD_CONTINUATION_HEIGHT_M} m) ...")
    uc500 = _upward_continue(G_fft, K, height_m=UPWARD_CONTINUATION_HEIGHT_M)

    # Residual
    log.info("  Computing RES  (residual Bouguer = Bouguer - UC500) ...")
    res = grid - uc500

    def _mask(arr):
        out = arr.copy()
        out[mask] = np.nan
        return out

    return {
        "gra_ggr_hgm_bmc":   _mask(hgm),
        "gra_ggr_tdr_bmc":   _mask(tdr),
        "gra_ggr_fvd_bmc":   _mask(fvd),
        "gra_ggr_as_bmc":    _mask(as_),
        "gra_ggr_uc500_bmc": _mask(uc500),
        "gra_ggr_res_bmc":   _mask(res),
    }


# -- Step 4 - GeoTIFF writer ---------------------------------------------------

def save_geotiff(arr: np.ndarray, dst_path: Path, ref_profile: dict) -> None:
    """Write a float32 2-D array as a single-band GeoTIFF."""
    profile = ref_profile.copy()
    profile.update({
        "count":      1,
        "dtype":      "float32",
        "driver":     "GTiff",
        "compress":   "lzw",
        "tiled":      True,
        "blockxsize": 256,
        "blockysize": 256,
        "nodata":     NODATA_VALUE,
    })
    out = np.where(np.isnan(arr), NODATA_VALUE, arr).astype(np.float32)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(out, 1)


# -- Step 5 - QC PNG export ----------------------------------------------------

def _pct_clip(arr: np.ndarray, lo: float = 2.0, hi: float = 98.0):
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return 0, 1
    return float(np.percentile(valid, lo)), float(np.percentile(valid, hi))


def _units_label(stem: str) -> str:
    if "fvd" in stem or "hgm" in stem or "as" in stem:
        return "mGal / m"
    if "tdr" in stem:
        return "rad"
    return "mGal"


def save_qc_png(
    arr: np.ndarray,
    name: str,
    title: str,
    dst_path: Path,
    source_label: str = "",
    res_m: float = 50.0,
) -> None:
    """Export a quick-look PNG of a gravity derivative grid."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    signed = any(tag in name for tag in ("fvd", "tdr", "res"))
    cmap   = "RdBu_r" if signed else "viridis"

    vmin, vmax = _pct_clip(arr)
    if signed:
        extreme = max(abs(vmin), abs(vmax))
        vmin, vmax = -extreme, extreme

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    im = ax.imshow(
        arr, cmap=cmap, vmin=vmin, vmax=vmax,
        interpolation="bilinear", aspect="equal",
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label(_units_label(name), fontsize=9)

    ax.set_title(
        f"Bathurst Mining Camp -- {title}\n"
        f"source: {source_label}  |  EPSG:2953  |  {res_m:.0f} m",
        fontsize=11, fontweight="bold", pad=10,
    )
    ax.set_xlabel(f"Column ({res_m:.0f} m pixels)", fontsize=8)
    ax.set_ylabel(f"Row ({res_m:.0f} m pixels)", fontsize=8)
    ax.tick_params(labelsize=7)

    valid = arr[~np.isnan(arr)]
    stats_txt = (
        f"min={float(valid.min()):.4g}  "
        f"max={float(valid.max()):.4g}  "
        f"mean={float(valid.mean()):.4g}  "
        f"std={float(valid.std()):.4g}"
    )
    fig.text(0.5, 0.01, stats_txt, ha="center", fontsize=7, color="grey")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(dst_path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"    QC PNG -> {dst_path.name}")


# -- CLI + Main ----------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute 7 gravity derivatives from a Bouguer gravity raster."
    )
    p.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help=(
            "Source raster filename (looked up in data/raw/rasters/) "
            "or an absolute path. Default: %(default)s"
        ),
    )
    return p.parse_args()


def resolve_source(source_arg: str) -> Path:
    """Return an absolute Path to the source raster."""
    p = Path(source_arg)
    if p.is_absolute() and p.exists():
        return p
    candidate = RASTERS_DIR / p.name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Source raster not found.\n"
        f"  Tried: {p}\n"
        f"  Tried: {candidate}\n"
        f"  Place the file in data/raw/rasters/ or pass a full path."
    )


def main():
    args          = parse_args()
    source_raster = resolve_source(args.source)
    source_stem   = source_raster.stem      # e.g. gra_ggr_bmc_combined

    out_dir    = GRAV_DERIVATIVES_DIR / source_stem
    reproj_tif = out_dir / f"{source_stem}_epsg2953_{GRAV_DERIVATIVE_RESOLUTION_M}m.tif"
    qc_dir     = out_dir / "qc_plots"

    log.info("=== Gravity Derivative Computation -- BMC ===")
    log.info(f"Source   : {source_raster}")
    log.info(f"Output   : {out_dir}")
    log.info(f"Res      : {GRAV_DERIVATIVE_RESOLUTION_M} m (final reprojected)")
    log.info(f"UC height: {UPWARD_CONTINUATION_HEIGHT_M} m")

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load Raw Native Grid ------------------------------------------
    log.info("\n[1/3] Loading raw native grid ...")
    with rasterio.open(source_raster) as src:
        src_profile = src.profile.copy()
        src_bounds = src.bounds
        raw    = src.read(1).astype(np.float64)
        nodata = src.nodata if src.nodata is not None else NODATA_VALUE
        dx     = src.res[0]

    grid    = np.where(raw == nodata, np.nan, raw).astype(np.float64)
    n_valid = int(np.sum(~np.isnan(grid)))
    log.info(f"  Grid shape : {grid.shape[0]} x {grid.shape[1]}")
    log.info(f"  Pixel size : {dx:.2f} m (native)")
    log.info(f"  Valid cells: {n_valid:,} / {grid.size:,}")

    # ── Step 2: Compute on Native Grid ----------------------------------------
    log.info("\n[2/3] Computing derivatives on native grid ...")
    deriv_grids = compute_all_derivatives(grid, dx)

    # ── Step 3: Reproject and Save --------------------------------------------
    log.info("\n[3/3] Reprojecting and saving to EPSG:2953 at 50 m ...")
    
    # Save reprojected raw grid
    log.info(f"  Reprojecting raw grid to {reproj_tif.name} ...")
    reproj_raw = reproject_and_save_array(
        grid, src_profile, src_bounds,
        reproj_tif, GRAV_DERIVATIVE_RESOLUTION_M
    )
    # Also save raw grid to rasters_reprojected
    raw_reproj_path = PROCESSED_DIR / "rasters_reprojected" / f"{source_stem}_epsg2953_{GRAV_DERIVATIVE_RESOLUTION_M}m.tif"
    reproject_and_save_array(
        grid, src_profile, src_bounds,
        raw_reproj_path, GRAV_DERIVATIVE_RESOLUTION_M
    )

    log.info("\n--- Writing GeoTIFFs + QC PNGs ---")
    summary_rows = []

    for stem, title in tqdm(DERIVATIVES.items(), desc="Writing outputs"):
        native_arr = deriv_grids[stem]
        tif_path = out_dir / f"{stem}.tif"
        png_path = qc_dir  / f"{stem}_qc.png"

        # Reproject and save to out_dir
        reproj_arr = reproject_and_save_array(
            native_arr, src_profile, src_bounds,
            tif_path, GRAV_DERIVATIVE_RESOLUTION_M
        )

        # Also save to rasters_reprojected (without epsg suffix for derivatives)
        reproject_and_save_array(
            native_arr, src_profile, src_bounds,
            PROCESSED_DIR / "rasters_reprojected" / f"{stem}.tif",
            GRAV_DERIVATIVE_RESOLUTION_M
        )

        # Save QC PNG using the reprojected 50m array
        save_qc_png(reproj_arr, stem, title, png_path, source_label=source_stem, res_m=GRAV_DERIVATIVE_RESOLUTION_M)

        valid = reproj_arr[~np.isnan(reproj_arr)]
        summary_rows.append({
            "derivative": stem,
            "min":     float(valid.min()) if len(valid) > 0 else np.nan,
            "max":     float(valid.max()) if len(valid) > 0 else np.nan,
            "mean":    float(valid.mean()) if len(valid) > 0 else np.nan,
            "std":     float(valid.std()) if len(valid) > 0 else np.nan,
        })
        log.info(f"  OK {stem}.tif")

    # Summary
    log.info("\n--- Summary Statistics (50m Reprojected) ---")
    log.info(f"{'Derivative':<24} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12}")
    log.info("-" * 76)
    for r in summary_rows:
        log.info(
            f"  {r['derivative']:<22} "
            f"{r['min']:>12.4g} "
            f"{r['max']:>12.4g} "
            f"{r['mean']:>12.4g} "
            f"{r['std']:>12.4g}"
        )

    log.info(f"\nAll derivatives written  : {out_dir}")
    log.info(f"QC plots                 : {qc_dir}")
    log.info("Run next: python pipeline/02_preprocessing/extract_features.py")


if __name__ == "__main__":
    main()
