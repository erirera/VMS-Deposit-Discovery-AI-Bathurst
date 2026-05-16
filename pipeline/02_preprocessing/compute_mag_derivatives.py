"""
compute_mag_derivatives.py
──────────────────────────
Computes six standard geophysical derivative grids from the compiled
Residual Magnetic Intensity (RMI) raster for the Bathurst Mining Camp.

Source raster  : data/raw/rasters/mag_rmi_bmc_compiled.tif
Output rasters : data/processed/mag_derivatives/  (one GeoTIFF per derivative)
QC figures     : data/processed/mag_derivatives/qc_plots/  (one PNG per derivative)

Derivatives computed
────────────────────
  1. FVD   — First Vertical Derivative       (FFT wavenumber domain)
  2. THG   — Total Horizontal Gradient       (numpy.gradient on 2-D grid)
  3. AS    — Analytic Signal Amplitude       = sqrt(THG² + FVD²)
  4. TDR   — Tilt Derivative                 = arctan(FVD / THG)
  5. THDR  — Tilt Horizontal Gradient        = horizontal gradient of TDR
  6. SVD   — Second Vertical Derivative      (FFT wavenumber domain)

All outputs share the same CRS, transform, and resolution as the source
(after reprojection to EPSG:2953 at 100 m).

Usage
─────
    python pipeline/02_preprocessing/compute_mag_derivatives.py

Dependencies
────────────
    rasterio, numpy, scipy (for FFT padding), matplotlib, tqdm
    All available in the standard pipeline environment.
"""

import sys
import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import matplotlib
matplotlib.use("Agg")          # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm

# ── Path bootstrap ────────────────────────────────────────────────────────────
PIPELINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_DIR))
from config import (
    RASTERS_DIR, PROCESSED_DIR, MAG_DERIVATIVES_DIR,
    CRS_TARGET, MAG_DERIVATIVE_RESOLUTION_M, NODATA_VALUE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
SOURCE_RASTER  = RASTERS_DIR / "mag_rmi_bmc_compiled.tif"
REPROJ_RASTER  = MAG_DERIVATIVES_DIR / "mag_rmi_bmc_epsg2953_50m.tif"
QC_DIR         = MAG_DERIVATIVES_DIR / "qc_plots"
EPSILON        = 1e-10          # Guard against divide-by-zero

DERIVATIVES = {
    "mag_rmi_fvd_bmc":  "First Vertical Derivative (FVD)",
    "mag_rmi_thg_bmc":  "Total Horizontal Gradient (THG)",
    "mag_rmi_as_bmc":   "Analytic Signal Amplitude (AS)",
    "mag_rmi_tdr_bmc":  "Tilt Derivative (TDR)",
    "mag_rmi_thdr_bmc": "Tilt Horizontal Gradient (THDR)",
    "mag_rmi_svd_bmc":  "Second Vertical Derivative (SVD)",
}


# ── Step 1 — Reprojection ─────────────────────────────────────────────────────

def reproject_to_target(src_path: Path, dst_path: Path) -> None:
    """Reproject source raster to EPSG:2953 at TARGET_RESOLUTION_M."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src_path) as src:
        log.info(f"  Source CRS       : {src.crs}")
        log.info(f"  Source shape     : {src.height} × {src.width}")
        transform, width, height = calculate_default_transform(
            src.crs, CRS_TARGET,
            src.width, src.height, *src.bounds,
            resolution=MAG_DERIVATIVE_RESOLUTION_M,
        )
        meta = src.meta.copy()
        meta.update({
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
        with rasterio.open(dst_path, "w", **meta) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=CRS_TARGET,
                resampling=Resampling.bilinear,
            )
    log.info(f"  Output shape     : {height} × {width}")
    log.info(f"  ✅ Reprojected → {dst_path.name}")


# ── Step 2 — FFT-based vertical derivative ────────────────────────────────────

def _pad_to_power_of_two(arr: np.ndarray):
    """
    Zero-pad a 2-D array so both dimensions are the next power of two.
    Returns padded array and original shape for later trimming.
    """
    orig_shape = arr.shape
    new_rows = 1 << (arr.shape[0] - 1).bit_length()
    new_cols = 1 << (arr.shape[1] - 1).bit_length()
    padded = np.zeros((new_rows, new_cols), dtype=np.float64)
    padded[: arr.shape[0], : arr.shape[1]] = arr
    return padded, orig_shape


def fft_vertical_derivative(grid: np.ndarray, dx: float, order: int = 1) -> np.ndarray:
    """
    Compute the n-th order vertical derivative of a potential field grid
    using the wavenumber-domain method.

    In the frequency domain, the n-th vertical derivative corresponds to
    multiplying the Fourier transform by (2π |k|)^n, where |k| is the
    radial wavenumber (cycles per unit distance).

    Parameters
    ----------
    grid  : 2-D numpy array — the field values (NaNs replaced by mean)
    dx    : cell size in metres (assumed square pixels)
    order : derivative order (1 = FVD, 2 = SVD)

    Returns
    -------
    2-D numpy array — same shape as input
    """
    fill_val = np.nanmean(grid)
    g = np.where(np.isnan(grid), fill_val, grid)

    padded, orig_shape = _pad_to_power_of_two(g)
    rows, cols = padded.shape

    # Build radial wavenumber array (cycles/m → rad/m)
    kx = np.fft.fftfreq(cols, d=dx) * 2 * np.pi   # rad/m
    ky = np.fft.fftfreq(rows, d=dx) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)                     # radial wavenumber |k|

    # FFT → multiply → IFFT
    F = np.fft.fft2(padded)
    F_deriv = F * (K ** order)
    result = np.real(np.fft.ifft2(F_deriv))

    # Trim back to original shape, restore NaN mask
    result = result[: orig_shape[0], : orig_shape[1]]
    result[np.isnan(grid)] = np.nan
    return result.astype(np.float32)


# ── Step 3 — Horizontal derivative helpers ────────────────────────────────────

def horizontal_gradient(grid: np.ndarray, dx: float):
    """
    Compute (∂B/∂x, ∂B/∂y) using central finite differences (numpy.gradient).
    Returns two arrays: (dBdx, dBdy), both float32.
    """
    # numpy.gradient returns [row_gradient, col_gradient]
    # row direction = y, col direction = x
    dBdy, dBdx = np.gradient(np.where(np.isnan(grid), 0.0, grid), dx)
    mask = np.isnan(grid)
    dBdx[mask] = np.nan
    dBdy[mask] = np.nan
    return dBdx.astype(np.float32), dBdy.astype(np.float32)


# ── Step 4 — All derivatives ──────────────────────────────────────────────────

def compute_all_derivatives(
    grid: np.ndarray, dx: float
) -> dict[str, np.ndarray]:
    """
    Given a 2-D magnetic field grid and cell size (metres), return a dict
    of all six derivative grids keyed by output filename stem.
    """
    log.info("  Computing FVD  (FFT order-1 vertical derivative) ...")
    fvd = fft_vertical_derivative(grid, dx, order=1)

    log.info("  Computing SVD  (FFT order-2 vertical derivative) ...")
    svd = fft_vertical_derivative(grid, dx, order=2)

    log.info("  Computing THG  (total horizontal gradient) ...")
    dBdx, dBdy = horizontal_gradient(grid, dx)
    thg = np.sqrt(dBdx**2 + dBdy**2 + EPSILON).astype(np.float32)

    log.info("  Computing AS   (analytic signal amplitude) ...")
    as_ = np.sqrt(thg**2 + fvd**2 + EPSILON).astype(np.float32)

    log.info("  Computing TDR  (tilt derivative) ...")
    tdr = np.arctan2(fvd, thg + EPSILON).astype(np.float32)

    log.info("  Computing THDR (tilt horizontal gradient) ...")
    tdr_dBdx, tdr_dBdy = horizontal_gradient(tdr, dx)
    thdr = np.sqrt(tdr_dBdx**2 + tdr_dBdy**2 + EPSILON).astype(np.float32)

    return {
        "mag_rmi_fvd_bmc":  fvd,
        "mag_rmi_thg_bmc":  thg,
        "mag_rmi_as_bmc":   as_,
        "mag_rmi_tdr_bmc":  tdr,
        "mag_rmi_thdr_bmc": thdr,
        "mag_rmi_svd_bmc":  svd,
    }


# ── Step 5 — GeoTIFF writer ───────────────────────────────────────────────────

def save_geotiff(
    arr: np.ndarray,
    dst_path: Path,
    ref_profile: dict,
) -> None:
    """Write a float32 2-D array as a single-band GeoTIFF."""
    profile = ref_profile.copy()
    profile.update({
        "count": 1,
        "dtype": "float32",
        "driver": "GTiff",
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "nodata": NODATA_VALUE,
    })
    out = np.where(np.isnan(arr), NODATA_VALUE, arr).astype(np.float32)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(out, 1)


# ── Step 6 — QC PNG export ────────────────────────────────────────────────────

def _pct_clip(arr: np.ndarray, lo: float = 2.0, hi: float = 98.0):
    """Return vmin/vmax clipped to percentile range for display."""
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return 0, 1
    return float(np.percentile(valid, lo)), float(np.percentile(valid, hi))


def save_qc_png(
    arr: np.ndarray,
    name: str,
    title: str,
    dst_path: Path,
) -> None:
    """Export a quick-look PNG of a derivative grid."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # Choose diverging colourmap for signed derivatives, sequential for positive
    signed = name in {"mag_rmi_fvd_bmc", "mag_rmi_tdr_bmc", "mag_rmi_svd_bmc"}
    cmap = "RdBu_r" if signed else "magma"

    vmin, vmax = _pct_clip(arr)
    if signed:
        # Symmetric around zero
        extreme = max(abs(vmin), abs(vmax))
        vmin, vmax = -extreme, extreme

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    im = ax.imshow(
        arr,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
        aspect="equal",
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("nT / m" if "fvd" in name or "svd" in name else "nT/m" if "thg" in name or "as_" in name or "thdr" in name else "rad", fontsize=9)
    ax.set_title(f"Bathurst Mining Camp — {title}", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Column (100 m pixels)", fontsize=8)
    ax.set_ylabel("Row (100 m pixels)", fontsize=8)
    ax.tick_params(labelsize=7)

    # Stats annotation
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
    log.info(f"    QC PNG → {dst_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("═══ Magnetic Derivative Computation — BMC ═══")
    log.info(f"Source : {SOURCE_RASTER}")
    log.info(f"Output : {MAG_DERIVATIVES_DIR}")

    # ── Guard: source raster must exist ──────────────────────────────────────
    if not SOURCE_RASTER.exists():
        raise FileNotFoundError(
            f"\n✗ Source raster not found:\n  {SOURCE_RASTER}\n"
            "  Ensure mag_rmi_bmc_compiled.tif is placed in data/raw/rasters/"
        )

    # ── Step 1: Reproject ─────────────────────────────────────────────────────
    MAG_DERIVATIVES_DIR.mkdir(parents=True, exist_ok=True)

    if REPROJ_RASTER.exists():
        log.info(f"\n[1/3] Reprojected raster already exists — skipping.")
        log.info(f"      ({REPROJ_RASTER.name})")
    else:
        log.info(f"\n[1/3] Reprojecting to EPSG:2953 at {MAG_DERIVATIVE_RESOLUTION_M} m ...")
        reproject_to_target(SOURCE_RASTER, REPROJ_RASTER)

    # ── Step 2: Load reprojected grid ─────────────────────────────────────────
    log.info("\n[2/3] Loading reprojected grid ...")
    with rasterio.open(REPROJ_RASTER) as src:
        ref_profile = src.profile.copy()
        raw = src.read(1).astype(np.float64)
        nodata = src.nodata if src.nodata is not None else NODATA_VALUE
        dx = src.res[0]    # pixel size in metres (square pixels assumed)

    # Mask nodata
    grid = np.where(raw == nodata, np.nan, raw).astype(np.float64)
    n_valid = int(np.sum(~np.isnan(grid)))
    log.info(f"  Grid shape : {grid.shape[0]} × {grid.shape[1]}")
    log.info(f"  Pixel size : {dx:.1f} m")
    log.info(f"  Valid cells: {n_valid:,} / {grid.size:,}")

    # ── Step 3: Compute derivatives ───────────────────────────────────────────
    log.info("\n[3/3] Computing derivatives ...")
    deriv_grids = compute_all_derivatives(grid, dx)

    # ── Write outputs ─────────────────────────────────────────────────────────
    log.info("\n─── Writing GeoTIFFs + QC PNGs ───")
    summary_rows = []

    for stem, title in tqdm(DERIVATIVES.items(), desc="Writing outputs"):
        arr = deriv_grids[stem]
        tif_path = MAG_DERIVATIVES_DIR / f"{stem}.tif"
        png_path  = QC_DIR / f"{stem}_qc.png"

        save_geotiff(arr, tif_path, ref_profile)
        save_qc_png(arr, stem, title, png_path)

        valid = arr[~np.isnan(arr)]
        summary_rows.append({
            "derivative": stem,
            "min":   float(valid.min()),
            "max":   float(valid.max()),
            "mean":  float(valid.mean()),
            "std":   float(valid.std()),
            "n_valid": len(valid),
        })
        log.info(f"  ✅ {stem}.tif")

    # ── Summary table ─────────────────────────────────────────────────────────
    log.info("\n─── Summary Statistics ───")
    log.info(f"{'Derivative':<22} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12}")
    log.info("─" * 72)
    for r in summary_rows:
        log.info(
            f"  {r['derivative']:<20} "
            f"{r['min']:>12.4g} "
            f"{r['max']:>12.4g} "
            f"{r['mean']:>12.4g} "
            f"{r['std']:>12.4g}"
        )

    log.info(f"\n✅ All derivatives written → {MAG_DERIVATIVES_DIR}")
    log.info(f"   QC plots            → {QC_DIR}")
    log.info("   Run next: python pipeline/02_preprocessing/extract_features.py")


if __name__ == "__main__":
    main()
