"""
compute_rad_derivatives.py
--------------------------
Reprojects all radiometric rasters to EPSG:2953 at 50 m and computes
derived ratio bands for the Bathurst Mining Camp VMS pipeline.

Primary bands processed
-----------------------
  rad_k_bmc_combined.tif   -- Potassium (K%)
  rad_th_bmc_combined.tif  -- Thorium (Th ppm)
  rad_u_bmc_combined.tif   -- Uranium (U ppm)
  rad_thk_bmc_combined.tif -- Th/K ratio (if provided, else computed)
  rad_uk_bmc_combined.tif  -- U/K ratio  (if provided, else computed)

Derived ratios computed
-----------------------
  K/Th   -- Potassic alteration indicator (high K/Th = sericitic/potassic zones)
  U/Th   -- Uranium mobility index (high U/Th = hydrothermal leaching)
  Dose   -- Total dose rate proxy: K*0.313 + Th*0.0430 + U*0.277 (nGy/h)

All outputs written to:
  data/processed/rad_derivatives/<source_stem>/

Usage
-----
  python pipeline/02_preprocessing/compute_rad_derivatives.py
  python pipeline/02_preprocessing/compute_rad_derivatives.py --k rad_k_bmc_combined.tif
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
from config import RASTERS_DIR, PROCESSED_DIR, CRS_TARGET, NODATA_VALUE

# -- Constants -----------------------------------------------------------------
RAD_RESOLUTION_M   = 50
RAD_DERIVATIVES_DIR = PROCESSED_DIR / "rad_derivatives"

# Default source filenames (in data/raw/rasters/)
DEFAULTS = {
    "k":   "rad_k_bmc_combined.tif",
    "th":  "rad_th_bmc_combined.tif",
    "u":   "rad_u_bmc_combined.tif",
    "thk": "rad_thk_bmc_combined.tif",
    "uk":  "rad_uk_bmc_combined.tif",
}

# Dose rate conversion coefficients (IAEA, nGy/h per unit)
DOSE_K_COEF  = 0.313   # per % K
DOSE_TH_COEF = 0.0430  # per ppm Th
DOSE_U_COEF  = 0.277   # per ppm U

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# -- Helpers -------------------------------------------------------------------

def resolve(fname: str) -> Path:
    p = Path(fname)
    if p.is_absolute() and p.exists():
        return p
    c = RASTERS_DIR / p.name
    if c.exists():
        return c
    raise FileNotFoundError(f"Not found: {fname} (also tried {c})")


def reproject_raster(src_path: Path, dst_path: Path) -> None:
    """Reproject to CRS_TARGET at RAD_RESOLUTION_M."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    res = RAD_RESOLUTION_M
    with rasterio.open(src_path) as src:
        log.info(f"  {src_path.name}: {src.crs} {src.height}x{src.width} -> {CRS_TARGET} {res}m")
        transform, width, height = calculate_default_transform(
            src.crs, CRS_TARGET, src.width, src.height,
            *src.bounds, resolution=(res, res),
        )
        profile = src.profile.copy()
        profile.update({
            "crs": CRS_TARGET, "transform": transform,
            "width": width, "height": height,
            "nodata": NODATA_VALUE, "dtype": "float32",
            "compress": "lzw", "driver": "GTiff",
        })
        with rasterio.open(dst_path, "w", **profile) as dst:
            for b in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, b),
                    destination=rasterio.band(dst, b),
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=transform, dst_crs=CRS_TARGET,
                    resampling=Resampling.bilinear,
                )


def load_grid(tif_path: Path, nodata=None) -> tuple:
    """Load a reprojected TIF as a float64 NaN-masked array. Returns (grid, profile, dx)."""
    with rasterio.open(tif_path) as src:
        profile = src.profile.copy()
        raw = src.read(1).astype(np.float64)
        nd  = src.nodata if src.nodata is not None else (nodata or NODATA_VALUE)
        dx  = src.res[0]
    grid = np.where(raw == nd, np.nan, raw)
    return grid, profile, dx


def save_geotiff(arr: np.ndarray, dst_path: Path, ref_profile: dict) -> None:
    profile = ref_profile.copy()
    profile.update({
        "count": 1, "dtype": "float32", "driver": "GTiff",
        "compress": "lzw", "nodata": NODATA_VALUE,
    })
    out = np.where(np.isnan(arr), NODATA_VALUE, arr).astype(np.float32)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(out, 1)


def _pct_clip(arr, lo=2.0, hi=98.0):
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return 0, 1
    return float(np.percentile(valid, lo)), float(np.percentile(valid, hi))


def save_qc_png(arr, name, title, dst_path, source_label="", res_m=50.0):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    signed = any(t in name for t in ("ratio", "dose"))
    cmap   = "YlOrRd" if not signed else "RdBu_r"
    if "k_th" in name or "k_bmc" in name:
        cmap = "YlGn"
    elif "th_bmc" in name or "th_k" in name:
        cmap = "OrRd"
    elif "u_bmc" in name or "u_th" in name or "u_k" in name:
        cmap = "PuRd"
    elif "dose" in name:
        cmap = "hot"

    vmin, vmax = _pct_clip(arr)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="bilinear", aspect="equal")
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)

    units = {"k_bmc": "% K", "th_bmc": "ppm Th", "u_bmc": "ppm U",
             "th_k": "Th/K", "u_k": "U/K", "k_th": "K/Th",
             "u_th": "U/Th", "dose": "nGy/h"}
    unit_label = next((v for k, v in units.items() if k in name), "")
    cbar.set_label(unit_label, fontsize=9)

    ax.set_title(
        f"Bathurst Mining Camp -- {title}\n"
        f"source: {source_label}  |  EPSG:2953  |  {res_m:.0f} m",
        fontsize=11, fontweight="bold", pad=10,
    )
    ax.set_xlabel(f"Column ({res_m:.0f} m pixels)", fontsize=8)
    ax.set_ylabel(f"Row ({res_m:.0f} m pixels)", fontsize=8)
    ax.tick_params(labelsize=7)

    valid = arr[~np.isnan(arr)]
    stats = (f"min={float(valid.min()):.4g}  max={float(valid.max()):.4g}  "
             f"mean={float(valid.mean()):.4g}  std={float(valid.std()):.4g}")
    fig.text(0.5, 0.01, stats, ha="center", fontsize=7, color="grey")
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(dst_path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"    QC PNG -> {dst_path.name}")


# -- Main ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Reproject radiometric rasters and compute ratio bands."
    )
    for key, default in DEFAULTS.items():
        p.add_argument(f"--{key}", default=default,
                       help=f"Source file for {key.upper()} band (default: %(default)s)")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve paths
    sources = {}
    for key in ("k", "th", "u", "thk", "uk"):
        try:
            sources[key] = resolve(getattr(args, key))
            log.info(f"Found {key.upper()}: {sources[key].name}")
        except FileNotFoundError as e:
            if key in ("thk", "uk"):
                log.info(f"Optional {key.upper()} not found -- will compute from K/Th/U")
                sources[key] = None
            else:
                raise

    out_dir = RAD_DERIVATIVES_DIR / "rad_bmc_combined"
    qc_dir  = out_dir / "qc_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"\n=== Radiometric Processing -- BMC ===")
    log.info(f"Output : {out_dir}")
    log.info(f"Res    : {RAD_RESOLUTION_M} m")

    # -- Step 1: Reproject primary bands --------------------------------------
    log.info("\n[1/3] Reprojecting primary bands ...")
    reproj = {}
    for key in ("k", "th", "u", "thk", "uk"):
        if sources[key] is None:
            reproj[key] = None
            continue
        stem = sources[key].stem
        dst  = out_dir / f"{stem}_epsg2953_{RAD_RESOLUTION_M}m.tif"
        if dst.exists():
            log.info(f"  Skipping {stem} (already reprojected)")
        else:
            reproject_raster(sources[key], dst)
        reproj[key] = dst

    # -- Step 2: Load grids ---------------------------------------------------
    log.info("\n[2/3] Loading reprojected grids ...")
    grids = {}
    ref_profile, ref_dx = None, None
    for key in ("k", "th", "u", "thk", "uk"):
        if reproj[key] is None:
            grids[key] = None
            continue
        g, prof, dx = load_grid(reproj[key])
        grids[key] = g
        if ref_profile is None:
            ref_profile, ref_dx = prof, dx
        log.info(f"  {key.upper():4s}: shape={g.shape}, valid={int(np.sum(~np.isnan(g))):,}")

    # -- Step 3: Compute ratio bands ------------------------------------------
    log.info("\n[3/3] Computing derived ratio bands ...")
    eps = 1e-6

    k  = grids["k"]
    th = grids["th"]
    u  = grids["u"]

    # Th/K (use provided file or compute)
    if grids["thk"] is not None:
        th_k = grids["thk"]
        log.info("  Th/K: using provided rad_thk file")
    else:
        th_k = th / (k + eps)
        th_k[np.isnan(k) | np.isnan(th)] = np.nan
        log.info("  Th/K: computed from Th/K bands")

    # U/K (use provided file or compute)
    if grids["uk"] is not None:
        u_k = grids["uk"]
        log.info("  U/K : using provided rad_uk file")
    else:
        u_k = u / (k + eps)
        u_k[np.isnan(k) | np.isnan(u)] = np.nan
        log.info("  U/K : computed from U/K bands")

    # K/Th
    k_th = k / (th + eps)
    k_th[np.isnan(k) | np.isnan(th)] = np.nan
    log.info("  K/Th: computed")

    # U/Th
    u_th = u / (th + eps)
    u_th[np.isnan(u) | np.isnan(th)] = np.nan
    log.info("  U/Th: computed")

    # Total dose rate (IAEA coefficients)
    dose = DOSE_K_COEF * k + DOSE_TH_COEF * th + DOSE_U_COEF * u
    dose[np.isnan(k) | np.isnan(th) | np.isnan(u)] = np.nan
    log.info("  Dose rate: computed (K*0.313 + Th*0.043 + U*0.277)")

    # -- Write outputs --------------------------------------------------------
    outputs = {
        "rad_k_bmc":   (k,    "Potassium K (%)"),
        "rad_th_bmc":  (th,   "Thorium Th (ppm)"),
        "rad_u_bmc":   (u,    "Uranium U (ppm)"),
        "rad_th_k_bmc": (th_k, "Th/K Ratio (potassic alteration)"),
        "rad_u_k_bmc":  (u_k,  "U/K Ratio"),
        "rad_k_th_bmc": (k_th, "K/Th Ratio (K enrichment)"),
        "rad_u_th_bmc": (u_th, "U/Th Ratio (U mobility)"),
        "rad_dose_bmc": (dose, "Total Dose Rate (nGy/h)"),
    }

    log.info("\n--- Writing GeoTIFFs + QC PNGs ---")
    summary_rows = []

    for stem, (arr, title) in tqdm(outputs.items(), desc="Writing outputs"):
        if arr is None:
            continue
        tif_path = out_dir / f"{stem}.tif"
        png_path = qc_dir  / f"{stem}_qc.png"
        save_geotiff(arr, tif_path, ref_profile)
        save_qc_png(arr, stem, title, png_path,
                    source_label="rad_bmc_combined", res_m=ref_dx)
        valid = arr[~np.isnan(arr)]
        summary_rows.append({
            "band": stem, "min": float(valid.min()), "max": float(valid.max()),
            "mean": float(valid.mean()), "std": float(valid.std()),
        })
        log.info(f"  OK {stem}.tif")

    # Summary
    log.info("\n--- Summary Statistics ---")
    log.info(f"{'Band':<22} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
    log.info("-" * 66)
    for r in summary_rows:
        log.info(f"  {r['band']:<20} {r['min']:>10.4g} {r['max']:>10.4g} "
                 f"{r['mean']:>10.4g} {r['std']:>10.4g}")

    log.info(f"\nAll bands written : {out_dir}")
    log.info(f"QC plots          : {qc_dir}")
    log.info("Run next: python pipeline/02_preprocessing/extract_features.py")


if __name__ == "__main__":
    main()
