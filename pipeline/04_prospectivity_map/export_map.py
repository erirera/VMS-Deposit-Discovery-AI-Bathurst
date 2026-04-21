"""
export_map.py
──────────────
Exports the prospectivity GeoTIFF to publication-ready formats:
  • PNG figure with colorbar, BMC boundary, and deposit locations overlaid
  • Optionally a PDF-quality high-resolution version

This PNG is intended for manuscript submission and the GitHub README.

Usage:
    python pipeline/04_prospectivity_map/export_map.py
"""

import sys
import logging
from pathlib import Path
import numpy as np
import rasterio
from rasterio.plot import show
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import contextily as ctx

PIPELINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_DIR))
from config import (
    OUTPUTS_DIR, PROSPECTIVITY_TIFF,
    VMS_LABELS_GPKG, CRS_TARGET
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# Custom prospectivity colormap: blue (low) → green → gold → red (high)
PROSPECT_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "prospectivity",
    ["#1e3a5f", "#1e6b8c", "#10b981", "#f59e0b", "#dc2626"],
    N=256
)


def load_prospectivity_raster():
    """Load the GeoTIFF and mask nodata."""
    if not PROSPECTIVITY_TIFF.exists():
        raise FileNotFoundError(
            f"Prospectivity raster not found: {PROSPECTIVITY_TIFF}\n"
            "Run: python pipeline/04_prospectivity_map/predict_full_extent.py"
        )
    with rasterio.open(PROSPECTIVITY_TIFF) as src:
        data    = src.read(1, masked=True)
        transform = src.transform
        bounds  = src.bounds
        crs     = src.crs
    log.info(f"  Loaded raster: {data.shape}  CRS={crs}")
    return data, transform, bounds


def load_vms_deposits():
    """Load known VMS deposit locations."""
    if not VMS_LABELS_GPKG.exists():
        log.warning("VMS labels not found — deposit overlay skipped")
        return None
    gdf = gpd.read_file(VMS_LABELS_GPKG, layer="vms_deposits")
    return gdf.to_crs(CRS_TARGET) if gdf.crs.to_string() != CRS_TARGET else gdf


def export_png(data, transform, bounds, deposits):
    """Publication-ready PNG figure."""
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0d1117")

    # Raster
    img = ax.imshow(
        data,
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        origin="upper",
        cmap=PROSPECT_CMAP,
        vmin=0, vmax=1,
        alpha=0.85
    )

    # Colorbar
    cbar = fig.colorbar(img, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("VMS Mineralisation Probability", color="#cbd5e1", fontsize=11)
    cbar.ax.yaxis.set_tick_params(color="#94a3b8")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#94a3b8")

    # Deposit overlay
    if deposits is not None:
        ax.scatter(
            deposits.geometry.x, deposits.geometry.y,
            c="#f59e0b", edgecolors="#ffffff", linewidths=0.8,
            s=80, zorder=5, label="Known VMS Deposits (n=45)"
        )

    # Labels & formatting
    ax.set_title(
        "AI Prospectivity Map — Bathurst Mining Camp, New Brunswick\n"
        "Random Forest / XGBoost Ensemble · NRCan Geophysics + NB Till Geochemistry",
        color="#f8fafc", fontsize=13, fontweight="bold", pad=15
    )
    ax.set_xlabel("Easting (m) — EPSG:2953", color="#94a3b8")
    ax.set_ylabel("Northing (m) — EPSG:2953", color="#94a3b8")
    ax.tick_params(colors="#94a3b8")

    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

    legend_elements = [
        Patch(facecolor="#1e3a5f", label="Low probability"),
        Patch(facecolor="#10b981", label="Moderate probability"),
        Patch(facecolor="#dc2626", label="High probability (>0.7)"),
        plt.scatter([], [], c="#f59e0b", edgecolors="w", s=60,
                    label="Known VMS deposit"),
    ]
    ax.legend(
        handles=legend_elements, loc="lower left",
        framealpha=0.15, labelcolor="#cbd5e1", fontsize=9
    )

    # Attribution
    fig.text(
        0.99, 0.01,
        "Falebita (2026) · CC0-1.0 · Data: NRCan + NB GSB",
        ha="right", va="bottom", color="#475569", fontsize=8
    )

    fig.tight_layout()
    out_png = OUTPUTS_DIR / "bmc_prospectivity_map.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="#0f172a")
    log.info(f"  ✅ PNG saved → {out_png}")

    # High-res PDF for manuscript
    out_pdf = OUTPUTS_DIR / "bmc_prospectivity_map_hires.pdf"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="#0f172a")
    log.info(f"  ✅ PDF saved → {out_pdf}")
    plt.close(fig)


def main():
    log.info("═══ Prospectivity Map Export ═══")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    data, transform, bounds = load_prospectivity_raster()
    deposits = load_vms_deposits()

    log.info("\n[Exporting publication-ready map ...]")
    export_png(data, transform, bounds, deposits)

    # Print summary statistics
    valid = data.compressed()
    log.info("\n─── Prospectivity Statistics ───")
    log.info(f"  Median probability   : {np.median(valid):.4f}")
    log.info(f"  Mean probability     : {np.mean(valid):.4f}")
    log.info(f"  >0.5 (moderate+)     : {(valid > 0.5).sum():,} cells  ({100*(valid>0.5).mean():.1f}%)")
    log.info(f"  >0.7 (high priority) : {(valid > 0.7).sum():,} cells  ({100*(valid>0.7).mean():.1f}%)")
    log.info(f"  >0.9 (very high)     : {(valid > 0.9).sum():,} cells  ({100*(valid>0.9).mean():.1f}%)")

    log.info(f"\n✅ Map export complete. Check outputs/ directory.")


if __name__ == "__main__":
    main()
