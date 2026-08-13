"""
success_rate_curve.py
─────────────────────
Computes and plots the Success Rate Curve (SRC) — the canonical evaluation
metric for Mineral Prospectivity Mapping (Agterberg & Bonham-Carter, 2005;
Carranza & Laborte, 2015; Parsa et al., 2023).

Definition
----------
Given a predicted prospectivity raster, rank ALL pixels from highest to lowest
probability. For each threshold t (expressed as the cumulative proportion of
study area covered, from most to least prospective), compute the proportion of
known deposits that fall within the top-t fraction of the map.

Plotting cumulative deposit capture (y) vs. cumulative area fraction (x)
gives the Success Rate Curve. A perfect model hugs the top-left corner;
random prediction is the diagonal.

The AUC of this curve (Success-Rate AUC, SR-AUC) measures exploration
targeting efficiency: SR-AUC > 0.5 means the model outperforms random
prioritisation.

Interpretation
--------------
  SR-AUC = 0.5  → random (no spatial discriminative power)
  SR-AUC = 0.8  → the top 20% of area contains ~80% of deposits (rough heuristic)
  SR-AUC = 1.0  → perfect (all deposits in the single highest-probability pixel)

Usage
-----
    python pipeline/03_training/success_rate_curve.py

Prerequisites
-------------
  • outputs/rf_prospectivity.tif   (from predict_full_extent.py)
  • outputs/xgb_prospectivity.tif  (from predict_full_extent.py)
  • data/raw/labels/vms_positive_labels.gpkg
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.transform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import auc as sklearn_auc
from datetime import datetime, timezone

PIPELINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_DIR))
from config import LABELS_DIR, VMS_LABELS_GPKG, PROCESSED_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO_ROOT   = PIPELINE_DIR.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

RASTER_PATHS = {
    "Random Forest": OUTPUTS_DIR / "rf_prospectivity_map.tif",
    "XGBoost":       OUTPUTS_DIR / "xgb_prospectivity_map.tif",
}

COLORS = {
    "Random Forest": "#2ecc71",
    "XGBoost":       "#e74c3c",
}


def load_prospectivity_scores(raster_path: Path) -> tuple[np.ndarray, object]:
    """Load valid (non-NoData) pixel values and rasterio transform from a raster."""
    with rasterio.open(raster_path) as src:
        arr    = src.read(1).astype(np.float64)
        nd     = src.nodata if src.nodata is not None else -9999
        transform = src.transform
        crs    = src.crs
        shape  = arr.shape

    mask  = (arr != nd) & ~np.isnan(arr) & (arr > -9990)
    valid = arr[mask]
    log.info(f"  {raster_path.name}: {valid.size:,} valid pixels")
    return valid, arr, mask, transform, shape, crs


def sample_deposit_scores(
    arr: np.ndarray,
    mask: np.ndarray,
    transform,
    shape: tuple,
    deposits_gdf: gpd.GeoDataFrame,
) -> np.ndarray:
    """
    Extract predicted prospectivity scores at each known deposit location.
    Points outside the raster extent or on NoData pixels are dropped.
    """
    scores = []
    xs = deposits_gdf.geometry.x.values
    ys = deposits_gdf.geometry.y.values

    rows, cols = rasterio.transform.rowcol(transform, xs, ys)
    rows = np.asarray(rows)
    cols = np.asarray(cols)

    in_bounds = (
        (rows >= 0) & (rows < shape[0]) &
        (cols >= 0) & (cols < shape[1])
    )
    r_valid = rows[in_bounds]
    c_valid = cols[in_bounds]

    # Only keep deposits that fall on valid (non-NoData) pixels
    on_valid = mask[r_valid, c_valid]
    r_final  = r_valid[on_valid]
    c_final  = c_valid[on_valid]
    scores   = arr[r_final, c_final]

    n_total   = len(deposits_gdf)
    n_sampled = len(scores)
    if n_sampled < n_total:
        log.warning(
            f"  {n_total - n_sampled} deposits outside raster extent / on NoData — "
            f"using {n_sampled} / {n_total} deposits for success rate curve."
        )
    else:
        log.info(f"  All {n_sampled} deposits sampled successfully.")
    return scores


def compute_success_rate_curve(
    pixel_scores: np.ndarray,
    deposit_scores: np.ndarray,
    n_thresholds: int = 2_000,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute the success rate curve using quantile-sampled thresholds.

    Parameters
    ----------
    pixel_scores   : 1-D array of all valid pixel prospectivity values.
    deposit_scores : 1-D array of prospectivity values at known deposit locations.
    n_thresholds   : Number of evenly-spaced quantile thresholds to sample (default 2000).
                     Using quantiles rather than every unique value keeps runtime O(n_thresholds)
                     instead of O(n_pixels), with negligible AUC error.

    Returns
    -------
    area_fractions     : x-axis — cumulative proportion of study area covered (0→1).
    deposit_fractions  : y-axis — cumulative proportion of deposits captured (0→1).
    sr_auc             : area under the success rate curve (sklearn trapezoid rule).
    """
    n_pixels   = len(pixel_scores)
    n_deposits = len(deposit_scores)

    # Sample n_thresholds quantile breakpoints from the pixel distribution.
    # This covers the full probability range without iterating over all ~1.2M unique values.
    quantile_probs = np.linspace(0.0, 1.0, n_thresholds)
    thresholds = np.quantile(pixel_scores, quantile_probs[::-1])  # descending

    # Use sorted arrays + searchsorted for O(n·log n + k·log n) runtime
    # instead of O(k·n) with a Python loop over all unique values.
    px_sorted  = np.sort(pixel_scores)          # ascending
    dep_sorted = np.sort(deposit_scores)        # ascending

    # For each threshold t:
    #   area fraction    = number of pixels   >= t  / n_pixels
    #   deposit fraction = number of deposits >= t  / n_deposits
    # bisect_left gives index of first element >= t in ascending sorted array.
    px_counts  = n_pixels   - np.searchsorted(px_sorted,  thresholds, side="left")
    dep_counts = n_deposits - np.searchsorted(dep_sorted, thresholds, side="left")

    area_fracs    = px_counts  / n_pixels
    deposit_fracs = dep_counts / n_deposits

    # Add (0,0) and (1,1) endpoints
    area_fracs    = np.concatenate([[0.0], area_fracs,    [1.0]])
    deposit_fracs = np.concatenate([[0.0], deposit_fracs, [1.0]])

    # Sort by area fraction for AUC computation
    order         = np.argsort(area_fracs)
    area_fracs    = area_fracs[order]
    deposit_fracs = deposit_fracs[order]

    sr_auc = sklearn_auc(area_fracs, deposit_fracs)
    return area_fracs, deposit_fracs, sr_auc


def write_metrics_report(
    results: dict,
    out_path: Path,
    models_dir: Path,
    target_areas: list[float] = [0.10, 0.20, 0.30],
) -> None:
    """
    Write a Markdown performance report combining CV metrics (ROC-AUC, AP, BA)
    from the model training CSVs with operating-point deposit-capture rates
    computed from the success rate curves.

    Parameters
    ----------
    results     : dict keyed by model name with keys 'area', 'deposit', 'sr_auc'.
    out_path    : Destination .md file path.
    models_dir  : Directory containing rf_cv_metrics.csv and xgb_cv_metrics.csv.
    target_areas: Area fractions at which to report deposit capture rates.
    """
    csv_map = {
        "Random Forest": models_dir / "rf_cv_metrics.csv",
        "XGBoost":       models_dir / "xgb_cv_metrics.csv",
    }

    # Load CV metrics
    cv = {}
    for model_name, csv_path in csv_map.items():
        if csv_path.exists():
            row = pd.read_csv(csv_path).iloc[0]
            cv[model_name] = row
        else:
            log.warning(f"  CV metrics CSV not found: {csv_path}")

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# VMS Prospectivity Model Performance Report",
        "",
        f"**Generated:** {timestamp}  ",
        f"**Study area:** Bathurst Mining Camp, New Brunswick, Canada  ",
        f"**Validation:** 5-fold spatial block cross-validation  ",
        f"**Models:** Random Forest (RF) · XGBoost  ",
        "",
        "---",
        "",
        "## 1. Cross-Validation Metrics",
        "",
        "| Metric | Random Forest | XGBoost |",
        "|---|---|---|",
    ]

    metrics = [
        ("ROC-AUC (mean ± SD)",      "roc_auc_mean",      "roc_auc_std"),
        ("Average Precision (mean ± SD)", "avg_prec_mean", "avg_prec_std"),
        ("Balanced Accuracy (mean ± SD)", "balanced_acc_mean", "balanced_acc_std"),
    ]

    for label, mean_key, std_key in metrics:
        rf_val  = f"{cv['Random Forest'][mean_key]:.4f} ± {cv['Random Forest'][std_key]:.4f}" if "Random Forest" in cv else "N/A"
        xgb_val = f"{cv['XGBoost'][mean_key]:.4f} ± {cv['XGBoost'][std_key]:.4f}"             if "XGBoost" in cv       else "N/A"
        lines.append(f"| {label} | {rf_val} | {xgb_val} |")

    # SR-AUC row
    rf_sr  = f"{results['Random Forest']['sr_auc']:.4f}" if "Random Forest" in results else "N/A"
    xgb_sr = f"{results['XGBoost']['sr_auc']:.4f}"       if "XGBoost" in results       else "N/A"
    lines.append(f"| Success Rate AUC | **{rf_sr}** | {xgb_sr} |")

    lines += [
        "",
        "---",
        "",
        "## 2. Deposit Capture Operating Points",
        "",
        "Percentage of known BMC VMS deposits (n = 45) captured when the study area",
        "is ranked by prospectivity index and the top-N% is prioritised.",
        "",
        "| Area Threshold | Random Forest | XGBoost |",
        "|---|---|---|",
    ]

    for target in target_areas:
        rf_capture  = "N/A"
        xgb_capture = "N/A"
        if "Random Forest" in results:
            af = results["Random Forest"]["area"]
            df = results["Random Forest"]["deposit"]
            idx = min(np.searchsorted(af, target), len(df) - 1)
            rf_capture = f"{df[idx]*100:.1f}%"
        if "XGBoost" in results:
            af = results["XGBoost"]["area"]
            df = results["XGBoost"]["deposit"]
            idx = min(np.searchsorted(af, target), len(df) - 1)
            xgb_capture = f"{df[idx]*100:.1f}%"
        lines.append(f"| Top {int(target*100)}% | **{rf_capture}** | {xgb_capture} |")

    lines += [
        "",
        "---",
        "",
        "## 3. Hyperparameter Summary",
        "",
    ]

    for model_name, csv_path in csv_map.items():
        if model_name in cv:
            params = cv[model_name].get("best_params", "N/A")
            lines.append(f"**{model_name}:** `{params}`  ")
            lines.append("")

    lines += [
        "---",
        "",
        "*Report auto-generated by `pipeline/03_training/success_rate_curve.py`*",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"\n📄 Performance report saved → {out_path}")


def main():
    log.info("══════════════════════════════════════════")
    log.info("  Success Rate Curve — VMS Prospectivity  ")
    log.info("══════════════════════════════════════════")

    # ── Load deposit locations ────────────────────────────────────────────────
    vms_path = LABELS_DIR / VMS_LABELS_GPKG.name
    if not vms_path.exists():
        vms_path = VMS_LABELS_GPKG
    deposits = gpd.read_file(vms_path)
    log.info(f"Deposits loaded: {len(deposits)} known VMS occurrences")

    # ── Compute curves for each model ─────────────────────────────────────────
    results = {}
    for model_name, raster_path in RASTER_PATHS.items():
        if not raster_path.exists():
            log.warning(f"  Raster not found: {raster_path} — skipping {model_name}")
            continue

        log.info(f"\n[{model_name}]")
        valid_scores, arr, mask, transform, shape, crs = load_prospectivity_scores(
            raster_path
        )

        # Reproject deposits to raster CRS if needed
        dep_reproj = deposits.to_crs(crs) if deposits.crs != crs else deposits

        dep_scores = sample_deposit_scores(
            arr, mask, transform, shape, dep_reproj
        )

        if len(dep_scores) == 0:
            log.error(f"  No deposit scores extracted — cannot compute SRC for {model_name}.")
            continue

        af, df, sr_auc = compute_success_rate_curve(valid_scores, dep_scores)

        results[model_name] = {"area": af, "deposit": df, "sr_auc": sr_auc}
        log.info(f"  SR-AUC ({model_name}): {sr_auc:.4f}")

        # Key operating points
        for target_area in [0.10, 0.20, 0.30]:
            idx = np.searchsorted(af, target_area)
            idx = min(idx, len(df) - 1)
            log.info(
                f"    Top {target_area*100:.0f}% area → "
                f"{df[idx]*100:.1f}% deposits captured"
            )

    if not results:
        log.error("No results to plot — ensure predict_full_extent.py has been run.")
        return

    # ── Summary table ─────────────────────────────────────────────────────────
    log.info("\n══════════════════════════════════════════")
    log.info("  SR-AUC Summary")
    log.info("══════════════════════════════════════════")
    for model_name, res in results.items():
        log.info(f"  {model_name:<20}: SR-AUC = {res['sr_auc']:.4f}")
    log.info("══════════════════════════════════════════")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))

    for model_name, res in results.items():
        ax.plot(
            res["area"], res["deposit"],
            color=COLORS.get(model_name, "steelblue"),
            lw=2,
            label=f"{model_name}  (SR-AUC = {res['sr_auc']:.3f})",
        )

    # Random prediction diagonal
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (SR-AUC = 0.500)")

    # Shade area under RF curve (first model)
    first_model = list(results.values())[0]
    ax.fill_between(
        first_model["area"], first_model["deposit"],
        alpha=0.08, color=COLORS.get(list(results.keys())[0], "steelblue"),
    )

    ax.set_xlabel("Cumulative Area Fraction (study area, ranked by prospectivity)", fontsize=11)
    ax.set_ylabel("Cumulative Deposit Capture Fraction", fontsize=11)
    ax.set_title(
        "Success Rate Curve — Bathurst VMS Prospectivity\n"
        "(Mahalanobis dissimilarity pseudo-absences, Parsa & Cumani 2025)",
        fontsize=11,
    )
    ax.legend(fontsize=10, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    out_path = OUTPUTS_DIR / "success_rate_curve.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"\n✅ Success rate curve saved → {out_path}")

    # ── Write Markdown performance report ────────────────────────────────────
    from config import MODELS_DIR
    report_path = OUTPUTS_DIR / "model_performance_report.md"
    write_metrics_report(results, report_path, MODELS_DIR)

    log.info("   Run next: python pipeline/05_explainability/shap_analysis.py")


if __name__ == "__main__":
    main()
