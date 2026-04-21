"""
engineer_features.py
─────────────────────
Derives secondary geophysical features from the primary raster bands already
extracted into the feature matrix. These derived features often have stronger
geological signal than the raw band values for VMS targeting.

Derived features:
  Magnetics:
    • TMI Horizontal Gradient Magnitude  = sqrt(dTMI/dx² + dTMI/dy²)
    • Analytic Signal Amplitude          = sqrt(dTMI/dx² + dTMI/dy² + dTMI/dz²)
    ↳ Both approximate using TMI and FVD (as a proxy for dTMI/dz)

  Radiometrics:
    • K/Th ratio   — high values → potassic alteration halos around VMS
    • U/Th ratio   — useful for separating alteration styles
    • eTh/eK ratio — indicator of thorium-enriched zones

  Geochemistry:
    • log-transformed pathfinder elements (lognormal distributions)
    • Multi-element anomaly score: standardised Z sum of Zn, Pb, Cu, As

After running this script, the feature matrix parquet is updated in-place
with the new derived columns appended.

Usage:
    python pipeline/02_preprocessing/engineer_features.py
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd

PIPELINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_DIR))
from config import FEATURE_MATRIX_PQ

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# Minimum valid values (to guard against divide-by-zero / log(0))
EPSILON = 1e-6

# Expected column names for raw features (must match raster file stems)
MAG_TMI  = "mag_tmi_nb_2013"
MAG_FVD  = "mag_fvd_nb_2013"
RAD_K    = "rad_k"
RAD_TH   = "rad_th"
RAD_U    = "rad_u"

GEOCHEM_COLS = ["zn_ppm", "pb_ppm", "cu_ppm", "ag_ppm", "au_ppb", "as_ppm"]


# ── Magnetics Derived Features ─────────────────────────────────────────────────

def compute_horizontal_gradient(tmi: np.ndarray, fvd: np.ndarray) -> np.ndarray:
    """
    Approximate horizontal gradient magnitude.
    For a 1D series, use gradient of TMI as proxy for dx component,
    and FVD (vertical derivative) as the dz component.
    HGM ≈ sqrt(grad_tmi² + fvd²)
    """
    grad_tmi = np.gradient(tmi)
    return np.sqrt(grad_tmi**2 + fvd**2 + EPSILON)


def compute_analytic_signal(tmi: np.ndarray, fvd: np.ndarray) -> np.ndarray:
    """
    Analytic Signal Amplitude approximation.
    AS = sqrt(dx_tmi² + dy_tmi² + fvd²)
    For tabular data (points, not a 2D grid), uses gradient of TMI
    as a 1D approximation of the horizontal components.
    Note: For full 2D analytic signal, run on the reprojected raster grid.
    """
    grad_tmi = np.gradient(tmi)
    return np.sqrt(2 * grad_tmi**2 + fvd**2 + EPSILON)


# ── Radiometric Derived Features ───────────────────────────────────────────────

def compute_k_th_ratio(k: np.ndarray, th: np.ndarray) -> np.ndarray:
    """K/Th ratio — elevated values indicate potassic alteration (VMS footprint)."""
    th_safe = np.where(th < EPSILON, EPSILON, th)
    return k / th_safe


def compute_u_th_ratio(u: np.ndarray, th: np.ndarray) -> np.ndarray:
    """U/Th ratio — useful for distinguishing alteration styles."""
    th_safe = np.where(th < EPSILON, EPSILON, th)
    return u / th_safe


def compute_th_k_ratio(th: np.ndarray, k: np.ndarray) -> np.ndarray:
    """eTh/eK ratio — thorium enriched zones."""
    k_safe = np.where(k < EPSILON, EPSILON, k)
    return th / k_safe


# ── Geochemistry Derived Features ──────────────────────────────────────────────

def log_transform(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Apply log10 transformation to skewed geochemistry columns."""
    for col in cols:
        if col in df.columns:
            raw = df[col].values.astype(float)
            # Clip to small positive value before log
            raw_clipped = np.clip(raw, EPSILON, None)
            df[f"log_{col}"] = np.where(
                np.isnan(raw), np.nan, np.log10(raw_clipped)
            )
            log.info(f"    log_{col}: computed")
    return df


def compute_multi_element_score(df: pd.DataFrame) -> pd.Series:
    """
    Multi-element anomaly score (MEAS).
    Standardises Zn, Pb, Cu, As individually then sums Z-scores.
    High values → geochemical signature consistent with VMS pathfinder train.
    """
    score = pd.Series(0.0, index=df.index)
    contributing = 0
    for col in ["zn_ppm", "pb_ppm", "cu_ppm", "as_ppm"]:
        if col in df.columns and df[col].notna().any():
            z = (df[col] - df[col].mean()) / (df[col].std() + EPSILON)
            score += z.fillna(0)
            contributing += 1
    if contributing == 0:
        return pd.Series(np.nan, index=df.index)
    return score / contributing   # Normalise by number of contributing elements


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    log.info("═══ Feature Engineering ═══")

    if not FEATURE_MATRIX_PQ.exists():
        raise FileNotFoundError(
            f"Feature matrix not found: {FEATURE_MATRIX_PQ}\n"
            "Run: python pipeline/02_preprocessing/extract_features.py"
        )

    df = pd.read_parquet(FEATURE_MATRIX_PQ)
    log.info(f"Loaded feature matrix: {df.shape}")
    initial_cols = set(df.columns)

    # ── Magnetics features ───────────────────────────────────────────────────
    if MAG_TMI in df.columns and MAG_FVD in df.columns:
        log.info("\n[Magnetics] Computing derived features ...")
        tmi = df[MAG_TMI].fillna(0).values
        fvd = df[MAG_FVD].fillna(0).values

        df["mag_hgm"]  = compute_horizontal_gradient(tmi, fvd)
        df["mag_as"]   = compute_analytic_signal(tmi, fvd)
        log.info("  ✅ mag_hgm (horizontal gradient magnitude)")
        log.info("  ✅ mag_as  (analytic signal amplitude)")
    else:
        log.warning(
            f"  Magnetic columns not found ({MAG_TMI}, {MAG_FVD}). "
            "Skipping magnetic feature derivation – will be populated after raster download."
        )
        df["mag_hgm"] = np.nan
        df["mag_as"]  = np.nan

    # ── Radiometrics features ────────────────────────────────────────────────
    rad_available = all(c in df.columns for c in [RAD_K, RAD_TH, RAD_U])
    if rad_available:
        log.info("\n[Radiometrics] Computing ratios ...")
        k  = df[RAD_K].fillna(0).values
        th = df[RAD_TH].fillna(EPSILON).values
        u  = df[RAD_U].fillna(0).values

        df["rad_k_th"] = compute_k_th_ratio(k, th)
        df["rad_u_th"] = compute_u_th_ratio(u, th)
        df["rad_th_k"] = compute_th_k_ratio(th, k)
        log.info("  ✅ rad_k_th (K/Th ratio)")
        log.info("  ✅ rad_u_th (U/Th ratio)")
        log.info("  ✅ rad_th_k (Th/K ratio)")
    else:
        log.warning(
            "  Radiometric columns not found. "
            "Skipping radiometric ratios – will be populated after raster download."
        )
        df["rad_k_th"] = np.nan
        df["rad_u_th"] = np.nan
        df["rad_th_k"] = np.nan

    # ── Geochemistry features ────────────────────────────────────────────────
    geochem_available = any(c in df.columns for c in GEOCHEM_COLS)
    if geochem_available:
        log.info("\n[Geochemistry] Log-transforming pathfinder elements ...")
        df = log_transform(df, GEOCHEM_COLS)

        log.info("  Computing multi-element anomaly score (MEAS) ...")
        df["geochem_meas"] = compute_multi_element_score(df)
        log.info("  ✅ geochem_meas")
    else:
        log.warning("  No geochemistry columns found. Skipping geochem features.")
        df["geochem_meas"] = np.nan

    # ── Summary ──────────────────────────────────────────────────────────────
    new_cols = set(df.columns) - initial_cols
    log.info(f"\n─── {len(new_cols)} new features engineered ───")
    for col in sorted(new_cols):
        n_valid = df[col].notna().sum()
        log.info(f"  {col:30s}: {n_valid:5d} valid / {len(df)} total")

    # ── Save updated feature matrix ──────────────────────────────────────────
    df.to_parquet(FEATURE_MATRIX_PQ, index=False)
    log.info(f"\n✅ Updated feature matrix saved → {FEATURE_MATRIX_PQ}")
    log.info(f"   Final shape: {df.shape}")
    log.info("   Run next: python pipeline/03_training/build_dataset.py")


if __name__ == "__main__":
    main()
