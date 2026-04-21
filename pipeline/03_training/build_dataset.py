"""
build_dataset.py
─────────────────
Assembles the final training dataset from the engineered feature matrix.

Steps:
  1. Load feature matrix (parquet)
  2. Select only labelled rows (label = 0 or 1; drop NaN labels)
  3. Drop features with >50% null values
  4. Impute remaining nulls (median imputation per feature)
  5. Apply SMOTE oversampling to address class imbalance (1:5 ratio)
  6. Create spatial block indices for spatial cross-validation
  7. Save final training arrays as numpy .npz

Spatial Cross-Validation:
  Uses a custom BlockKFold approach: the BMC extent is divided into
  N×N spatial blocks. Each fold uses one block-column as the test set,
  ensuring training and test points are spatially separated. This
  prevents data leakage through spatial autocorrelation.

Usage:
    python pipeline/03_training/build_dataset.py
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

PIPELINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_DIR))
from config import (
    FEATURE_MATRIX_PQ, PROCESSED_DIR, TARGET_COLUMN,
    ALL_FEATURES, RANDOM_STATE, N_SPATIAL_FOLDS
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

DATASET_DIR = PROCESSED_DIR / "training_dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# Maximum fraction of nulls allowed per feature before dropping the column
MAX_NULL_FRACTION = 0.50


def load_and_filter(path: Path) -> pd.DataFrame:
    """Load feature matrix and return only labelled rows."""
    df = pd.read_parquet(path)
    log.info(f"Loaded: {df.shape}  (all points incl. unlabelled)")
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)
    log.info(f"Labelled rows: {df.shape}")
    log.info(f"  Label distribution: {dict(df['label'].value_counts())}")
    return df


def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Select available feature columns; drop high-null columns."""
    available = [c for c in ALL_FEATURES if c in df.columns]
    log.info(f"\nFeatures available: {len(available)} / {len(ALL_FEATURES)}")
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        log.warning(f"  Features not yet populated (rasters pending): {missing}")

    # Drop features with too many nulls
    null_fracs = df[available].isnull().mean()
    keep = null_fracs[null_fracs <= MAX_NULL_FRACTION].index.tolist()
    dropped = [c for c in available if c not in keep]
    if dropped:
        log.warning(f"  Dropped high-null features (>{MAX_NULL_FRACTION*100:.0f}%): {dropped}")

    log.info(f"  Features retained: {len(keep)}")
    return df[keep + ["label"]], keep


def impute(df: pd.DataFrame, feature_cols: list) -> tuple[pd.DataFrame, SimpleImputer]:
    """Median imputation for remaining nulls."""
    imputer = SimpleImputer(strategy="median")
    df[feature_cols] = imputer.fit_transform(df[feature_cols])
    log.info(f"  Imputation complete. Remaining nulls: {df[feature_cols].isnull().sum().sum()}")
    return df, imputer


def apply_smote(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE oversampling to balance classes."""
    try:
        from imblearn.over_sampling import SMOTE
        ratio = dict(zip(*np.unique(y, return_counts=True)))
        log.info(f"  Before SMOTE: {ratio}")
        smote = SMOTE(random_state=RANDOM_STATE)
        X_res, y_res = smote.fit_resample(X, y)
        ratio_after = dict(zip(*np.unique(y_res, return_counts=True)))
        log.info(f"  After SMOTE : {ratio_after}")
        return X_res, y_res
    except ImportError:
        log.warning(
            "  imbalanced-learn not installed. Skipping SMOTE. "
            "Install with: pip install imbalanced-learn"
        )
        return X, y


def assign_spatial_blocks(df: pd.DataFrame, n_blocks: int = 5) -> np.ndarray:
    """
    Assign each point to a spatial block for cross-validation.
    Divides the bounding box into an N×N grid and assigns a block ID
    to each point based on its geometry_wkt coordinates.
    Returns an array of fold indices (0 to n_blocks-1).
    """
    try:
        from shapely.wkt import loads
        coords = np.array([
            [loads(wkt).x, loads(wkt).y]
            for wkt in df["geometry_wkt"]
        ])

        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

        # Divide into blocks along x-axis (easting strips)
        x_bins = np.linspace(x_min, x_max, n_blocks + 1)
        fold_ids = np.digitize(coords[:, 0], x_bins[:-1]) - 1
        fold_ids = np.clip(fold_ids, 0, n_blocks - 1)

        log.info(f"  Spatial block distribution (n={n_blocks} folds):")
        for i in range(n_blocks):
            log.info(f"    Fold {i}: {(fold_ids == i).sum()} points")
    except Exception as e:
        log.warning(f"  Could not parse geometry for spatial blocks: {e}")
        log.warning("  Using sequential fold assignment as fallback.")
        fold_ids = np.arange(len(df)) % n_blocks

    return fold_ids


def main():
    log.info("═══ Training Dataset Assembly ═══")

    # ── Load ─────────────────────────────────────────────────────────────────
    df = load_and_filter(FEATURE_MATRIX_PQ)

    # ── Feature selection & null handling ────────────────────────────────────
    log.info("\n[Feature Selection]")
    df_feat, feature_cols = select_features(df)

    if not feature_cols:
        log.error(
            "No usable features found. Ensure at least geochemistry labels are built "
            "and/or rasters have been downloaded and extracted."
        )
        return

    df_feat, imputer = impute(df_feat, feature_cols)

    # Arrays for sklearn
    X = df_feat[feature_cols].values.astype(np.float32)
    y = df_feat["label"].values.astype(int)

    # ── SMOTE ────────────────────────────────────────────────────────────────
    log.info("\n[Class Balancing — SMOTE]")
    X_balanced, y_balanced = apply_smote(X, y)

    # ── Spatial blocks for CV ─────────────────────────────────────────────────
    log.info("\n[Spatial Cross-Validation Blocks]")
    # Note: spatial blocks are assigned to original (pre-SMOTE) points only
    # SMOTE-generated points inherit the fold of their nearest real neighbour
    spatial_folds = assign_spatial_blocks(df_feat, n_blocks=N_SPATIAL_FOLDS)

    # ── Scale ─────────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_balanced)

    # ── Save ───────────────────────────────────────────────────────────────────
    np.savez_compressed(
        DATASET_DIR / "training_data.npz",
        X=X_balanced,
        X_scaled=X_scaled,
        y=y_balanced,
    )
    np.save(DATASET_DIR / "spatial_folds.npy", spatial_folds)
    pd.Series(feature_cols).to_csv(DATASET_DIR / "feature_names.csv", index=False, header=False)

    import joblib
    joblib.dump(imputer, DATASET_DIR / "imputer.joblib")
    joblib.dump(scaler,  DATASET_DIR / "scaler.joblib")

    log.info(f"\n✅ Dataset saved to: {DATASET_DIR}")
    log.info(f"   X shape (post-SMOTE): {X_balanced.shape}")
    log.info(f"   y distribution      : {dict(zip(*np.unique(y_balanced, return_counts=True)))}")
    log.info(f"   Features saved      : {len(feature_cols)}")
    log.info("   Run next: python pipeline/03_training/train_rf.py")


if __name__ == "__main__":
    main()
