"""
config.py — Central configuration for VMS Deposit Discovery AI Pipeline
Bathurst Mining Camp, New Brunswick, Canada

All pipeline scripts import from this module to ensure consistent
paths, CRS settings, and hyperparameters.
"""

from pathlib import Path

# ── Repository Root ───────────────────────────────────────────────────────────
# Resolves to the repo root regardless of where the script is called from.
REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Data Directory Structure ──────────────────────────────────────────────────
DATA_DIR        = REPO_ROOT / "data"
RAW_DIR         = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"
RASTERS_DIR     = RAW_DIR / "rasters"          # Geophysical grids
LABELS_DIR      = RAW_DIR / "labels"           # Deposit & barren locations
MODELS_DIR      = REPO_ROOT / "models"         # Saved .joblib model files
OUTPUTS_DIR     = REPO_ROOT / "outputs"        # Maps, figures, reports
NOTEBOOKS_DIR   = REPO_ROOT / "pipeline" / "notebooks"

# Create directories if they don't exist
for _dir in [RAW_DIR, PROCESSED_DIR, RASTERS_DIR, LABELS_DIR, MODELS_DIR, OUTPUTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── Coordinate Reference Systems ──────────────────────────────────────────────
CRS_SOURCE  = "EPSG:4326"   # WGS84 — source CRS for most downloaded data
CRS_TARGET  = "EPSG:2953"   # NAD83 / New Brunswick Double Stereographic
                             # Standard for NB provincial spatial datasets

# ── Study Area (Bathurst Mining Camp) ────────────────────────────────────────
# Approximate bounding box in WGS84 (lon_min, lat_min, lon_max, lat_max)
BMC_BBOX_WGS84 = (-66.85, 47.10, -65.20, 47.95)

# Buffer (metres) applied around known VMS deposits for positive labelling
POSITIVE_BUFFER_M = 500   # 500m radius = confident mineralised zone
NEGATIVE_BUFFER_M = 500   # 500m radius around confirmed barren holes

# ── Raster Processing ─────────────────────────────────────────────────────────
TARGET_RESOLUTION_M = 100   # 100m pixel resolution for all geophysical grids
NODATA_VALUE        = -9999

# ── Feature Engineering ───────────────────────────────────────────────────────
# Geophysical raster bands expected after download/preprocessing
RASTER_FEATURES = [
    "mag_tmi",          # Total Magnetic Intensity
    "mag_fvd",          # First Vertical Derivative
    "mag_as",           # Analytic Signal (derived)
    "mag_hg",           # Horizontal Gradient Magnitude (derived)
    "rad_k",            # Radiometric potassium %
    "rad_th",           # Thorium (ppm)
    "rad_u",            # Uranium (ppm)
    "rad_k_th_ratio",   # K/Th ratio (derived)
    "em_conductivity",  # Apparent conductivity (EM)
    "gravity_bouguer",  # Bouguer anomaly
]

# Till geochemistry pathfinder elements
GEOCHEM_FEATURES = [
    "zn_ppm", "pb_ppm", "cu_ppm",
    "ag_ppm", "au_ppb", "as_ppm",
]

ALL_FEATURES = RASTER_FEATURES + GEOCHEM_FEATURES
TARGET_COLUMN = "label"   # 1 = VMS mineralised, 0 = barren

# ── Model Training ────────────────────────────────────────────────────────────
N_SPATIAL_FOLDS   = 5       # Spatial cross-validation folds
RANDOM_STATE      = 42
N_OPTUNA_TRIALS   = 50      # Bayesian hyperparameter search trials
CLASS_WEIGHT      = "balanced"

# ── Output File Names ─────────────────────────────────────────────────────────
GEOCHEMISTRY_GPKG   = RAW_DIR      / "nb_till_geochemistry.gpkg"
VMS_LABELS_GPKG     = LABELS_DIR   / "vms_positive_labels.gpkg"
BARREN_LABELS_GPKG  = LABELS_DIR   / "barren_negative_labels.gpkg"
FEATURE_MATRIX_PQ   = PROCESSED_DIR / "feature_matrix.parquet"
RF_MODEL_PATH       = MODELS_DIR   / "rf_best_model.joblib"
XGB_MODEL_PATH      = MODELS_DIR   / "xgb_best_model.joblib"
PROSPECTIVITY_TIFF  = OUTPUTS_DIR  / "bmc_prospectivity_map.tif"

print(f"[config] Repo root   : {REPO_ROOT}")
print(f"[config] Data dir    : {DATA_DIR}")
print(f"[config] Target CRS  : {CRS_TARGET}")
