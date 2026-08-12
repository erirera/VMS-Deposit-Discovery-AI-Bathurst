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
PROCESSED_DIR       = DATA_DIR / "processed"
RASTERS_DIR         = RAW_DIR / "rasters"              # Geophysical grids
LABELS_DIR          = RAW_DIR / "labels"               # Deposit & barren locations
MAG_DERIVATIVES_DIR = PROCESSED_DIR / "mag_derivatives" # Computed magnetic derivatives
GRAV_DERIVATIVES_DIR = PROCESSED_DIR / "grav_derivatives"   # Computed gravity derivatives
RAD_DERIVATIVES_DIR = PROCESSED_DIR / "rad_derivatives"     # Computed radiometric derivatives
MODELS_DIR          = REPO_ROOT / "models"             # Saved .joblib model files
OUTPUTS_DIR         = REPO_ROOT / "outputs"            # Maps, figures, reports
NOTEBOOKS_DIR       = REPO_ROOT / "pipeline" / "notebooks"

# Create directories if they don't exist
for _dir in [RAW_DIR, PROCESSED_DIR, RASTERS_DIR, LABELS_DIR,
             MAG_DERIVATIVES_DIR, GRAV_DERIVATIVES_DIR, RAD_DERIVATIVES_DIR, MODELS_DIR, OUTPUTS_DIR]:
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

# ── Pseudo-Absence Dissimilarity Sampling (Parsa & Cumani, 2025) ──────────────
# Pseudo-absences are selected by maximising Mahalanobis distance in
# multi-dimensional feature space from the VMS deposit centroid, rather than
# by a simple geographic exclusion buffer.
PSEUDO_ABSENCE_CANDIDATE_SPACING_M = 100   # Dense candidate grid spacing (m); matches TARGET_RESOLUTION_M
PSEUDO_ABSENCE_MIN_BUFFER_M        = 1000  # Secondary geographic guard — hard minimum distance from any deposit
PSEUDO_ABSENCE_DISSIM_QUANTILE     = 0.60  # Sample from top-40% most dissimilar candidates (quantile threshold)
PSEUDO_ABSENCE_N_SPATIAL_STRATA    = 4     # Spatial quadrants for stratified selection (ensures geographic spread)

# ── Raster Processing ─────────────────────────────────────────────────────────
TARGET_RESOLUTION_M      = 100   # 100m pixel resolution for all geophysical grids
MAG_DERIVATIVE_RESOLUTION_M = 50 # 50m resolution for magnetic derivative grids
NODATA_VALUE             = -9999

# ── Feature Engineering ───────────────────────────────────────────────────────
# Geophysical raster bands expected after download/preprocessing
RASTER_FEATURES = [
    "mag_tmi",          # Total Magnetic Intensity
    # ── Magnetic derivatives (computed by compute_mag_derivatives.py) ──
    "mag_rmi_bmc_combined1",  # Compiled TMI grid (source for mag_hgm / mag_as derivation)
    "mag_rmi_fvd_bmc",  # First Vertical Derivative
    "mag_rmi_thg_bmc",  # Total Horizontal Gradient
    "mag_rmi_as_bmc",   # Analytic Signal Amplitude
    "mag_rmi_tdr_bmc",  # Tilt Derivative
    # -- Gravity derivatives (computed by compute_grav_derivatives.py) --
    "gra_ggr_hgm_bmc",   # Horizontal Gradient Magnitude
    "gra_ggr_tdr_bmc",   # Tilt Derivative
    "gra_ggr_fvd_bmc",   # First Vertical Derivative
    "gra_ggr_as_bmc",    # Analytic Signal Amplitude
    "gra_ggr_uc500_bmc", # Upward Continued 500 m
    "gra_ggr_res_bmc",   # Residual Bouguer
    # -- Radiometrics (computed by compute_rad_derivatives.py) --
    "rad_k_bmc",        # Radiometric potassium %
    "rad_th_bmc",       # Thorium (ppm)
    "rad_u_bmc",        # Uranium (ppm)
    "rad_k_th_bmc",     # K/Th ratio (derived)
    "rad_u_th_bmc",     # U/Th ratio (derived)
    "rad_th_k_bmc",     # Th/K ratio (derived)
    "rad_u_k_bmc",      # U/K ratio (derived)
    "gravity_bouguer",  # Bouguer anomaly

    # -- Till geochemistry IDW raster surfaces (from interpolate_geochem.py) --
    "geochem_ag_ppm_idw",   # Silver (ppm)
    "geochem_as_ppm_idw",   # Arsenic (ppm)
    "geochem_ba_ppm_idw",   # Barium (ppm)
    "geochem_bi_ppm_idw",   # Bismuth (ppm)
    "geochem_cd_ppm_idw",   # Cadmium (ppm)
    "geochem_co_ppm_idw",   # Cobalt (ppm)
    "geochem_cu_ppm_idw",   # Copper (ppm)
    "geochem_fe_ppm_idw",   # Iron (ppm)
    "geochem_in_ppm_idw",   # Indium (ppm)
    "geochem_mn_ppm_idw",   # Manganese (ppm)
    "geochem_mo_ppm_idw",   # Molybdenum (ppm)
    "geochem_ni_ppm_idw",   # Nickel (ppm)
    "geochem_pb_ppm_idw",   # Lead (ppm)
    "geochem_sb_ppm_idw",   # Antimony (ppm)
    "geochem_sn_ppm_idw",   # Tin (ppm)
    "geochem_tl_ppm_idw",   # Thallium (ppm)
    "geochem_zn_ppm_idw",   # Zinc (ppm)

    # -- Till geochemistry PCA / FA scores (IDW) --
    "geochem_pca_pc1_idw",
    "geochem_pca_pc2_idw",
    "geochem_pca_pc3_idw",
    "geochem_pca_pc4_idw",
    "geochem_fa_factor1_idw",
    "geochem_fa_factor2_idw",
    "geochem_fa_factor3_idw",
    "geochem_fa_factor4_idw",

]

# Till geochemistry pathfinder elements
GEOCHEM_FEATURES = [
    "ag_ppm", "as_ppm", "ba_ppm", "bi_ppm", "cd_ppm", "co_ppm", "cu_ppm",
    "fe_ppm", "in_ppm", "mn_ppm", "mo_ppm", "ni_ppm", "pb_ppm", "sb_ppm",
    "sn_ppm", "tl_ppm", "zn_ppm"
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
RF_PROSPECTIVITY_TIFF  = OUTPUTS_DIR / "rf_prospectivity_map.tif"
XGB_PROSPECTIVITY_TIFF = OUTPUTS_DIR / "xgb_prospectivity_map.tif"
PROSPECTIVITY_TIFF     = RF_PROSPECTIVITY_TIFF   # canonical alias (RF = primary model)

print(f"[config] Repo root   : {REPO_ROOT}")
print(f"[config] Data dir    : {DATA_DIR}")
print(f"[config] Target CRS  : {CRS_TARGET}")
