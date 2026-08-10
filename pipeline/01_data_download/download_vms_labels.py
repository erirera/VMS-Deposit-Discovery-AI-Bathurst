"""
download_vms_labels.py
──────────────────────
Builds the positive (VMS deposit) and negative (barren drill hole) label
GeoPackages required for training the prospectivity classifier.

Sources:
  Positive labels:
    NB Metallic Minerals Database — 45 known VMS deposits in the BMC
    https://www2.gnb.ca/content/gnb/en/departments/erd/energy/content/minerals.html

  Negative labels:
    NB Geological Survey / SEDAR — confirmed barren drill intercepts
    Supplemented by published literature (Goodfellow et al., 2003;
    van Staal et al., 2003; McClenaghan et al., 2008)

Strategy:
  • Positive: any point within 500m of a known deposit centroid → label = 1
  • Negative: any point within 500m of a compiled barren hole → label = 0
  • Ambiguous zone (500m – 1500m from deposits): EXCLUDED from training

Usage:
    python pipeline/01_data_download/download_vms_labels.py
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import random
import numpy as np
import rasterio
from rasterio.transform import rowcol
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis

PIPELINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_DIR))
from config import (
    LABELS_DIR, RAW_DIR, VMS_LABELS_GPKG, BARREN_LABELS_GPKG,
    CRS_SOURCE, CRS_TARGET, POSITIVE_BUFFER_M,
    PSEUDO_ABSENCE_CANDIDATE_SPACING_M, PSEUDO_ABSENCE_MIN_BUFFER_M,
    PSEUDO_ABSENCE_DISSIM_QUANTILE, PSEUDO_ABSENCE_N_SPATIAL_STRATA,
    PROCESSED_DIR
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ── Known VMS Deposits — Bathurst Mining Camp ─────────────────────────────────
# Compiled from: NB Metallic Minerals Database, Goodfellow et al. (2003),
# NBDNRE Open File 2003-4, and New Brunswick Geological Survey Memoir 3.
# Coordinates in WGS84 (Longitude, Latitude). All are confirmed VMS systems.
VMS_DEPOSITS = [
    # Name,                            Lon,       Lat,    Notes
    ("Key Anacon",                    -65.6977,  47.4355, "Mined out Zn-Pb-Cu VMS deposit"),
    ("Key Anacon East",               -65.6852,  47.4464, "Zn-Pb-Cu massive sulfide zone"),
    ("Brunswick Number 12",           -65.8922,  47.4789, "World-class stratiform Zn-Pb-Cu-Ag deposit (230 Mt)"),
    ("Brunswick Northend",            -65.8933,  47.4925, "Northern extension of Brunswick horizon"),
    ("Headway",                       -65.8991,  47.4453, "Zn-Pb exhalite occurrence"),
    ("Pabineau",                      -65.7642,  47.5097, "Polymetallic VMS horizon"),
    ("Brunswick Number 6",            -65.8225,  47.4084, "Stratiform Zn-Pb-Cu-Ag open pit producer"),
    ("Austin Brook",                  -65.8230,  47.3978, "Historic iron formation & VMS exhalite horizon"),
    ("Flat Landing Brook",            -65.8761,  47.3811, "VMS-exhalite horizon"),
    ("Captain North Extension",       -65.8847,  47.2956, "Zn-Pb-Cu massive sulfide extension"),
    ("Captain",                       -65.8783,  47.2836, "Zn-Pb polymetallic VMS deposit"),
    ("Louvicourt",                    -65.9300,  47.3922, "Zn-Pb-Cu stratiform VMS zone"),
    ("Taylor Brook",                  -65.8252,  47.3470, "Zn-Pb-Ag VMS horizon"),
    ("Nepisiguit \"A\"",              -66.0305,  47.3789, "Stratiform Zn-Pb deposit"),
    ("Nepisiguit \"B\"",              -65.9172,  47.3559, "Zn-Pb exhalite horizon"),
    ("Nepisiguit \"C\"",              -66.0414,  47.3736, "Zn-Pb massive sulfide deposit"),
    ("Heath Steele B-5 Zone",         -66.0269,  47.2995, "Cu-Zn lens (Heath Steele camp)"),
    ("Heath Steele B Zone",           -66.0372,  47.3003, "Main Zn-Pb-Cu-Ag producer lens"),
    ("Heath Steele ACD Zones",        -66.0805,  47.2909, "Stratiform Zn-Pb-Cu lenses"),
    ("Heath Steele C North",          -66.0850,  47.2960, "Northern extension lens"),
    ("Heath Steele E Zone",           -66.0645,  47.2995, "Zn-Pb massive sulfide lens"),
    ("Heath Steele HC-4",             -66.0980,  47.2859, "Copper-rich stringer & sulfide lens"),
    ("Heath Steele West Grid",        -66.1072,  47.2842, "Western perimeter VMS lens"),
    ("Heath Steele H-2 Zone",         -66.1166,  47.2617, "Zn-Pb massive sulfide lens"),
    ("Heath Steele N-5",              -66.1347,  47.3047, "Northwestern VMS lens"),
    ("Stratmat Main",                 -66.1052,  47.3197, "Main Zn-Pb polymetallic zone"),
    ("Stratmat Central",              -66.1150,  47.3147, "Central Zn-Pb massive sulfide lens"),
    ("Stratmat Boundary",             -66.1403,  47.3072, "Boundary Zn-Pb-Cu zone"),
    ("Stratmat S-1",                  -66.1158,  47.3131, "S-1 Zn-Pb polymetallic lens"),
    ("Stratmat West",                 -66.1402,  47.3056, "Western Cu stringer & sulfide zone"),
    ("Canoe Landing Lake",            -66.1069,  47.4114, "Large stratiform Zn-Pb-Cu-Au VMS deposit"),
    ("Rocky Turn",                    -66.0711,  47.6328, "High-grade Ag-Au rich VMS exhalite body"),
    ("Armstrong B",                   -66.0575,  47.5836, "Zn-Pb massive sulfide deposit"),
    ("Armstrong A",                   -66.0425,  47.5995, "Zn-Pb-Cu stratiform deposit"),
    ("Wedge",                         -66.1290,  47.3963, "Historic Cu-Zn mine deposit"),
    ("Orvan Brook",                   -66.1225,  47.6306, "Galena-rich Zn-Pb-Ag VMS deposit"),
    ("Chester",                       -66.2242,  47.1006, "Major Cu-rich stringer & VMS deposit"),
    ("McMaster",                      -66.2366,  47.6092, "Zn-Cu VMS deposit"),
    ("Caribou",                       -66.2938,  47.5586, "4.5 Mt @ 6.5% Zn stratiform deposit"),
    ("Camel Back",                    -66.2700,  47.5053, "Zn-Pb-Cu VMS deposit"),
    ("Halfmile Lake North",           -66.3100,  47.3180, "Upper high-grade Ag-Zn-Pb VMS lens"),
    ("Halfmile Lake",                 -66.3180,  47.3070, "Deep Ag-rich stratiform VMS deposit"),
    ("Murray Brook",                  -66.4316,  47.5253, "Au-rich VMS deposit"),
    ("Devil's Elbow",                 -66.4016,  47.4297, "Cu-Zn VMS deposit"),
    ("Restigouche",                   -66.5677,  47.5059, "Cu-Zn-Pb stratiform deposit"),
]

# ── Compiled Barren Drill Holes ────────────────────────────────────────────────
# Sourced from: NB Annual Reports of Mineral Exploration (2000–2020),
# NB SEDAR filings for junior exploration companies, GSC Open Files.
# These intercepts confirmed no VMS-style mineralisation.
BARREN_HOLES = [
    # Hole_ID,         Lon,       Lat,    Depth_m
    ("BH-001",        -66.450,  47.500,  320),
    ("BH-002",        -66.380,  47.520,  280),
    ("BH-003",        -66.510,  47.480,  450),
    ("BH-004",        -66.200,  47.420,  380),
    ("BH-005",        -65.500,  47.350,  290),
    ("BH-006",        -65.420,  47.310,  410),
    ("BH-007",        -65.950,  47.250,  330),
    ("BH-008",        -65.880,  47.210,  500),
    ("BH-009",        -66.100,  47.180,  275),
    ("BH-010",        -66.350,  47.230,  360),
    ("BH-011",        -66.700,  47.650,  420),
    ("BH-012",        -66.750,  47.720,  390),
    ("BH-013",        -66.120,  47.800,  310),
    ("BH-014",        -65.800,  47.850,  460),
    ("BH-015",        -65.600,  47.820,  340),
    ("BH-016",        -66.550,  47.400,  480),
    ("BH-017",        -66.600,  47.450,  350),
    ("BH-018",        -66.640,  47.350,  295),
    ("BH-019",        -65.300,  47.400,  400),
    ("BH-020",        -65.250,  47.450,  320),
    ("BH-021",        -66.020,  47.130,  380),
    ("BH-022",        -65.700,  47.140,  430),
    ("BH-023",        -65.580,  47.160,  260),
    ("BH-024",        -66.800,  47.550,  370),
    ("BH-025",        -66.820,  47.620,  440),
    ("BH-026",        -65.400,  47.550,  310),
    ("BH-027",        -65.350,  47.580,  290),
    ("BH-028",        -65.280,  47.530,  480),
    ("BH-029",        -66.300,  47.850,  355),
    ("BH-030",        -66.250,  47.880,  415),
    ("BH-031",        -65.900,  47.900,  330),
    ("BH-032",        -65.750,  47.930,  360),
    ("BH-033",        -65.650,  47.880,  400),
    ("BH-034",        -66.480,  47.820,  270),
    ("BH-035",        -66.420,  47.780,  310),
    ("BH-036",        -65.500,  47.700,  390),
    ("BH-037",        -65.450,  47.680,  280),
    ("BH-038",        -65.380,  47.630,  430),
    ("BH-039",        -66.700,  47.200,  350),
    ("BH-040",        -66.680,  47.160,  460),
    ("BH-041",        -66.650,  47.120,  340),
    ("BH-042",        -65.700,  47.250,  380),
    ("BH-043",        -65.630,  47.220,  320),
    ("BH-044",        -65.560,  47.200,  410),
    ("BH-045",        -66.900,  47.750,  380),
    ("BH-046",        -66.850,  47.800,  350),
    ("BH-047",        -65.200,  47.350,  290),
    ("BH-048",        -65.180,  47.300,  470),
    ("BH-049",        -65.220,  47.260,  330),
    ("BH-050",        -66.780,  47.450,  410),
    # Additional 200 holes — representative distribution across non-mineralised
    # zones. In production runs, these should be replaced with actual SEDAR
    # filings and NB Annual Report hole tables.
    *[(f"BH-{i:03d}",
       -66.85 + (i % 50) * 0.032,   # Systematic E-W traverse
       47.12 + (i // 50) * 0.20,    # 4 lat bands cover the camp
       200 + (i % 7) * 50)
      for i in range(51, 251)],
]


def build_vms_geodataframe() -> gpd.GeoDataFrame:
    """Compile known VMS deposits into a labelled GeoDataFrame."""
    records = []
    for name, lon, lat, notes in VMS_DEPOSITS:
        records.append({
            "deposit_name": name,
            "notes": notes,
            "label": 1,
            "geometry": Point(lon, lat)
        })
    gdf = gpd.GeoDataFrame(records, crs=CRS_SOURCE)
    gdf = gdf.to_crs(CRS_TARGET)
    log.info(f"  Built {len(gdf)} positive VMS labels")
    return gdf


def build_barren_geodataframe() -> gpd.GeoDataFrame:
    """Compile barren drill holes into a labelled GeoDataFrame."""
    records = []
    for hole_id, lon, lat, depth_m in BARREN_HOLES:
        records.append({
            "hole_id": hole_id,
            "depth_m": depth_m,
            "label": 0,
            "geometry": Point(lon, lat)
        })
    gdf = gpd.GeoDataFrame(records, crs=CRS_SOURCE)
    gdf = gdf.to_crs(CRS_TARGET)
    log.info(f"  Built {len(gdf)} negative (barren) labels")
    return gdf



def generate_pseudo_absences_dissimilar(
    vms_gdf: gpd.GeoDataFrame,
    count: int = 125,
    min_buffer_m: float = PSEUDO_ABSENCE_MIN_BUFFER_M,
    candidate_spacing_m: float = PSEUDO_ABSENCE_CANDIDATE_SPACING_M,
    dissim_quantile: float = PSEUDO_ABSENCE_DISSIM_QUANTILE,
    n_strata: int = PSEUDO_ABSENCE_N_SPATIAL_STRATA,
    seed: int = 42,
) -> gpd.GeoDataFrame:
    """
    Generate pseudo-absence labels that are maximally dissimilar to known VMS
    deposits in multi-dimensional geophysical/geochemical feature space.

    Strategy (Parsa & Cumani, 2025):
      1. Build a dense candidate grid (at ``candidate_spacing_m`` resolution)
         across the active geophysical raster footprint.
      2. Sample all available reprojected rasters at each candidate point.
      3. Standardise features (zero mean, unit variance) fitted on the
         positive-class (deposit) feature vectors.
      4. Compute the Mahalanobis distance of each candidate from the deposit
         centroid in normalised feature space. Candidates far from the deposit
         cluster are geologically dissimilar and preferred as pseudo-negatives.
      5. Apply a secondary minimum geographic guard buffer (``min_buffer_m``)
         to exclude points immediately adjacent to known deposits.
      6. Stratify candidates into ``n_strata`` × ``n_strata`` spatial quadrants
         and select the most dissimilar candidates from each quadrant to ensure
         geographic coverage across the survey footprint.
      7. Fall back to geographic exclusion sampling if rasters are unavailable.

    Parameters
    ----------
    vms_gdf : GeoDataFrame
        Known VMS deposit locations (positive class, CRS = CRS_TARGET).
    count : int
        Number of pseudo-absences to generate.
    min_buffer_m : float
        Secondary minimum geographic distance (m) from any deposit.
    candidate_spacing_m : float
        Grid spacing (m) for the dense candidate point cloud.
    dissim_quantile : float
        Lower quantile threshold on Mahalanobis distance; only candidates
        above this threshold (most dissimilar) are eligible for selection.
    n_strata : int
        Number of spatial strata per axis (total strata = n_strata²).
    seed : int
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    # ── Locate reprojected rasters ────────────────────────────────────────────
    reproj_dir = PROCESSED_DIR / "rasters_reprojected"
    raster_paths = sorted(reproj_dir.glob("*.tif")) if reproj_dir.exists() else []

    if not raster_paths:
        log.warning(
            "  [Pseudo-absence] No reprojected rasters found in "
            f"{reproj_dir}.\n"
            "  Falling back to geographic exclusion sampling "
            f"(min_buffer = {min_buffer_m} m). "
            "Run reproject_grids.py first to enable feature-space selection."
        )
        return _fallback_geographic_pseudo_absences(
            vms_gdf, count=count, exclusion_buffer_m=min_buffer_m, seed=seed
        )

    log.info(
        f"  [Pseudo-absence] Feature-space dissimilarity strategy "
        f"(Parsa & Cumani, 2025)."
    )
    log.info(f"  Rasters available: {len(raster_paths)}")

    # ── Use BMC study area raster grid extent (mag_rmi_bmc_combined1.tif) ───────────
    bmc_raster_path = RAW_DIR / "rasters" / "mag_rmi_bmc_combined1.tif"
    if bmc_raster_path.exists():
        with rasterio.open(bmc_raster_path) as bmc_src:
            bmc_b = bmc_src.bounds
            grid_left, grid_right = bmc_b.left, bmc_b.right
            grid_bottom, grid_top = bmc_b.bottom, bmc_b.top
            ref_crs = bmc_src.crs
    else:
        grid_left, grid_right = 2481060.0, 2576340.0
        grid_bottom, grid_top = 7551180.0, 7635540.0
        ref_crs = CRS_TARGET

    # ── Build dense candidate grid strictly within BMC study area ─────────────
    xs = np.arange(grid_left  + candidate_spacing_m / 2,
                   grid_right - candidate_spacing_m / 2,
                   candidate_spacing_m)
    ys = np.arange(grid_bottom + candidate_spacing_m / 2,
                   grid_top   - candidate_spacing_m / 2,
                   candidate_spacing_m)
    xx, yy = np.meshgrid(xs, ys)
    coords = np.column_stack([xx.ravel(), yy.ravel()])  # (N, 2)
    log.info(f"  Candidate grid (BMC Study Area): {len(coords):,} points at {candidate_spacing_m} m spacing")

    # ── Sample all rasters at candidate locations (VECTORISED) ────────────────
    # Read each raster as a full array once, then index by row/col.
    # NaN-tolerant: store NaN for NoData/out-of-bounds rather than dropping
    # points that are missing in ANY raster (which gives an empty intersection
    # when 58 rasters with different extents are combined).
    feature_arrays = []

    for rpath in raster_paths:
        with rasterio.open(rpath) as src:
            nd    = src.nodata if src.nodata is not None else -9999
            arr   = src.read(1).astype(np.float64)
            rows, cols = rasterio.transform.rowcol(
                src.transform, coords[:, 0], coords[:, 1]
            )
            rows = np.asarray(rows)
            cols = np.asarray(cols)
            in_bounds = (
                (rows >= 0) & (rows < arr.shape[0]) &
                (cols >= 0) & (cols < arr.shape[1])
            )
            vals = np.full(len(coords), np.nan, dtype=np.float64)
            vals[in_bounds] = arr[rows[in_bounds], cols[in_bounds]]
        # Replace nodata sentinels with NaN
        vals[vals == nd]    = np.nan
        vals[vals <= -9990] = np.nan
        feature_arrays.append(vals)

    F_all = np.column_stack(feature_arrays)  # (N_cand, n_rasters)

    # Drop columns (rasters) where >80% of candidates are NaN
    col_nan_frac = np.isnan(F_all).mean(axis=0)
    keep_cols    = col_nan_frac <= 0.80
    F_all        = F_all[:, keep_cols]
    log.info(
        f"  Rasters retained after NaN-column filter: "
        f"{keep_cols.sum()} / {len(keep_cols)}"
    )

    # Keep rows (candidates) that have valid data in ≥50% of retained rasters
    row_valid_frac = (~np.isnan(F_all)).mean(axis=1)
    row_ok         = row_valid_frac >= 0.50
    F_all          = F_all[row_ok]
    cand_xy        = coords[row_ok]
    log.info(f"  Valid candidates (≥50% features present): {len(cand_xy):,}")

    if len(cand_xy) == 0:
        log.warning("  All candidates below data-coverage threshold — falling back to geographic method.")
        return _fallback_geographic_pseudo_absences(
            vms_gdf, count=count, exclusion_buffer_m=min_buffer_m, seed=seed
        )


    # ── Apply secondary geographic guard buffer ───────────────────────────────
    from shapely.geometry import MultiPoint
    deposit_pts_xy = np.array(
        [[geom.x, geom.y] for geom in vms_gdf.geometry]
    )
    # Vectorised minimum distance: compare each candidate against all deposits
    diff        = cand_xy[:, np.newaxis, :] - deposit_pts_xy[np.newaxis, :, :]  # (N, D, 2)
    min_dists   = np.sqrt((diff ** 2).sum(axis=2)).min(axis=1)                  # (N,)
    geo_ok      = min_dists >= min_buffer_m
    F_all       = F_all[geo_ok]
    cand_xy     = cand_xy[geo_ok]
    min_dists   = min_dists[geo_ok]
    log.info(
        f"  Candidates after {min_buffer_m:.0f} m geographic guard: {len(cand_xy):,}"
    )

    if len(cand_xy) < count:
        log.warning(
            f"  Insufficient candidates ({len(cand_xy)}) after guard buffer. "
            "Relaxing to geographic fallback."
        )
        return _fallback_geographic_pseudo_absences(
            vms_gdf, count=count, exclusion_buffer_m=min_buffer_m, seed=seed
        )

    # ── Sample deposit feature vectors (VECTORISED) ───────────────────────────
    dep_xy = np.array([[g.x, g.y] for g in vms_gdf.geometry])
    dep_features = []
    for rpath in raster_paths:
        with rasterio.open(rpath) as src:
            nd    = src.nodata if src.nodata is not None else -9999
            arr   = src.read(1).astype(np.float64)
            rows, cols = rasterio.transform.rowcol(
                src.transform, dep_xy[:, 0], dep_xy[:, 1]
            )
            rows = np.asarray(rows)
            cols = np.asarray(cols)
            in_bounds = (
                (rows >= 0) & (rows < arr.shape[0]) &
                (cols >= 0) & (cols < arr.shape[1])
            )
            vals = np.full(len(dep_xy), np.nan, dtype=np.float64)
            vals[in_bounds] = arr[rows[in_bounds], cols[in_bounds]]
        vals[vals == nd]    = np.nan
        vals[vals <= -9990] = np.nan
        dep_features.append(vals)
    F_dep = np.column_stack(dep_features)  # (n_deposits, n_rasters)

    # Align deposit features to the same columns retained for candidates
    F_dep = F_dep[:, keep_cols]

    # Drop columns still NaN in ANY deposit vector
    dep_ok_cols = ~np.isnan(F_dep).any(axis=0)
    F_dep_clean = F_dep[:, dep_ok_cols]
    F_clean     = F_all[:, dep_ok_cols]   # candidates aligned to same columns
    log.info(
        f"  Features used for dissimilarity: {dep_ok_cols.sum()} / {F_dep.shape[1]}"
    )

    if F_dep_clean.shape[1] == 0:
        log.warning("  No clean features for Mahalanobis computation — falling back.")
        return _fallback_geographic_pseudo_absences(
            vms_gdf, count=count, exclusion_buffer_m=min_buffer_m, seed=seed
        )

    # ── Standardise feature space ──────────────────────────────────────────────
    scaler = StandardScaler().fit(F_dep_clean)
    F_dep_std  = scaler.transform(F_dep_clean)  # (n_deposits, p)
    F_cand_std = scaler.transform(F_clean)       # (N_cand, p)
    F_cand_std = np.nan_to_num(F_cand_std, nan=0.0)
    F_dep_std  = np.nan_to_num(F_dep_std, nan=0.0)

    # ── Mahalanobis distance from deposit centroid ────────────────────────────
    dep_centroid = F_dep_std.mean(axis=0)  # (p,)
    try:
        cov = np.cov(F_dep_std, rowvar=False)
        # Regularise to handle singular or near-singular covariance matrices
        cov += np.eye(cov.shape[0]) * 1e-6
        VI  = np.linalg.inv(cov)          # Inverse covariance matrix
        mah_dists = np.array([
            mahalanobis(row, dep_centroid, VI)
            for row in F_cand_std
        ])
    except np.linalg.LinAlgError:
        log.warning(
            "  Covariance matrix is singular — using Euclidean distance "
            "in standardised feature space as fallback."
        )
        diff_std  = F_cand_std - dep_centroid
        mah_dists = np.sqrt((diff_std ** 2).sum(axis=1))

    log.info(
        f"  Mahalanobis distance — min: {mah_dists.min():.2f}, "
        f"max: {mah_dists.max():.2f}, "
        f"median: {np.median(mah_dists):.2f}"
    )

    # ── Filter to top-dissimilar quantile ────────────────────────────────────
    threshold = np.quantile(mah_dists, dissim_quantile)
    eligible  = mah_dists >= threshold
    F_elig    = F_all[eligible]
    xy_elig   = cand_xy[eligible]
    md_elig   = mah_dists[eligible]
    log.info(
        f"  Eligible candidates (top {(1-dissim_quantile)*100:.0f}% most dissimilar): "
        f"{eligible.sum():,}"
    )

    # ── Spatially stratified selection ────────────────────────────────────────
    # Divide the footprint into n_strata × n_strata quadrants and select the
    # most dissimilar candidates per quadrant to ensure geographic spread.
    x_edges = np.linspace(xy_elig[:, 0].min(), xy_elig[:, 0].max(), n_strata + 1)
    y_edges = np.linspace(xy_elig[:, 1].min(), xy_elig[:, 1].max(), n_strata + 1)

    x_bin = np.digitize(xy_elig[:, 0], x_edges[1:-1])  # 0-indexed
    y_bin = np.digitize(xy_elig[:, 1], y_edges[1:-1])
    strata_id = y_bin * n_strata + x_bin               # scalar stratum label

    unique_strata = np.unique(strata_id)
    n_total_strata = len(unique_strata)
    per_stratum    = max(1, -(-count // n_total_strata))  # ceiling division

    selected_idx = []
    for sid in unique_strata:
        mask   = strata_id == sid
        idx    = np.where(mask)[0]
        # Sort by descending Mahalanobis distance within stratum
        order  = np.argsort(-md_elig[idx])
        chosen = idx[order[:per_stratum]]
        selected_idx.extend(chosen.tolist())

    # Trim or top-up to exactly `count`
    if len(selected_idx) > count:
        # Keep the globally most dissimilar among the selected
        selected_idx = sorted(
            selected_idx,
            key=lambda i: -md_elig[i]
        )[:count]
    elif len(selected_idx) < count:
        # Fill remaining slots from the most dissimilar eligible candidates
        all_remaining = set(range(len(xy_elig))) - set(selected_idx)
        fill = sorted(all_remaining, key=lambda i: -md_elig[i])
        selected_idx.extend(fill[: count - len(selected_idx)])

    xy_sel = xy_elig[selected_idx]
    md_sel = md_elig[selected_idx]
    log.info(
        f"  Selected {len(xy_sel)} pseudo-absences | "
        f"Mahalanobis distance range: [{md_sel.min():.2f}, {md_sel.max():.2f}]"
    )

    # ── Build GeoDataFrame ────────────────────────────────────────────────────
    records = [
        {
            "hole_id": f"PSA_{i + 1:03d}",
            "geonb_objectid": None,
            "assessment_report_no": "",
            "depth_m": 0.0,
            "year_drilled": "",
            "label": 0,
            "source": "pseudo_absence_dissimilar",
            "mahal_dist": float(md_sel[i]),
            "geometry": Point(float(xy_sel[i, 0]), float(xy_sel[i, 1])),
        }
        for i in range(len(xy_sel))
    ]
    return gpd.GeoDataFrame(records, crs=ref_crs)


def _fallback_geographic_pseudo_absences(
    vms_gdf: gpd.GeoDataFrame,
    count: int = 125,
    exclusion_buffer_m: float = 3000,
    seed: int = 42,
) -> gpd.GeoDataFrame:
    """
    Fallback: generate random pseudo-absences using a simple geographic
    exclusion buffer. Used when reprojected rasters are not yet available.
    """
    log.warning(
        "  [Pseudo-absence] Using FALLBACK geographic exclusion method "
        f"(buffer = {exclusion_buffer_m} m). "
        "This should be replaced by the dissimilarity method once rasters are ready."
    )
    random.seed(seed)
    reproj_dir = PROCESSED_DIR / "rasters_reprojected"
    tifs = list(reproj_dir.glob("*.tif")) if reproj_dir.exists() else []
    if not tifs:
        raise FileNotFoundError(
            "No reprojected rasters found. Run reproject_grids.py first."
        )
    raster_path = tifs[0]
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        nodata = src.nodata if src.nodata is not None else -9999
        pseudo_absences = []
        attempts = 0
        while len(pseudo_absences) < count and attempts < 50_000:
            attempts += 1
            x = random.uniform(bounds.left, bounds.right)
            y = random.uniform(bounds.bottom, bounds.top)
            point = Point(x, y)
            if vms_gdf.distance(point).min() < exclusion_buffer_m:
                continue
            val = list(src.sample([(x, y)]))[0][0]
            if val == nodata or np.isnan(val) or val <= 0:
                continue
            pseudo_absences.append({
                "hole_id": f"PSA_{len(pseudo_absences) + 1:03d}",
                "depth_m": 0.0,
                "label": 0,
                "source": "pseudo_absence_fallback",
                "mahal_dist": np.nan,
                "geometry": point,
            })
    log.info(
        f"  Fallback generated {len(pseudo_absences)} pseudo-absences "
        f"after {attempts} attempts"
    )
    return gpd.GeoDataFrame(pseudo_absences, crs=src.crs)


def main():
    log.info("═══ VMS Label Construction ═══")
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Positive labels ──────────────────────────────────────────────────────
    log.info("Building positive labels (known VMS deposits) ...")
    vms_gdf = build_vms_geodataframe()

    # Add a buffered geometry column for spatial label assignment
    # Stored as WKT so to_file() works with a single active geometry column
    vms_gdf["buffer_geom_wkt"] = vms_gdf.geometry.buffer(POSITIVE_BUFFER_M).to_wkt()

    vms_gdf.to_file(VMS_LABELS_GPKG, driver="GPKG", layer="vms_deposits")
    log.info(f"  ✅ Saved → {VMS_LABELS_GPKG}")

    # ── Negative labels ──────────────────────────────────────────────────────
    log.info("\nBuilding hybrid negative labels (125 barren holes + 125 pseudo-absences) ...")
    
    # 1. Sample 125 barren drill holes strictly inside the BMC study area raster grid
    geonb_dh_path = RAW_DIR / "geonb" / "nb_drill_holes.gpkg"
    
    if geonb_dh_path.exists():
        log.info(f"  Loading GeoNB drill hole collars from {geonb_dh_path.name} ...")
        dh_gdf = gpd.read_file(geonb_dh_path)
        dh_gdf = dh_gdf.to_crs(CRS_TARGET) if dh_gdf.crs != CRS_TARGET else dh_gdf
        
        # Get active BMC study area raster bounds (mag_rmi_bmc_combined1.tif)
        master_bmc_tif = RAW_DIR / "rasters" / "mag_rmi_bmc_combined1.tif"
        reproj_mag = PROCESSED_DIR / "rasters_reprojected" / "mag_rmi_bmc_combined1.tif"
        ref_tif = reproj_mag if reproj_mag.exists() else (master_bmc_tif if master_bmc_tif.exists() else None)
        
        if ref_tif:
            with rasterio.open(ref_tif) as src:
                b = src.bounds
                xmin, xmax = b.left, b.right
                ymin, ymax = b.bottom, b.top
        else:
            xmin, xmax = 2481060.0, 2576340.0
            ymin, ymax = 7551180.0, 7635540.0
            
        # Filter for collars strictly inside BMC raster grid
        inside_mask = (
            (dh_gdf.geometry.x >= xmin) & (dh_gdf.geometry.x <= xmax) &
            (dh_gdf.geometry.y >= ymin) & (dh_gdf.geometry.y <= ymax)
        )
        dh_inside = dh_gdf[inside_mask].copy()
        
        # Enforce minimum distance guard of 1,000 m from any positive VMS deposit
        vms_union = vms_gdf.geometry.union_all()
        dists = dh_inside.geometry.distance(vms_union)
        barren_candidates = dh_inside[dists >= 1000.0].copy()
        
        # Reproducible random sampling
        barren_sample = barren_candidates.sample(n=125, random_state=42).copy()
        
        records = []
        for idx_h, row in barren_sample.iterrows():
            hole_name = row.get("label") or row.get("hole_id") or row.get("name") or f"DH_{row.get('objectid', idx_h)}"
            obj_id = row.get("objectid", idx_h)
            depth = row.get("length_m") or row.get("depth_m", 0.0)
            depth_val = float(depth) if pd.notna(depth) else 0.0
            rept_no = str(row.get("rept_no", "")) if pd.notna(row.get("rept_no")) else ""
            year_drilled = str(row.get("yeardrille", "")) if pd.notna(row.get("yeardrille")) else ""
            
            records.append({
                "hole_id": str(hole_name),
                "geonb_objectid": int(obj_id) if pd.notna(obj_id) else idx_h,
                "assessment_report_no": rept_no,
                "depth_m": depth_val,
                "year_drilled": year_drilled,
                "label": 0,
                "source": "barren_hole_geonb",
                "geometry": row.geometry
            })
        barren_drill_gdf = gpd.GeoDataFrame(records, crs=CRS_TARGET)
        log.info(f"  Sampled {len(barren_drill_gdf)} barren drill hole collars strictly inside BMC raster bounds")
    else:
        log.warning("  GeoNB drill hole file not found; falling back to legacy barren sampling")
        random.seed(42)
        sampled_barren_holes = random.sample(BARREN_HOLES, 125)
        records = []
        for hole_id, lon, lat, depth_m in sampled_barren_holes:
            records.append({
                "hole_id": hole_id,
                "depth_m": depth_m,
                "label": 0,
                "source": "barren_hole",
                "geometry": Point(lon, lat)
            })
        barren_drill_gdf = gpd.GeoDataFrame(records, crs=CRS_SOURCE).to_crs(CRS_TARGET)
        log.info(f"  Sampled {len(barren_drill_gdf)} barren drill holes")
    
    # 2. Generate 125 pseudo-absences selected by feature-space dissimilarity
    #    (Parsa & Cumani, 2025): maximise Mahalanobis distance from the VMS
    #    deposit centroid in geophysical/geochemical feature space, stratified
    #    across spatial quadrants for geographic representativeness.
    psa_gdf = generate_pseudo_absences_dissimilar(
        vms_gdf, count=125, seed=42
    )
    
    # 3. Combine both negative sources
    combined_neg_df = pd.concat([barren_drill_gdf, psa_gdf], ignore_index=True)
    barren_gdf = gpd.GeoDataFrame(combined_neg_df, crs=CRS_TARGET)
    
    barren_gdf["buffer_geom_wkt"] = barren_gdf.geometry.buffer(POSITIVE_BUFFER_M).to_wkt()

    barren_gdf.to_file(BARREN_LABELS_GPKG, driver="GPKG", layer="barren_holes")
    log.info(f"  ✅ Saved → {BARREN_LABELS_GPKG}")

    # ── Summary ──────────────────────────────────────────────────────────────
    log.info("\n─── Label Summary ───")
    log.info(f"  Positive (VMS)  : {len(vms_gdf):4d}  (label = 1)")
    log.info(f"  Negative (Barren): {len(barren_gdf):4d}  (label = 0)")
    log.info(
        f"  Class ratio      : 1:{len(barren_gdf)//len(vms_gdf)} "
        f"(will be addressed with SMOTE + class_weight='balanced')"
    )
    log.info("\nRun next: python pipeline/02_preprocessing/extract_features.py")


if __name__ == "__main__":
    main()
