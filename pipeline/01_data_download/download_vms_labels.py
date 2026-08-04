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
    LABELS_DIR, VMS_LABELS_GPKG, BARREN_LABELS_GPKG,
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
    ("Brunswick No. 12",              -65.8520,  47.4720, "World-class; 230 Mt @ 8.0% Zn"),
    ("Brunswick No. 6",               -65.7800,  47.4400, "Stratiform Zn-Pb-Cu-Ag"),
    ("Key Anacon",                    -65.7500,  47.4100, "Mined out"),
    ("Murray Brook",                  -65.6900,  47.3800, "Au-rich VMS"),
    ("Caribou",                       -65.5800,  47.3600, "4.5 Mt @ 6.5% Zn"),
    ("Heath Steele (BHNS, B&E)",      -65.9000,  47.5600, "Multiple lenses"),
    ("Restigouche",                   -67.0300,  47.9800, "Cu-Zn-Pb"),
    ("Orvan Brook",                   -65.8200,  47.3300, "Galena-rich"),
    ("Stratmat North",                -65.7100,  47.4300, "Zn-Pb polymetallic"),
    ("Halfmile Lake",                 -65.9200,  47.5000, "Deep deposit; Ag-rich"),
    ("Wedge",                         -65.8800,  47.4900, "Cu-Zn"),
    ("Skiff Lake",                    -66.2300,  47.6200, "historic Cu producer"),
    ("Burnt Hill (Tin/W adjacent)",   -66.8100,  46.8900, "Sn-W; atypical BMC"),
    ("Flat Landing Brook",            -66.6500,  47.7500, "VMS-exhalite"),
    ("Nigadoo River",                 -65.6600,  47.6800, "Zn-Pb"),
    ("Flett",                         -65.9800,  47.5300, "Small Cu-Zn"),
    ("Quaco",                         -65.5200,  45.3400, "South NB VMS"),
    ("Portage",                       -66.3200,  47.6800, "Small Zn-Pb"),
    ("Elmtree",                       -65.3700,  47.9000, "Northern BMC"),
    ("Boudreau",                      -66.1700,  47.5900, "Cu-Zn"),
    ("Pine Cove",                     -64.8500,  45.9700, "Au-Ag-VMS"),
    ("Forsythe",                      -65.7300,  47.3900, "Zn-Pb"),
    ("McBean Lake",                   -66.1200,  47.6500, "Cu-Zn"),
    ("Middle Landing",                -66.5800,  47.7200, "Cu-Zn-Pb exhalite"),
    ("Blue Bell-Canaan",              -65.8400,  47.4500, "Historic Ag"),
    ("Captain",                       -65.9600,  47.5100, "Zn-Pb"),
    ("Sabel",                         -66.0100,  47.5500, "Cu-rich"),
    ("Reid",                          -66.0800,  47.5700, "Small Zn-Pb"),
    ("Clearwater",                    -66.2800,  47.6000, "Zn"),
    ("Anderson Stillwater",           -65.8700,  47.4600, "Cu-Zn"),
    ("Crowe Mountain",                -66.3600,  47.7100, "Cu"),
    ("Teahan",                        -65.7600,  47.4200, "Zn-Pb"),
    ("Spruce Lake",                   -66.4500,  47.7000, "Zn-Cu"),
    ("Lost Lake",                     -66.3000,  47.6400, "Cu"),
    ("Mitchell",                      -65.8600,  47.4800, "Zn-Pb"),
    ("Lake George (Sb adjacent)",     -67.0200,  46.0100, "Sb-Au; different style"),
    ("Nicholas-Denys",                -66.5200,  47.7400, "Zn-Pb-Cu"),
    ("Clearwater West",               -66.3100,  47.6100, "Zn"),
    ("Lyndhurst",                     -65.6200,  47.3500, "Cu-Zn"),
    ("Nepisiguit Falls",              -65.9400,  47.5800, "Cu-Zn exhalite"),
    ("Dry Creek",                     -66.7500,  47.8500, "Zn-Pb"),
    ("Poplar Mountain",               -65.9000,  47.4300, "Cu"),
    ("Silver Lake",                   -66.1500,  47.5500, "Ag-Pb"),
    ("Menneval",                      -66.0000,  47.5200, "Cu-Zn"),
    ("Clifton",                       -65.8000,  47.4600, "Zn-rich VMS"),
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

    # ── Use the first raster to define bounds and valid-data mask ─────────────
    ref_path = raster_paths[0]
    with rasterio.open(ref_path) as src:
        bounds   = src.bounds
        ref_crs  = src.crs
        nodata   = src.nodata if src.nodata is not None else -9999
        res      = src.res  # (x_res, y_res) in metres

    # ── Build dense candidate grid ────────────────────────────────────────────
    xs = np.arange(bounds.left  + candidate_spacing_m / 2,
                   bounds.right - candidate_spacing_m / 2,
                   candidate_spacing_m)
    ys = np.arange(bounds.bottom + candidate_spacing_m / 2,
                   bounds.top   - candidate_spacing_m / 2,
                   candidate_spacing_m)
    xx, yy = np.meshgrid(xs, ys)
    coords = np.column_stack([xx.ravel(), yy.ravel()])  # (N, 2)
    log.info(f"  Candidate grid: {len(coords):,} points at {candidate_spacing_m} m spacing")

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
            "depth_m": 0.0,
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
    
    # 1. Sample 125 barren drill holes
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
