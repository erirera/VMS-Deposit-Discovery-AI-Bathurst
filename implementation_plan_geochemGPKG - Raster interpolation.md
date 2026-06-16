# Implementation Plan: Geochem GPKG → Raster Interpolation

## Goal
Write a new preprocessing script (`pipeline/02_preprocessing/interpolate_geochem.py`) that reads each of the 17 `bmc_*.gpkg` till geochemistry GeoPackage files and interpolates them into **GeoTIFF raster grids**. These grids will be saved into `data/processed/rasters_reprojected/` so that the existing `extract_features.py` can automatically sample them alongside the geophysical layers.

## Background

The current pipeline's `extract_features.py` samples features from **all `.tif` files** found in `data/processed/rasters_reprojected/`. Once each geochemistry element is converted to a raster, it will automatically be included in the feature matrix — no further code changes are needed.

**Key parameters matched to existing raster grid:**
- **CRS:** `EPSG:2953` (NAD83(CSRS) / NB Double Stereographic)
- **Resolution:** `50 m` per pixel (matched to existing derivatives)
- **Extent:** Geochemistry point cloud bounds: `[2481231, 7550777, 2576103, 7635031]`
- **NoData value:** `-9999` (from `config.py`)
- **Interpolation method:** Inverse Distance Weighting (IDW) — fast, deterministic, no extra dependencies

## Proposed Changes

---

### Pipeline Preprocessing

#### [NEW] [interpolate_geochem.py](file:///c:/Users/delef/.gemini/antigravity/bathurst%20VMS/VMS-Deposit-Discovery-AI-Bathurst/pipeline/02_preprocessing/interpolate_geochem.py)

New script that:
1. Iterates over all `bmc_*.gpkg` files in `data/raw/rasters/`
2. Reads the `VALUE` column (element concentration in PPM) and `geometry` (point location)
3. Applies **IDW interpolation** (power=2) to create a continuous surface on a 50m grid
4. Writes the result as a GeoTIFF to `data/processed/rasters_reprojected/geochem_<ELEMENT>.tif`
5. Skips any file where the output already exists (safe to re-run)

**Output files (17 total):**

| Input GPKG | Output GeoTIFF |
| :--- | :--- |
| `bmc_Ag.gpkg` | `geochem_ag_ppm.tif` |
| `bmc_As.gpkg` | `geochem_as_ppm.tif` |
| `bmc_Ba.gpkg` | `geochem_ba_ppm.tif` |
| `bmc_Bi.gpkg` | `geochem_bi_ppm.tif` |
| `bmc_Cd.gpkg` | `geochem_cd_ppm.tif` |
| `bmc_Co.gpkg` | `geochem_co_ppm.tif` |
| `bmc_Cu.gpkg` | `geochem_cu_ppm.tif` |
| `bmc_Fe.gpkg` | `geochem_fe_ppm.tif` |
| `bmc_In.gpkg` | `geochem_in_ppm.tif` |
| `bmc_Mn.gpkg` | `geochem_mn_ppm.tif` |
| `bmc_Mo.gpkg` | `geochem_mo_ppm.tif` |
| `bmc_Ni.gpkg` | `geochem_ni_ppm.tif` |
| `bmc_Pb.gpkg` | `geochem_pb_ppm.tif` |
| `bmc_Sb.gpkg` | `geochem_sb_ppm.tif` |
| `bmc_Sn.gpkg` | `geochem_sn_ppm.tif` |
| `bmc_Tl.gpkg` | `geochem_tl_ppm.tif` |
| `bmc_Zn.gpkg` | `geochem_zn_ppm.tif` |

---

### Config Update

#### [MODIFY] [config.py](file:///c:/Users/delef/.gemini/antigravity/bathurst%20VMS/VMS-Deposit-Discovery-AI-Bathurst/pipeline/config.py)

Add the 17 new geochemistry raster feature names to `RASTER_FEATURES` so they are documented in the feature registry.

> [!NOTE]
> The `extract_features.py` script already samples **all** `.tif` files it finds in the rasters directory. Adding the names to `config.py` is just for documentation and future reference — no functional change is required.

---

## Verification Plan

### Automated
- Run `python pipeline/02_preprocessing/interpolate_geochem.py` and confirm 17 `.tif` files are created in `data/processed/rasters_reprojected/`.
- Run `python pipeline/02_preprocessing/extract_features.py` and confirm the feature matrix now includes the new geochem raster columns.

### Manual
- Check one output raster (e.g. `geochem_cu_ppm.tif`) by loading it in QGIS or with Python and verifying the value range matches the original GPKG data (Cu: 0 – 1,900 PPM).

## Open Questions

> [!IMPORTANT]
> **Log-transform before rasterizing?** Element distributions like As (0–110,000 PPM) and Sn (0–100,000 PPM) are extremely right-skewed. Should we save the raw PPM values in the raster, or apply a `log10(x + 1)` transform first to normalize the distribution for the ML model?
>
> **Recommendation:** Save **raw PPM** in the rasters (preserving scientific meaning), and handle log-transform in `engineer_features.py` as part of feature engineering.

> [!NOTE]
> **IDW neighbourhood size:** Using all points for IDW on a large dataset is slow. The script will use a `k=12` nearest-neighbour IDW, which is standard practice and much faster. Kriging would produce better spatial estimates but requires additional libraries (`pykrige`) that are not in `requirements.txt`.
