# Data Acquisition Guide
## VMS Deposit Discovery AI — Bathurst Mining Camp
### Manual Download Instructions for All Geoscience Datasets

> All datasets listed here are **free and open access** — no account registration required.
> Total data cost: **$0**. Estimated total download size: ~2–4 GB.

---

## Quick Summary — What to Download

| Dataset | Source Portal | Page |
|---|---|---|
| Till Geochemistry (Zn, Pb, Cu, Ag, Au, As) | NB Geological Survey Open Data | [Link ↓](#1-till-geochemistry--nb-geological-survey) |
| Aeromagnetic — Compiled province-wide grid | NB DNRED Open Data | [Link ↓](#2-aeromagnetic-data--nb-dnred) |
| Aeromagnetic — Individual surveys (BMC area) | NB DNRED Open Data | [Link ↓](#2-aeromagnetic-data--nb-dnred) |
| Radiometric (K%, Th, U) — Compiled | NB DNRED Open Data | [Link ↓](#3-radiometric-data--nb-dnred) |
| Gravity Gradiometry | NB DNRED Open Data | [Link ↓](#4-gravity-gradiometry--nb-dnred) |
| VMS Deposit Locations (Metallic Minerals DB) | NB Geological Survey Open Data | [Link ↓](#5-metallic-minerals-database--vms-deposit-locations) |
| Bedrock Geology (structural context) | NB Geological Survey Open Data | [Link ↓](#6-bedrock-geology--optional-context-layer) |
| NRCan Aeromagnetic GDR (national archive) | NRCan Geoscience Data Repository | [Link ↓](#7-nrcan-geoscience-data-repository-gdr--national-archive) |

---

## Where to Put Downloaded Files

```
VMS-Deposit-Discovery-AI-Bathurst/
└── data/
    └── raw/
        ├── rasters/          ← All GeoTIFF/ASCII raster files go here
        │   ├── mag_tmi_nb_compiled.tif
        │   ├── rad_k_nb_compiled.tif
        │   ├── rad_th_nb_compiled.tif
        │   ├── rad_u_nb_compiled.tif
        │   └── gravity_bouguer_nb.tif
        ├── labels/           ← Auto-created by download_vms_labels.py
        └── nb_till_geochemistry_raw.csv   ← Geochemistry download
```

After placing files, **rename them to match** the names in `config.py` (see section 8 below), then run:

```powershell
python pipeline/02_preprocessing/reproject_grids.py
```

---

## Dataset-by-Dataset Instructions

---

### 1. Till Geochemistry — NB Geological Survey

**Portal:** NB Natural Resources and Energy Development — GIS Open Data
**Direct page:** https://www2.gnb.ca/content/gnb/en/departments/erd/open-data/geochemistry.html

**Steps:**
1. Navigate to the **Till Geochemistry** download page above
2. Click the **Download** button (CSV or Shapefile format — use **CSV** for this pipeline)
3. Save the file as `data/raw/nb_till_geochemistry_raw.csv`

**What you get:**
- Multi-element ICP-MS analyses of glacial till samples across NB
- Key columns: `LONGITUDE`, `LATITUDE`, `ZN_PPM`, `PB_PPM`, `CU_PPM`, `AG_PPM`, `AU_PPB`, `AS_PPM`
- ~5,000–15,000 sample locations province-wide

**Notes:**
- The `download_nb_geochemistry.py` script will also attempt this automatically
- If the script fails, this is the manual fallback
- No registration required

---

### 2. Aeromagnetic Data — NB DNRED

**Portal:** NB Natural Resources and Energy Development — GIS Open Data
**Parent page:** https://www2.gnb.ca/content/gnb/en/departments/erd/open-data/geophysical-data.html

Two sub-datasets available:

#### 2a. Compiled Province-Wide Grid (Recommended — start here)

**Page:** https://www2.gnb.ca/content/gnb/en/departments/erd/open-data/geophysical-data/aeromagnetic-data-compiled.html

**Steps:**
1. Navigate to the Compiled Aeromagnetic Data page above
2. Select the **TMI (Total Magnetic Intensity)** grid — download as **GeoTIFF** (preferred) or ASCII grid
3. Also download the **First Vertical Derivative (FVD)** grid if available separately
4. Save to `data/raw/rasters/`

**Why this is the best starting point:**
- Province-wide unified compilation at consistent resolution
- Already projected into a provincial CRS (may need reprojection to EPSG:2953)
- Covers the full BMC extent in a single file

#### 2b. Individual Surveys (for higher local resolution in the BMC)

**Page:** https://www2.gnb.ca/content/gnb/en/departments/erd/open-data/geophysical-data/aeromagnetic-data-individual.html

**Steps:**
1. Navigate to the Individual Surveys page
2. Look for surveys covering the Bathurst area (northern NB, ~47.0°N–48.0°N, 65.5°W–67.0°W)
   - Look for survey names referencing **Bathurst**, **Tetagouche**, **Nepisiguit**, **Gloucester**, or **Restigouche** counties
3. Download the TMI and FVD grids for each relevant survey
4. Save to `data/raw/rasters/`, using suffixes to identify surveys (e.g., `mag_tmi_bathurst_survey_1.tif`)

**Notes:**
- Individual surveys often have higher resolution (50m vs. 200m for the compiled grid)
- The BMC area was surveyed multiple times — multiple overlapping datasets can be merged or used individually

**Rename downloaded files to:**
```
mag_tmi_nb_compiled.tif    (or mag_tmi_nb_2013.tif for individual surveys)
mag_fvd_nb_compiled.tif
```

---

### 3. Radiometric Data — NB DNRED

**Portal:** NB Natural Resources and Energy Development — GIS Open Data
**Parent page:** https://www2.gnb.ca/content/gnb/en/departments/erd/open-data/geophysical-data.html

#### 3a. Compiled Province-Wide Grid (Recommended)

**Page:** https://www2.gnb.ca/content/gnb/en/departments/erd/open-data/geophysical-data/radiometric-data-compiled.html

**Steps:**
1. Navigate to the Compiled Radiometric Data page
2. Download the following individual channel grids:
   - **Potassium (K%)** → save as `rad_k_nb_compiled.tif`
   - **Thorium (Th ppm)** → save as `rad_th_nb_compiled.tif`
   - **Uranium (U ppm)** → save as `rad_u_nb_compiled.tif`
   - **Total Count (TC)** → optional, save as `rad_tc_nb_compiled.tif`
3. Save all to `data/raw/rasters/`

**Geological significance for VMS targeting:**
- **K/Th ratio**: elevated values → potassic footwall alteration halos beneath VMS systems
- **U enrichment**: can indicate hydrothermal fluid pathways
- The compiled grid has province-wide coverage at harmonised resolution

#### 3b. Individual Radiometric Surveys

**Page:** https://www2.gnb.ca/content/gnb/en/departments/erd/open-data/geophysical-data/radiometric-data-individual.html

Same area-selection procedure as for aeromagnetic individual surveys (§2b).

---

### 4. Gravity Gradiometry — NB DNRED

**Page (Compiled):** https://www2.gnb.ca/content/gnb/en/departments/erd/open-data/geophysical-data/gravity-gradiometry-compiled.html
**Page (Individual):** https://www2.gnb.ca/content/gnb/en/departments/erd/open-data/geophysical-data/gravity-gradiometry-individual.html

**Steps:**
1. Navigate to the Compiled Gravity Gradiometry page
2. Download the **Bouguer anomaly** grid (or the available gravity product)
3. Save as `data/raw/rasters/gravity_bouguer_nb.tif`

**Notes:**
- Gravity data may be available as a gradiometry product (Full Tensor Gravity) rather than a classical Bouguer grid
- If Bouguer anomaly is not directly available, use the **Gz (vertical gradient)** component as a proxy
- Density contrast from massive sulphide lenses (~4.0–4.5 g/cm³) vs. host rock (~2.7 g/cm³) creates a detectable gravity signal

---

### 5. Metallic Minerals Database — VMS Deposit Locations

**Portal:** NB Geological Survey — GIS Open Data
**Page:** https://www2.gnb.ca/content/gnb/en/departments/erd/open-data/metallic-minerals.html

**Steps:**
1. Navigate to the Metallic Minerals page
2. Download the dataset (Shapefile or CSV format)
3. The file will contain all metallic mineral occurrences in NB, including VMS deposits

**How it's used in the pipeline:**
- Filter by `COMMODITY_TYPE = 'Zn-Pb'` or `DEPOSIT_TYPE = 'VMS'` or similar field
- The 45 hardcoded deposit locations in `download_vms_labels.py` were compiled from this database
- You can use the downloaded database to verify, update, or expand the deposit list

**To use in the pipeline:**
- Load in QGIS or with geopandas to inspect fields
- Export filtered VMS deposits as CSV and place in `data/labels/`
- The pipeline's `download_vms_labels.py` already has the 45 deposits hardcoded, so this step is optional but recommended for QC

---

### 6. Bedrock Geology — Optional Context Layer

**Portal:** NB Geological Survey — GIS Open Data
**Page:** https://www2.gnb.ca/content/gnb/en/departments/erd/open-data/bedrock-geology.html

**Steps:**
1. Navigate to the Bedrock Geology page
2. Download the province-wide bedrock geology shapefile
3. Save to `data/raw/` (used for visualisation and geological validation, not as a ML feature in Phase 1)

**Notes:**
- Not used as a raster feature in the current pipeline (categorical data — would require one-hot encoding)
- Useful for overlaying on the prospectivity map to confirm that high-probability zones correlate with known VMS-hosting formations (Tetagouche Group, Tobique Group)

---

### 7. NRCan Geoscience Data Repository (GDR) — National Archive

The GDR is the **national** archive for airborne geophysical surveys, hosted by Natural Resources Canada. It complements the provincial NB DNRED data above, and includes older historical surveys not yet in the provincial portal.

**Main portal:** https://gdr.agg.nrcan.gc.ca/gdrdap/dap/portal
**Alternative URL:** https://geophysical-data.canada.ca/

**Steps to find Bathurst-area surveys:**

1. Navigate to: https://gdr.agg.nrcan.gc.ca/gdrdap/dap/portal
2. Click **"Search by Province"** → select **New Brunswick**
   - Or use **"Search by Map"** to draw a bounding box over the BMC (~47°N–48°N, 65.5°W–67°W)
3. Browse the survey list. Relevant surveys to look for:
   - Surveys flown by the **Geological Survey of Canada (GSC)** over northern NB
   - Look for surveys in the **Bathurst** or **Nepisiguit** area
   - Survey names may include: "NB-XX", "Bathurst Lake", "Heath Steele", "Tetagouche"
4. Click on each relevant survey → review the metadata (flight line spacing, altitude, acquisition year)
5. Click **"Download"** → select **GeoTIFF** format (preferred) or **ASCII grid**
   - If only binary Geosoft (`.grd`, `.ers`) format is available, see note below
6. Save downloaded files to `data/raw/rasters/`

> [!NOTE]
> **Geosoft format note:** If NRCan only provides grids in Geosoft `.grd` / `.ers` format and no GeoTIFF is available, you have two options:
> - Use **QGIS** (free): `Raster → Miscellaneous → Convert` to re-export as GeoTIFF
> - Use Python with **`osgeo.gdal`**: `gdal_translate -of GTiff input.grd output.tif`
> You do NOT need an Oasis Montaj license to read and convert these files with GDAL.

**Open Government Portal (alternative NRCan search):**
- URL: https://open.canada.ca/en/open-data
- Search terms: `"New Brunswick" aeromagnetic geophysical survey`
- Filter by: Organisation = "Natural Resources Canada"

---

### 8. File Naming Convention

After downloading, rename files to match the names expected by `config.py`:

| Pipeline expects | Rename your file to |
|---|---|
| `mag_tmi_nb_compiled.tif` | Province-wide TMI compiled grid |
| `mag_fvd_nb_compiled.tif` | Province-wide FVD compiled grid |
| `rad_k_nb_compiled.tif` | Potassium (K%) radiometric grid |
| `rad_th_nb_compiled.tif` | Thorium radiometric grid |
| `rad_u_nb_compiled.tif` | Uranium radiometric grid |
| `gravity_bouguer_nb.tif` | Bouguer anomaly or gravity gradiometry |

Place all in: `data/raw/rasters/`

---

### 9. After Downloading — Run the Pipeline

```powershell
# 1. Build label GeoPackages (works offline, no downloads needed)
python pipeline/01_data_download/download_vms_labels.py

# 2. Download till geochemistry (attempts automatic; manual fallback = step 1 above)
python pipeline/01_data_download/download_nb_geochemistry.py

# 3. Attempt automated raster downloads (fallback = manual steps above)
python pipeline/01_data_download/download_nrcan_mag.py

# 4. Reproject all rasters in data/raw/rasters/ to EPSG:2953 @ 100m
python pipeline/02_preprocessing/reproject_grids.py

# 5. Extract geophysical values at label point locations
python pipeline/02_preprocessing/extract_features.py

# 6. Derive secondary features (HGM, analytic signal, K/Th ratios, log-geochem)
python pipeline/02_preprocessing/engineer_features.py

# 7. Assemble training dataset with SMOTE and spatial CV blocks
python pipeline/03_training/build_dataset.py

# 8. Train Random Forest
python pipeline/03_training/train_rf.py

# 9. Train XGBoost
python pipeline/03_training/train_xgb.py

# 10. Compare models and generate evaluation figures
python pipeline/03_training/evaluate_models.py

# 11. Generate full BMC prospectivity map
python pipeline/04_prospectivity_map/predict_full_extent.py

# 12. Export publication-ready map
python pipeline/04_prospectivity_map/export_map.py

# 13. SHAP explainability analysis
python pipeline/05_explainability/shap_analysis.py
```

---

### 10. Contact Information — Data Questions

**NB Geological Survey Branch (geophysical and geochemical data):**
- Location: 670 King Street, Fredericton, NB
- Phone: 506-453-2206
- Email: minerals@gnb.ca
- Web: https://www2.gnb.ca/content/gnb/en/departments/erd/energy/content/minerals.html

**NRCan Geoscience Data Repository support:**
- Email: gdr@nrcan.gc.ca
- Web: https://gdr.agg.nrcan.gc.ca/

---

*Last verified: April 2026. All URLs are live NB Government and NRCan sources — no registration required for access.*
