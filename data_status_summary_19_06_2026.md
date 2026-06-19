# Bathurst VMS Pipeline — Data Status Summary

## 1. Data Categories

The pipeline uses **four categories** of input data, all covering the Bathurst Mining Camp (BMC), New Brunswick.

---

### 🧲 A. Geophysics — Magnetics
| File(s) | What it represents |
|---|---|
| `mag_rmi_bmc_combined*.gpkg / .tif` | Total Magnetic Intensity (TMI) — raw aeromagnetic survey |
| `mag_rmi_fvd_bmc` | First Vertical Derivative — enhances shallow magnetic sources |
| `mag_rmi_thg_bmc` | Total Horizontal Gradient — edge detection of magnetic bodies |
| `mag_rmi_as_bmc` | Analytic Signal Amplitude — depth-independent magnetic response |
| `mag_rmi_tdr_bmc` | Tilt Derivative — normalises signal over shallow/deep sources |
| `mag_rmi_thdr_bmc` | Tilt Horizontal Gradient |
| `mag_rmi_svd_bmc` | Second Vertical Derivative |

> **Geological purpose:** VMS deposits are typically hosted in volcanic rocks with distinct magnetic signatures. Derivative transforms highlight structural corridors, faults, and hydrothermal alteration zones.

---

### 🌍 B. Geophysics — Gravity & Radiometrics
| File(s) | What it represents |
|---|---|
| `gra_ggr_bmc_combined*.tif` | Bouguer gravity anomaly — bulk density contrasts |
| `gra_ggr_hgm_bmc` | Horizontal Gradient Magnitude |
| `gra_ggr_tdr_bmc` | Tilt Derivative (gravity) |
| `gra_ggr_fvd_bmc` | First Vertical Derivative (gravity) |
| `gra_ggr_svd_bmc` | Second Vertical Derivative |
| `gra_ggr_as_bmc` | Analytic Signal |
| `gra_ggr_uc500_bmc` | Upward Continued 500 m — regional trend |
| `gra_ggr_res_bmc` | Residual Bouguer — local anomalies after regional removal |
| `rad_k_bmc` | Potassium % — marker of potassic alteration halos |
| `rad_th_bmc` | Thorium (ppm) |
| `rad_u_bmc` | Uranium (ppm) |
| `rad_k_th_bmc` | K/Th ratio — potassic alteration indicator |
| `rad_u_th_bmc` | U/Th ratio |
| `rad_th_k_bmc` | Th/K ratio |
| `rad_u_k_bmc` | U/K ratio |
| `rad_dose_bmc` | Total dose rate |

---

### ⚗️ C. Till Geochemistry (Point Samples → Continuous Surfaces)
**Raw point data** (`data/raw/rasters/bmc_*.gpkg`) for **17 elements**:

| Element | Significance for VMS |
|---|---|
| Zn, Pb, Cu | Primary VMS ore metals — key pathfinders |
| Ag | Silver — close-range VMS indicator |
| As, Sb | Alteration zone pathfinders |
| Cd, In | VMS-associated trace metals |
| Fe, Mn, Co, Ni | Mafic volcanic/hydrothermal indicators |
| Ba | Barite cap rock indicator |
| Bi, Tl, Sn, Mo | Distal-to-proximal pathfinder suite |

**Interpolated surfaces** (in `data/processed/rasters_reprojected/`) exist in **two versions**:
- `geochem_*_idw.tif` — Inverse Distance Weighting (fast, default)
- `geochem_*_kriging.tif` — Ordinary Kriging (geostatistically rigorous)

**Direct-sample pathfinder features** used in training: `Zn, Pb, Cu, Ag, Au, As` (ppm/ppb)

---

### 🏷️ D. Training Labels (Ground Truth)
| File | Label | Description |
|---|---|---|
| `vms_positive_labels.gpkg` | `1` | Confirmed VMS deposit locations (from NB Metallic Minerals DB) |
| `barren_negative_labels.gpkg` | `0` | Drill holes that are confirmed mineralisation-free |

**Supporting vector data:**
- `nb_mineral_occurrences.gpkg` — broader mineralisation context
- `nb_drill_holes.gpkg` — all NB drill hole locations
- `studyArea.shp` — BMC study area boundary polygon

---

## 2. What Has Been Done (Preprocessing Completed ✅)

The pipeline has progressed through **all five preprocessing sub-steps**:

```
raw data → reproject → derive geophysics → interpolate geochem → extract features → engineer features → build training dataset
```

| Step | Script | Status | Output |
|---|---|---|---|
| **2a. Reproject Grids** | `reproject_grids.py` | ✅ Done | All rasters in `data/processed/rasters_reprojected/` at EPSG:2953, 100 m resolution |
| **2b. Compute Mag Derivatives** | `compute_mag_derivatives.py` | ✅ Done | THG, AS, TDR, THDR, SVD `.tif` files created |
| **2c. Compute Grav Derivatives** | `compute_grav_derivatives.py` | ✅ Done | HGM, TDR, FVD, SVD, AS, UC500, residual `.tif` files created |
| **2d. Compute Rad Derivatives** | `compute_rad_derivatives.py` | ✅ Done | K/Th, U/Th, Th/K, U/K, dose rate `.tif` files created |
| **2e. Interpolate Geochem** | `interpolate_geochem.py` | ✅ Done | Both IDW and Kriging `.tif` surfaces for all 17 elements |
| **2f. Extract Features** | `extract_features.py` | ✅ Done | `feature_matrix.parquet` — all raster bands sampled at label points |
| **2g. Engineer Features** | `engineer_features.py` | ✅ Done | Derived: mag HGM, analytic signal, radiometric ratios, log-geochem, MEAS score |
| **3a. Build Dataset** | `build_dataset.py` | ✅ Done | `training_data.npz`, `spatial_folds.npy`, `feature_names.csv`, `imputer.joblib`, `scaler.joblib` |

> **Key preprocessing decisions applied:**
> - **CRS standardisation:** all layers → EPSG:2953 (NAD83 / NB Double Stereographic)
> - **Resolution:** 100 m uniform pixel grid
> - **Null handling:** features with >75% nulls dropped; remainder median-imputed
> - **SMOTE oversampling:** applied to address class imbalance (positive:negative ~1:5)
> - **Spatial block cross-validation:** 5-fold spatial blocking to prevent data leakage

---

## 3. What Has Been Done — Model Training ✅

Both classifiers have been **trained and saved**:

| Model | File | Tuning |
|---|---|---|
| Random Forest | `models/rf_best_model.joblib` | Bayesian (Optuna, 50 trials) |
| XGBoost | `models/xgb_best_model.joblib` | Bayesian (Optuna, 50 trials) |

CV metrics saved: `rf_cv_metrics.csv`, `xgb_cv_metrics.csv`  
Feature importances saved: `rf_feature_importances.csv`, `xgb_feature_importances.csv`

---

## 4. What Still Needs to Be Done 🔲

| Priority | Step | Script | Notes |
|---|---|---|---|
| 🔴 High | **Model Evaluation** | `evaluate_models.py` | Generate ROC curves, PR curves, feature importance comparison plots |
| 🔴 High | **Prospectivity Map** | `predict_prospectivity.py` | Run trained models over the full BMC raster grid → produce `bmc_prospectivity_map.tif` |
| 🔴 High | **Map Export** | `export_map.py` (pipeline/04) | Render/export the prospectivity map for GIS or publication |
| 🟡 Medium | **SHAP Explainability** | `shap_analysis.py` (pipeline/05) | SHAP values to identify which features drive predictions at deposit vs. barren sites |
| 🟡 Medium | **Validation against known deposits** | Manual / script | Overlay prospectivity map on NB mineral occurrences — are known deposits in high-probability zones? |
| 🟢 Low | **Kriging vs IDW comparison** | `engineer_features.py` re-run | Kriging surfaces exist but IDW was used by default — worth A/B testing feature impact |
| 🟢 Low | **Geology layer integration** | New script needed | Bedrock geology (lithostratigraphy) not yet incorporated as a feature |
> [!IMPORTANT]
> The most impactful immediate next step is running **`predict_prospectivity.py`** (pipeline step 04) to generate the prospectivity map — this is the primary deliverable of the entire pipeline.
