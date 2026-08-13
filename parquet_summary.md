# `feature_matrix.parquet` — Dataset Summary

**Path**: `data/processed/feature_matrix.parquet`
**Memory**: 0.24 MB

---

## Overview

| Property | Value |
|---|---|
| Rows | **295** |
| Columns | **102** |
| Target column | `label` |
| Positive samples (VMS = 1) | **45** (15.3 %) |
| Negative samples (background = 0) | **250** (84.7 %) |
| Class imbalance ratio | ~5.6 : 1 |

---

## Column Groups (102 total — verified)

| Group | Count | Contents |
|---|---:|---|
| **Identity / Spatial** | 3 | `point_id`, `source`, `geometry_wkt` |
| **Raw geochemistry** (ppm) | 17 | `ag_ppm` … `zn_ppm` (all 17 elements) |
| **Log-transformed geochem** | 17 | `log_ag_ppm` … `log_zn_ppm` |
| **IDW-interpolated (raw)** | 25 | 17 element surfaces + 4 PCA PCs + 4 FA factors |
| **IDW-interpolated (log)** | 17 | `log_geochem_ag_ppm_idw` … `log_geochem_zn_ppm_idw` |
| **Radiometric ratios** | 3 | `rad_k_th`, `rad_u_th`, `rad_th_k` |
| **Geophysics BMC features** | 18 | 6 gravity + 5 magnetics + 7 radiometric BMC derivatives |
| **Composite score** | 1 | `geochem_meas` (range: −1.11 → +0.62) |
| **Target** | 1 | `label` (0 = background, 1 = VMS deposit) |
| **Total** | **102** | ✅ All columns accounted for |

---

## Geochemical Elements Covered

**17 elements**: Ag, As, Ba, Bi, Cd, Co, Cu, Fe, In, Mn, Mo, Ni, Pb, Sb, Sn, Tl, Zn

Each element is represented in up to three forms:
1. Raw sparse `*_ppm` (variable missingness)
2. Log₁₀-transformed raw `log_*_ppm`
3. Spatially complete IDW-interpolated surface `geochem_*_ppm_idw` (0% null)

---

## Missing Data (raw ppm columns)

| Element group | Non-null | Null % | Status in training |
|---|---|---|---|
| Ag, Ba, Co, Cu, Mo, Ni, Pb, Zn | 223 / 295 | 24.4 % | Retained + imputed |
| As, Sb, Fe | 199 / 295 | 32.5 % | Retained + imputed |
| Sn | 159 / 295 | 46.1 % | Retained + imputed |
| Cd | 158 / 295 | 46.4 % | Retained + imputed |
| Mn | 123 / 295 | 58.3 % | Retained + imputed |
| Bi, In, Tl | 117–118 / 295 | 60–60.3 % | Retained + imputed |

> [!NOTE]
> The pipeline drop threshold is **75% null**. No raw element reached this threshold, so all 17 were retained. Missing values were imputed with column-wise median. IDW-interpolated counterparts are available for all 295 rows with 0% null, providing complete spatial coverage for every element regardless of sparse-data rates.

---

## Geophysics BMC Features (18 columns)

| Sub-group | Columns |
|---|---|
| **Magnetics** (5) | `mag_rmi_bmc_combined1`, `mag_rmi_fvd_bmc`, `mag_rmi_thg_bmc`, `mag_rmi_as_bmc`, `mag_rmi_tdr_bmc` |
| **Gravity** (6) | `gra_ggr_hgm_bmc`, `gra_ggr_tdr_bmc`, `gra_ggr_fvd_bmc`, `gra_ggr_as_bmc`, `gra_ggr_uc500_bmc`, `gra_ggr_res_bmc` |
| **Radiometric BMC** (7) | `rad_k_bmc`, `rad_th_bmc`, `rad_u_bmc`, `rad_k_th_bmc`, `rad_u_th_bmc`, `rad_th_k_bmc`, `rad_u_k_bmc` |

---

## IDW Dimensionality-Reduced Features

Embedded within the 25 raw IDW columns:
- **PCA**: `geochem_pca_pc1_idw` … `geochem_pca_pc4_idw`
- **FA**: `geochem_fa_factor1_idw` … `geochem_fa_factor4_idw`

Derived via CLR → StandardScaler → PCA / FactorAnalysis on the 17-element IDW raster stack (see `pipeline/02_preprocessing/pca_fa_geochem.py`).

---

## Notable Element Statistics (selected raw ppm)

| Element | n | Mean | Std | Median | Max |
|---|---|---|---|---|---|
| As | 199 | 22.8 | 34.4 | 13.0 | 200 |
| Cu | 223 | 31.9 | 32.1 | 27.2 | 130 |
| Zn | 223 | 121.4 | 86.6 | 101.0 | 950 |
| Pb | 223 | 78.8 | 289.5 | 29.0 | **3,400** |
| Sb | 199 | 2.81 | 6.27 | 1.1 | 49.0 |
| Sn | 159 | 14.9 | 31.9 | 1.3 | 100 |

> [!TIP]
> Pb has a maximum of 3,400 ppm vs. a median of 29 ppm — strongly right-skewed. Log-transformation is essential before modelling.

---

## Top-10 Predictive Features (SHAP mean |SHAP|)

> [!NOTE]
> Values are **mean |SHAP|** computed from `outputs/shap/shap_values_*.csv` (500 SMOTE-balanced samples). These are the authoritative importance scores reported in manuscript §4.4. The model CSV files (`models/*_feature_importances.csv`) contain Gini/gain-based importances and should **not** be used for ranking.

### Random Forest (SHAP)

| Rank | Feature | Mean \|SHAP\| | Domain |
|---|---|---|---|
| 1 | `rad_th_k_bmc` | 0.0531 | Radiometrics — Th/K alteration halo |
| 2 | `rad_th_bmc` | 0.0435 | Radiometrics — raw Th |
| 3 | `geochem_mo_ppm_idw` | 0.0340 | Geochemistry — Mo IDW |
| 4 | `rad_k_bmc` | 0.0232 | Radiometrics — raw K |
| 5 | `geochem_zn_ppm_idw` | 0.0208 | Geochemistry — Zn IDW |
| 6 | `geochem_bi_ppm_idw` | 0.0170 | Geochemistry — Bi IDW |
| 7 | `gra_ggr_hgm_bmc` | 0.0166 | Gravity — horizontal gradient |
| 8 | `geochem_pb_ppm_idw` | 0.0161 | Geochemistry — Pb IDW |
| 9 | `mo_ppm` | 0.0128 | Geochemistry — Mo raw |
| 10 | `geochem_fa_factor2_idw` | 0.0126 | Geochemistry — CLR-FA Factor 2 |

### XGBoost (SHAP)

| Rank | Feature | Mean \|SHAP\| | Domain |
|---|---|---|---|
| 1 | `rad_th_k_bmc` | 1.1991 | Radiometrics — Th/K alteration halo |
| 2 | `geochem_mo_ppm_idw` | 0.5331 | Geochemistry — Mo IDW |
| 3 | `zn_ppm` | 0.4783 | Geochemistry — Zn raw |
| 4 | `geochem_sn_ppm_idw` | 0.4046 | Geochemistry — Sn IDW |
| 5 | `gra_ggr_uc500_bmc` | 0.3952 | Gravity — upward-continued Bouguer |
| 6 | `rad_u_th_bmc` | 0.3735 | Radiometrics — U/Th ratio |
| 7 | `mo_ppm` | 0.3656 | Geochemistry — Mo raw |
| 8 | `ni_ppm` | 0.3580 | Geochemistry — Ni raw |
| 9 | `mag_rmi_as_bmc` | 0.3483 | Magnetics — analytic signal |
| 10 | `mag_rmi_fvd_bmc` | 0.3261 | Magnetics — first vertical derivative |

### Cross-model Predictor Domains

| Domain | RF top features | XGBoost top features |
|---|---|---|
| **Radiometric alteration halos** | rad_th_k, rad_th, rad_k | rad_th_k, rad_u_th |
| **Geochemical pathfinders** | Mo, Zn, Bi, Pb, FA2 | Mo, Zn, Sn, Ni |
| **Potential-field structural** | gra_ggr_hgm | gra_ggr_uc500, mag_rmi_as, mag_rmi_fvd |

---

## Summary

The matrix combines **direct borehole/soil geochemistry** (raw + log-transformed; 24–60% missingness in sparse columns) with **spatially complete IDW-interpolated surfaces** (0% null; including PCA and FA components) and **airborne geophysical signatures** (magnetics, gravity, radiometrics). The binary `label` marks **45 confirmed VMS deposit locations** against **250 background points** — a moderately imbalanced classification problem (~5.6 : 1) addressed via SMOTE within training folds.

