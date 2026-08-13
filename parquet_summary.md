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

## Column Groups (102 total)

| Group | Count | Examples |
|---|---|---|
| **Identity / Spatial** | 3 | `point_id`, `source`, `geometry_wkt` |
| **Raw geochemistry** (ppm) | 17 | `ag_ppm`, `cu_ppm`, `zn_ppm`, `pb_ppm`, `as_ppm`, … |
| **Log-transformed geochem** | 17 | `log_ag_ppm`, `log_cu_ppm`, `log_zn_ppm`, … |
| **IDW-interpolated (raw)** | 21 | `geochem_cu_ppm_idw`, `geochem_pca_pc1_idw`, `geochem_fa_factor1_idw`, … |
| **IDW-interpolated (log)** | 17 | `log_geochem_cu_ppm_idw`, `log_geochem_zn_ppm_idw`, … |
| **Radiometric ratios** | 3 | `rad_k_th`, `rad_u_th`, `rad_th_k` |
| **Geophysics BMC features** | 18 | `mag_rmi_as_bmc`, `rad_k_bmc`, `gra_ggr_fvd_bmc`, … |
| **Composite score** | 1 | `geochem_meas` (range: −1.11 → +0.62) |
| **Target** | 1 | `label` (0 = background, 1 = VMS deposit) |

---

## Geochemical Elements Covered

**17 elements**: Ag, As, Ba, Bi, Cd, Co, Cu, Fe, In, Mn, Mo, Ni, Pb, Sb, Sn, Tl, Zn

---

## Missing Data (raw ppm columns)

| Column | Non-null | Null % |
|---|---|---|
| `ag_ppm`, `ba_ppm`, `co_ppm`, `cu_ppm`, `mo_ppm`, `ni_ppm`, `pb_ppm`, `zn_ppm` | 223 / 295 | 24.4 % |
| `as_ppm`, `sb_ppm`, `fe_ppm` | 199 / 295 | 32.5 % |
| `sn_ppm` | 159 / 295 | 46.1 % |
| `cd_ppm` | 158 / 295 | 46.4 % |
| `mn_ppm` | 123 / 295 | 58.3 % |
| `bi_ppm`, `in_ppm`, `tl_ppm` | 117–118 / 295 | 60–60.3 % |

> [!NOTE]
> IDW-interpolated columns (`*_idw`) are complete for all 295 rows — they fill the spatial gaps left by the sparse direct measurements.

---

## Key Geophysical / Radiometric Features

| Column | Description | Mean | Std |
|---|---|---|---|
| `rad_k_th` | K/Th ratio | 5.04 × 10⁶ | 2.13 × 10⁷ |
| `rad_u_th` | U/Th ratio | 2.46 × 10⁷ | 5.01 × 10⁷ |
| `rad_th_k` | Th/K ratio | 5.71 × 10⁶ | 2.99 × 10⁷ |
| `geochem_meas` | Composite geochemical score | 0.00 | 0.55 |

---

## Notable Statistics (selected raw ppm)

| Element | Mean | Std | Min | Median | Max |
|---|---|---|---|---|---|
| Ag | — | — | 0.1 | — | — (log mean −3.64) |
| As | 22.8 | 34.4 | 2.0 | 13.0 | 200 |
| Cu | 31.9 | 32.1 | 6.0 | 27.2 | 130 |
| Pb | 78.8 | 289.5 | 0.0 | 29.0 | 3400 |
| Zn | 121.4 | 86.6 | 0.0 | 101.0 | 950 |
| Sb | 2.81 | 6.27 | 0.2 | 1.1 | 49.0 |
| Sn | 14.9 | 31.9 | 0.0 | 1.3 | 100 |

---

## Derived / Dimensionality-Reduced Features

The IDW group also includes **PCA** and **Factor Analysis** components:
- `geochem_pca_pc1_idw` … `geochem_pca_pc4_idw` (4 PCs)
- `geochem_fa_factor1_idw` … `geochem_fa_factor4_idw` (4 FA factors)

These capture multi-element covariance structure interpolated across the study area.

---

## Summary

The matrix combines **direct borehole / soil geochemistry** (raw + log-transformed, with significant missingness) with **spatially complete IDW-interpolated surfaces** and **airborne geophysical signatures** (magnetics, gravity, radiometrics). The binary `label` marks **45 confirmed VMS deposit locations** against **250 background points** — a moderately imbalanced classification problem well-suited for tree-based or probability-calibrated models.
