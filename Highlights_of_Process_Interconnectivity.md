# Machine Learning VMS Prospectivity Mapping: Pipeline Process Interconnectivity

This document outlines the data-flow architecture, cross-phase dependencies, and explicit interconnectivity of the machine learning prospectivity mapping methodology developed for volcanogenic massive sulphide (VMS) deposit discovery in the Bathurst Mining Camp (BMC), New Brunswick, Canada.

---

## Overview of Interconnected Data Flow Architecture

The Machine Learning VMS prospectivity mapping pipeline is structured as a Directed Acyclic Graph (DAG) spanning five sequential and mathematically connected phases:

```
[Phase 1: Geoscience Data Compilation]
   │
   ├── Airborne Geophysics (1A) ───────> Native Grid Derivatives (2A) ───────┐
   ├── Till Geochemistry (1B) ────────> Compositional Geochem (2B) ─────────┼─> [Master Raster Stack (2C)]
   └── Deposit & Drill DB (1C) ──┐                                            │           │
                                 ├──> Positive Deposit Labels (3A) ──────────┤           │ (Mahalanobis Feature Space)
                                 └──> Barren Drill Intercepts (3B) <─────────┘           │
                                             │                                           │ (Raster Feature Sampling)
                                             v                                           v
                                   [Hybrid Negative Labels (3B)] ────────> [Balanced Feature Matrix (3C)]
                                                                                     │
                                                                                     v
                                                                           [Spatial Block CV (4A)]
                                                                                     │ (CV Folds)
                                                                                     v
                                                                           [Model Optimization (4B)] ──> [OOF Evaluation (4C)]
                                                                                     │
                                   ┌─────────────────────────────────────────────────┴─────────────────────────────────────────────────┐
                                   │ (Trained Models)                                                                                  │ (Trees & Weights)
                                   v                                                                                                   v
[Master Raster Stack (2C)] ──> [Camp-Scale Inference (5A)] ──> [Target Delineation & GIS (5B)]                             [TreeSHAP Explainability (5C)]
```

---

## Phase-by-Phase Process Interconnectivity

### 1. Phase 1 → Phase 2: Data Preprocessing & Feature Synthesis
- **Airborne Geophysics (1A → 2A):** Total Magnetic Intensity (TMI), Bouguer Gravity, and Radiometrics (%K, eTh, eU) feed into Fourier-domain derivative processing on native grids, yielding First Vertical Derivative (FVD), Total Horizontal Gradient (THG), Tilt Derivative (TDR), Analytic Signal (AS), and radioelement alteration ratios (K/Th, U/Th).
- **Till Geochemistry (1B → 2B):** Unified 17-element till geochemistry sample coordinates are processed via Centered Log-Ratio (CLR) transformations to lift compositional data into unconstrained Euclidean space, followed by Inverse Distance Weighting (IDW) interpolation and Compositional PCA/FA to produce orthogonal hydrothermal alteration and lithological factor surfaces.
- **Derivative & Factor Fusion (2A + 2B → 2C):** All potential-field derivative rasters and geochemical factor score surfaces are resampled and spatially aligned to a master 100m raster grid (EPSG:2953), alongside log-transformed raw element concentrations and the Multi-Element Anomaly Score (MEAS), producing the unified **Master Raster Stack (2C)**.

---

### 2. Phase 1 & Phase 2 → Phase 3: Spatial Labeling & Feature Matrix Assembly
- **Deposit & Drilling Records (1C → 3A, 3B):** Known VMS deposit locations provide positive deposit centroids (**3A**), while confirmed barren drill hole collars provide geologically verified negative labels (**3B**).
- **Mahalanobis Feature-Space Dissimilarity (2C → 3B):** The multi-dimensional feature space covariance structure from the **Master Raster Stack (2C)** is evaluated against candidate grid points to select feature-space dissimilar negative labels by maximising Mahalanobis distance from the VMS deposit centroid.
- **Raster Feature Sampling (2C + 3A + 3B → 3C):** All 59 feature layers in the **Master Raster Stack (2C)** are sampled at the exact spatial coordinates of positive deposit labels (**3A**) and hybrid negative labels (**3B**).
- **Data Quality & Class Balancing (3C):** Features exceeding a 75% missing value threshold are excluded, remaining missing values undergo column-wise median imputation, and SMOTE over-sampling balances minority deposit labels to match negative labels (1:1 ratio, $n=500$), constructing the final **Balanced Feature Matrix (3C)**.

---

### 3. Phase 3 → Phase 4: Spatial Block Cross-Validation & Model Training
- **Spatial Partitioning (3C → 4A):** The **Balanced Feature Matrix (3C)** is partitioned using a 5-fold spatial BlockKFold scheme, isolating spatially disjoint geographical blocks to eliminate spatial autocorrelation data leakage.
- **Classifier Hyperparameter Tuning (4A → 4B):** Training folds generated by **Spatial Block CV (4A)** drive Optuna hyperparameter optimization for Random Forest (RF) and XGBoost classifiers (**4B**) with balanced class weighting.
- **Out-of-Fold Evaluation (4B → 4C):** Predictions generated across out-of-fold spatial validation blocks are evaluated using ROC-AUC, Average Precision (AP), Balanced Accuracy (BA), and cumulative Success Rate (SR-AUC) curves (**4C**).

---

### 4. Phase 4 & Phase 2 → Phase 5: Camp-Scale Prediction, GIS & Explainability
- **Full-Grid Inference (4B + 2C → 5A):** Best-performing trained RF and XGBoost classifiers (**4B**) evaluate all 1,194,109 grid cells (100m resolution) across the full extent of the **Master Raster Stack (2C)**, producing a continuous Prospectivity Index (PI) heatmap (**5A**).
- **GIS Export & Target Delineation (5A → 5B):** The prospectivity index raster (**5A**) is exported as a georeferenced GeoTIFF (EPSG:2953) for GIS integration, filtering high-priority exploration targets ($\text{PI} > 0.7$) and assessing area-normalized deposit capture efficiency (**5B**).
- **TreeSHAP Feature Attribution (4B → 5C):** Model trees and decision structures from trained classifiers (**4B**) feed into TreeSHAP feature importance analysis, generating global feature rankings, local SHAP attributions, and Partial Dependence Plots (PDP) to validate geological mechanisms (**5C**).

---

## File Synchronization & Artifacts

- **Workflow Generator Script:** [`pipeline/generate_workflow_diagram.py`](file:///c:/Users/delef/.gemini/antigravity/bathurst%20VMS/VMS-Deposit-Discovery-AI-Bathurst/pipeline/generate_workflow_diagram.py)
- **Publication Flowchart Image:** [`workflow.png`](file:///c:/Users/delef/.gemini/antigravity/bathurst%20VMS/VMS-Deposit-Discovery-AI-Bathurst/workflow.png)
- **Manuscript Reference:** [`manuscript_NRR.md`](file:///c:/Users/delef/.gemini/antigravity/bathurst%20VMS/VMS-Deposit-Discovery-AI-Bathurst/manuscript_NRR.md) (Section 3, Figure 2)
