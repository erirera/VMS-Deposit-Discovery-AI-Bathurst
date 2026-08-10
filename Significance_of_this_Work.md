# Scientific and Practical Significance of This Work

**Project:** Machine Learning Prospectivity Mapping of Volcanogenic Massive Sulphide Deposits  
**Study Area:** Bathurst Mining Camp (BMC), New Brunswick, Canada  
**Authors:** Dele Falebita, Mohammad Parsa, David Lentz  

---

## Executive Overview

This study advances data-driven **Mineral Prospectivity Mapping (MPM)** for volcanogenic massive sulphide (VMS) deposit discovery in glaciated, structurally complex terranes. Using the historic **Bathurst Mining Camp (BMC), New Brunswick, Canada** as a benchmark testbed, this work resolves fundamental mathematical, spatial, and geological challenges inherent in camp-scale machine learning workflows.

---

## Key Dimensions of Significance

### 1. Rigorous 100% Raster-Contained Label Baseline
* **Canonical Deposit Inventory:** Aligned positive VMS deposit labels ($n = 45$) with the canonical **van Staal et al. (2003)** *GSC Bulletin 566* deposit framework, eliminating legacy regional contamination (e.g., non-BMC Sb-Au or Sn-W deposits up to 200 km away).
* **Guaranteed Spatial Containment:** Achieved **100.00% spatial containment** for both positive deposits (45/45) and negative labels (250/250) strictly inside the master BMC study area raster grid (`mag_rmi_bmc_combined1.tif`), eliminating edge distortion, spatial clipping artifacts, and NoData boundary leakage.

### 2. Compositional Geochemistry (CoDA) & Multi-Scale Data Integration
* **Overcoming Compositional Closure:** Applied Centered Log-Ratio (**CLR**) transformations to a newly compiled, spatially unified 17-element till geochemistry dataset (2,753 sample locations). Subsequent **CLR-PCA** and **CLR-FA** with varimax rotation successfully unmixed false closure correlations into true hydrothermal mass-transfer footprints.
* **Dual-Scale Feature Architecture:** Combined smoothed regional CLR-FA component score surfaces (IDW interpolated) with raw elemental concentrations (Pb, Mo, Fe) as independent point-scale features. This preserved short-range, high-amplitude pathfinder anomalies that regional spatial interpolation normally suppresses.

### 3. Advanced Negative Evidence Engineering (*Parsa & Cumani, 2025*)
* **Resolving the Absence Problem:** Replaced arbitrary random or distance-buffered negative labels with a **hybrid negative label framework**:
  * **125 real drill collars** sampled from confirmed GeoNB drill holes strictly inside the BMC raster grid ($\ge 1,000\text{ m}$ buffer).
  * **125 feature-space Mahalanobis dissimilarity pseudo-absences**, selecting candidate locations that are maximally dissimilar to VMS deposit centroids in multi-dimensional geophysical and geochemical feature space.
* **Spatial Stratification:** Distributed pseudo-absences across four geographic quadrants to guarantee broad spatial spread while eliminating geographic clustering.

### 4. Native Fourier-Domain Derivatives & Spatial Autocorrelation Control
* **Fourier-Domain Potential Field Derivatives:** Computed First Vertical Derivative (FVD), Total Horizontal Gradient (THG), Analytic Signal (AS), and Tilt Derivative (TDR) directly in the Fourier domain on original survey grids prior to spatial resampling, preserving high-frequency shear-zone and fault-contact signals.
* **Spatial Block Cross-Validation:** Implemented 5-fold spatial block CV (`BlockKFold`) to prevent spatial autocorrelation leakage between training and validation splits.

### 5. Outstanding Exploration Efficiency & Model Performance
* **Discriminatory Power:** Random Forest achieved **ROC-AUC = 0.927 ± 0.047**, **PR-AUC = 0.740 ± 0.135**, and **Balanced Accuracy = 0.835 ± 0.052** under spatial cross-validation.
* **Unprecedented Discovery Efficiency:** Reached a **Success Rate AUC (SR-AUC) of 0.9679**:
  * **Top 10%** of ranked prospective area captures **91.1%** of all known VMS deposits in the camp.
  * **Top 20%** captures **97.8%** of deposits.
  * **Top 30%** captures **100.0%** of all deposits.
* **Area Reduction:** High-priority exploration targets (Prospectivity Index > 0.7) delineate just **23,097 grid cells (1.9% of total BMC study area)**, providing exploration geologists with tightly constrained, actionable drill targets.

### 6. Transparent & Geologically Actionable Explainability (TreeSHAP)
* **Demystifying the Black Box:** TreeSHAP feature attribution confirmed that predictions are driven by geologically consistent criteria:
  1. **Till Geochemistry Footprints:** Proximal Mo and Pb pathfinder anomalies (`geochem_mo_ppm_idw`, `geochem_pb_ppm_idw`).
  2. **Radiometric Alteration Halos:** Radioelement ratios (`rad_th_k_bmc`) capturing potassic/sericitic alteration.
  3. **Potential-Field Structures:** Gravity horizontal gradients (`gra_ggr_hgm_bmc`) mapping syn-volcanic bounding faults.

---

## Performance Summary Table

| Metric | Random Forest (RF) | XGBoost (XGB) | Operational Exploration Meaning |
| :--- | :---: | :---: | :--- |
| **ROC-AUC (Spatial CV)** | **0.927 ± 0.047** | 0.915 ± 0.056 | Exceptional global class discrimination |
| **PR-AUC (Spatial CV)** | **0.740 ± 0.135** | 0.639 ± 0.166 | Robust precision across positive predictions |
| **Balanced Accuracy** | 0.835 ± 0.052 | **0.844 ± 0.077** | Balanced sensitivity and specificity |
| **Success Rate AUC (SR-AUC)**| **0.968** | 0.940 | Captures 91.1% of deposits in top 10% area |
| **High Priority Area (>0.7)** | **1.9% of BMC** | 4.6% of BMC | Reduces exploration target footprint by ~98% |

---

## Impact & Broader Transferability

This work demonstrates that integrating CoDA-transformed geochemistry, Fourier potential-field derivatives, and Mahalanobis feature-dissimilarity negative labels yields a **geologically self-consistent, highly reliable predictive framework**. The methodology is directly transferable to other glacially covered VMS districts worldwide (e.g., Abitibi Greenstone Belt, Flin Flon–Snow Lake camp) where structural complexity and overburden cover present equivalent exploration challenges.
