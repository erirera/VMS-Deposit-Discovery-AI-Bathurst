# Walkthrough - Derivative & Kriging Data Removal

All requested features and Kriging data have been removed from the geophysics, radiometrics, and geochemistry pipelines. The model training and full-extent prediction scripts were successfully run to verify the changes end-to-end.

## Changes Made

### 1. Preprocessing Code & Config Updates
*   **[config.py](file:///c:/Users/delef/.gemini/antigravity/bathurst%20VMS/VMS-Deposit-Discovery-AI-Bathurst/pipeline/config.py):** Removed the obsolete geophysics derivatives (`mag_rmi_svd_bmc`, `gra_ggr_svd_bmc`, `mag_rmi_thdr_bmc`), the radiometric dose rate (`rad_dose_bmc`), and all Kriging-based PCA/FA scores (`geochem_pca_pc*_kriging`, `geochem_fa_factor*_kriging`).
*   **[compute_mag_derivatives.py](file:///c:/Users/delef/.gemini/antigravity/bathurst%20VMS/VMS-Deposit-Discovery-AI-Bathurst/pipeline/02_preprocessing/compute_mag_derivatives.py):** Deleted Second Vertical Derivative (`SVD`) and Tilt Horizontal Gradient (`THDR`) computations.
*   **[compute_grav_derivatives.py](file:///c:/Users/delef/.gemini/antigravity/bathurst%20VMS/VMS-Deposit-Discovery-AI-Bathurst/pipeline/02_preprocessing/compute_grav_derivatives.py):** Deleted Second Vertical Derivative (`SVD`) computation.
*   **[compute_rad_derivatives.py](file:///c:/Users/delef/.gemini/antigravity/bathurst%20VMS/VMS-Deposit-Discovery-AI-Bathurst/pipeline/02_preprocessing/compute_rad_derivatives.py):** Deleted Dose Rate Proxy computation and its IAEA conversion coefficients.
*   **[interpolate_geochem.py](file:///c:/Users/delef/.gemini/antigravity/bathurst%20VMS/VMS-Deposit-Discovery-AI-Bathurst/pipeline/02_preprocessing/interpolate_geochem.py):** Removed Ordinary Kriging logic and PyKrige imports to improve system dependency constraints. Geochemical interpolation is now run exclusively via IDW.
*   **[pca_fa_geochem.py](file:///c:/Users/delef/.gemini/antigravity/bathurst%20VMS/VMS-Deposit-Discovery-AI-Bathurst/pipeline/02_preprocessing/pca_fa_geochem.py):** Cleaned up to only perform CLR-PCA and CLR-FA on IDW geochemistry surfaces.

### 2. Workspace Cleanup
*   Deleted obsolete derivative and Kriging-based GeoTIFFs and QC plots from `data/processed/`.

---

## Verification & Execution Results

The pipeline was run step-by-step to recompute all remaining features, build the training dataset, retrain models, and generate predictions:

### 1. Preprocessing & Feature Extraction
*   Magnetic, gravity, and radiometric preprocessing completed without errors.
*   Geochemical CLR PCA/FA ran on IDW surfaces only, producing explained variance metrics:
    *   **PC1:** 58.05%
    *   **PC2:** 13.57%
    *   **PC3:** 6.82%
    *   **PC4:** 5.70% (Cumulative Variance: 84.15%)
*   `extract_features.py` built the new feature matrix, which correctly shrunk in shape from **(295, 93)** to **(295, 68)**.
*   `engineer_features.py` successfully appended 29 secondary features (final shape **295 x 97**).

### 2. Dataset Assembly & Class Balancing (SMOTE)
*   Excluded 6 raw pathfinder elements due to high null counts (as they are instead represented by 100% complete IDW surfaces).
*   42 final features were retained.
*   SMOTE successfully balanced the classes from 250 barren / 45 VMS points to **250 barren / 250 VMS** points.

### 3. Model Training & Feature Importance
*   **Random Forest Model:**
    *   **Best Params:** `{'n_estimators': 238, 'max_depth': 10, 'min_samples_leaf': 1, 'max_features': 0.5}`
    *   **ROC-AUC:** 0.7895 ± 0.1484 (Fold 2 reached 0.9400 ROC-AUC)
    *   **Top 5 Features:**
        1.  `mag_rmi_tdr_bmc` (Tilt Derivative) - 16.36% Gini Importance
        2.  `mag_rmi_thg_bmc` (Total Horizontal Gradient) - 8.10%
        3.  `geochem_fa_factor4_idw` - 3.63%
        4.  `geochem_sb_ppm_idw` (Antimony IDW surface) - 3.58%
        5.  `rad_th_bmc` (Thorium radiometric band) - 3.32%
*   **XGBoost Model:**
    *   **ROC-AUC:** 0.6999 ± 0.1607
    *   **Top 3 Features:**
        1.  `mag_rmi_tdr_bmc` (Tilt Derivative)
        2.  `mag_rmi_thg_bmc` (Total Horizontal Gradient)
        3.  `rad_u_bmc` (Uranium radiometric band)

### 4. Prospectivity Prediction Map
*   Full-extent prospectivity grid of **1,194,109 cells** at 100m resolution was predicted using the best model (Random Forest).
*   Prospectivity range: **0.0000 to 0.9956**
*   High-probability target cells (>0.7 probability): **112,778 cells (9.4% of study area)**
*   Publication maps exported successfully to `outputs/bmc_prospectivity_map.png` and `outputs/bmc_prospectivity_map_hires.pdf`.
*   SHAP explainability plots generated and saved in `outputs/shap/`.
