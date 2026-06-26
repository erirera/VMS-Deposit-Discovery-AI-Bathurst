# Hybrid Label Generation Strategy for VMS Prospectivity

Transition the label generation pipeline from a purely drill-hole-based negative dataset to a **hybrid negative dataset** combining:
1. **50% Confirmed Barren Drill Holes (125 points):** Captures local near-miss signatures and stratigraphic/alteration controls.
2. **50% Regional Pseudo-Absences (125 points):** Generated randomly within the active geophysical grid footprint, restricted to areas at least **3 km away** from any known VMS deposit. This represents the regional geological background and reduces spatial clustering bias.

---

## User Review Required

> [!IMPORTANT]
> **Key Parameters for the Hybrid Approach:**
> - **Exclusion Buffer:** We propose a **3.0 km exclusion buffer** around all 45 known VMS deposits for the pseudo-absence generator. This prevents generating "false negatives" within the alteration halo of known deposits.
> - **Data Footprint Boundary:** The pseudo-absences will be constrained within the active bounding box of the reprojected 50 m geophysical grids (Easting: `2481060.0` to `2576360.0`, Northing: `7551140.0` to `7635540.0` in EPSG:2953).
> - **Drill Hole Subset:** We will select 125 barren holes from the compiled `BARREN_HOLES` dataset using a reproducible random seed (`42`).

---

## Proposed Changes

### Label Downloader Component

#### [MODIFY] [download_vms_labels.py](file:///c:/Users/delef/.gemini/antigravity/bathurst%20VMS/VMS-Deposit-Discovery-AI-Bathurst/pipeline/01_data_download/download_vms_labels.py)
* Update the negative label generator to implement the hybrid strategy:
  * Randomly sample 125 confirmed barren drill holes from the `BARREN_HOLES` list using `random.seed(42)`.
  * Generate 125 random coordinates within the active geophysical grid footprint.
  * Filter out any random points that fall within 3.0 km of the 45 known VMS deposits.
  * Check that generated points fall within the valid data footprint of the master rasters.
  * Combine the 125 barren holes and 125 pseudo-absences into a unified `barren_negative_labels.gpkg` (total 250 points, keeping the 1:5.5 class ratio).

---

## Verification Plan

### Automated Execution & Pipeline Run
We will execute the complete preprocessing and training pipeline to verify the impact of the hybrid labels:
1. **Re-generate Labels:**
   ```bash
   python pipeline/01_data_download/download_vms_labels.py
   ```
2. **Re-extract Features:**
   ```bash
   python pipeline/02_preprocessing/extract_features.py
   ```
3. **Re-engineer Features:**
   ```bash
   python pipeline/02_preprocessing/engineer_features.py
   ```
4. **Re-build Dataset:**
   ```bash
   python pipeline/03_training/build_dataset.py
   ```
5. **Re-train Models:**
   ```bash
   python pipeline/03_training/train_rf.py
   python pipeline/03_training/train_xgb.py
   ```

### Manual Verification & QC
- Check the spatial distribution of the new negative labels.
- Verify that no pseudo-absence points are located within 3 km of any known VMS deposit.
- Compare model performance (cross-validation AUC and Balanced Accuracy) between the pure-barren model and the hybrid-label model.
