# Implementation Plan: Native Derivative Computation (Solution A)

This plan refactors the preprocessing pipeline for both magnetic and gravity datasets to compute derivatives on their native grids first, and then reproject and resample the computed derivative grids to the target CRS (`EPSG:2953`) and resolution (`50 m`). This eliminates the bilinear interpolation artifacts (waffle/grid lines) currently generated when derivatives are computed on upsampled grids.

## User Review Required

> [!IMPORTANT]
> **Resolution and Coordinate Consistency:**
> - Both scripts will reproject and output the final derivative grids at **`50 m`** resolution in `EPSG:2953`.
> - The final grids will be saved to both their individual subfolders (under `mag_derivatives/` and `grav_derivatives/`) and the central `data/processed/rasters_reprojected/` folder to ensure the downstream feature extraction and training pipeline continues to work automatically without any changes to other scripts.
>
> **Calculation Domain:**
> - FFT-based calculations (like FVD and SVD) will run on the native grid spacing `dx` (e.g. `145.34 m` for high-res magnetics and `597.41 m` for gravity) which is mathematically correct and prevents high-frequency amplification of interpolation slope discontinuities.

## Proposed Changes

---

### Preprocessing Pipelines

#### [MODIFY] [compute_mag_derivatives.py](file:///c:/Users/delef/.gemini/antigravity/bathurst%20VMS/VMS-Deposit-Discovery-AI-Bathurst/pipeline/02_preprocessing/compute_mag_derivatives.py)
* Refactor `main()` to load the raw `source_raster` directly.
* Compute FFT-based vertical derivatives (`FVD`, `SVD`), horizontal gradients, and Analytic Signal (`AS`) on the native grid spacing `dx` (e.g., `145.34 m`).
* Add a helper function `reproject_and_save_array` to reproject the computed derivative arrays from native CRS/resolution to `EPSG:2953` at `50 m` resolution using `rasterio.warp.reproject`.
* Write the reprojected grids to both `data/processed/mag_derivatives/<source_stem>/` and `data/processed/rasters_reprojected/`.
* Generate the QC PNG plots using the final reprojected `50 m` grids to check quality.

#### [MODIFY] [compute_grav_derivatives.py](file:///c:/Users/delef/.gemini/antigravity/bathurst%20VMS/VMS-Deposit-Discovery-AI-Bathurst/pipeline/02_preprocessing/compute_grav_derivatives.py)
* Refactor `main()` to load the raw gravity raster directly.
* Compute gravity derivatives (`HGM`, `TDR`, `FVD`, `SVD`, `AS`, `UC500`, `RES`) on the native grid spacing `dx` (e.g., `597.41 m`).
* Use the same `reproject_and_save_array` helper to reproject each computed derivative array to `EPSG:2953` at `50 m` resolution.
* Write the reprojected grids to both `data/processed/grav_derivatives/<source_stem>/` and `data/processed/rasters_reprojected/`.
* Generate the QC PNG plots using the final reprojected `50 m` grids.

---

## Verification Plan

### Automated Tests
We will execute the pipeline step-by-step to verify that the refactored scripts compute the new grids correctly, and that the model training pipeline works end-to-end:

1. **Compute Magnetic Derivatives**:
   ```powershell
   python pipeline/02_preprocessing/compute_mag_derivatives.py
   ```
2. **Compute Gravity Derivatives**:
   ```powershell
   python pipeline/02_preprocessing/compute_grav_derivatives.py
   ```
3. **Extract Features**:
   ```powershell
   python pipeline/02_preprocessing/extract_features.py
   ```
4. **Engineer Features**:
   ```powershell
   python pipeline/02_preprocessing/engineer_features.py
   ```
5. **Build Dataset**:
   ```powershell
   python pipeline/03_training/build_dataset.py
   ```
6. **Train Models**:
   ```powershell
   python pipeline/03_training/train_rf.py
   python pipeline/03_training/train_xgb.py
   ```

### Manual Verification
* Visual Inspection: Check the generated QC PNG plots in `data/processed/mag_derivatives/mag_rmi_bmc_compiled/qc_plots/` and `data/processed/grav_derivatives/gra_ggr_bmc_combined/qc_plots/` to confirm that the waffle/grid-line interpolation artifacts are gone.
* Statistics Check: Verify that output grid minimum/maximum/standard deviation statistics are reasonable and match scientific expectations.
