"""
pca_fa_geochem.py  --  Centered Log-Ratio (CLR) Transformation and PCA / Factor Analysis
========================================================================================
Bathurst Mining Camp VMS Deposit Discovery AI Pipeline
"""

import argparse
import logging
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler

# Resolve repo root and import shared config
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "pipeline"))
from config import CRS_TARGET, NODATA_VALUE, PROCESSED_DIR

RASTERS_DIR = PROCESSED_DIR / "rasters_reprojected"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pca_fa_geochem")

ELEMENTS = [
    "Ag", "As", "Ba", "Bi", "Cd", "Co", "Cu", "Fe", "In", "Mn", "Mo", "Ni", "Pb", "Sb", "Sn", "Tl", "Zn"
]

def impute_zeros(X: np.ndarray) -> np.ndarray:
    """Impute values <= 0 with 0.5 * the minimum positive value for each element."""
    X_imputed = X.copy()
    for col in range(X.shape[1]):
        vals = X_imputed[:, col]
        pos_vals = vals[vals > 0]
        if len(pos_vals) > 0:
            min_pos = pos_vals.min()
            impute_val = min_pos * 0.5
        else:
            impute_val = 1e-6
        X_imputed[vals <= 0, col] = impute_val
    return X_imputed

def main():
    parser = argparse.ArgumentParser(description="Run PCA & FA on geochemical rasters.")
    parser.add_argument("--method", choices=["idw", "kriging"], default="idw",
                        help="Interpolation method of geochemical rasters (default: idw)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output rasters")
    parser.add_argument("--max-fit-samples", type=int, default=100000,
                        help="Max samples to use for fitting PCA/FA models to speed up computation")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info(" Geochemistry CLR + PCA & Factor Analysis (Optimized)")
    log.info("   Method: %s", args.method.upper())
    log.info("=" * 60)

    # 1. Load the 17 rasters
    grids = []
    meta = None
    
    for elem in ELEMENTS:
        raster_name = f"geochem_{elem.lower()}_ppm_{args.method}.tif"
        raster_path = RASTERS_DIR / raster_name
        if not raster_path.exists():
            log.error("Required raster not found: %s. Run interpolate_geochem.py first.", raster_path)
            sys.exit(1)
            
        with rasterio.open(raster_path) as src:
            if meta is None:
                meta = src.meta.copy()
            grid = src.read(1)
            grids.append(grid)
            
    # Stack to shape (17, height, width)
    grids = np.stack(grids, axis=0)
    height, width = grids.shape[1], grids.shape[2]
    
    # 2. Identify valid pixels (not nodata in any raster)
    valid_mask = (grids != NODATA_VALUE).all(axis=0)
    n_valid = valid_mask.sum()
    log.info("Grid size: %d rows x %d cols = %d pixels", height, width, height * width)
    log.info("Valid geochemical pixels: %d (%.2f%%)", n_valid, 100.0 * n_valid / (height * width))
    
    if n_valid == 0:
        log.error("No valid pixels found (all pixels contain NODATA).")
        sys.exit(1)
        
    # Extract data for valid pixels
    # X shape: (n_valid, 17)
    X = grids[:, valid_mask].T
    
    # 3. Impute zeros/negatives
    X_imputed = impute_zeros(X)
    
    # 4. CLR Transformation
    log_X = np.log(X_imputed)
    log_g = np.mean(log_X, axis=1, keepdims=True)
    clr_X = log_X - log_g
    
    # 5. Standardise CLR-transformed data
    scaler = StandardScaler()
    
    # If the dataset is very large, fit scaler/PCA/FA on a representative subset
    if n_valid > args.max_fit_samples:
        log.info("Dataset size (%d) exceeds max fit samples (%d).", n_valid, args.max_fit_samples)
        log.info("Sampling %d random pixels for fitting...", args.max_fit_samples)
        np.random.seed(42)
        idx = np.random.choice(n_valid, size=args.max_fit_samples, replace=False)
        clr_fit_data = clr_X[idx]
        
        # Fit scaler on subset
        scaler.fit(clr_fit_data)
        clr_scaled_fit = scaler.transform(clr_fit_data)
    else:
        clr_scaled_fit = scaler.fit_transform(clr_X)
        
    # Transform full dataset
    clr_scaled_full = scaler.transform(clr_X)
    
    n_components = 4
    
    # PCA
    log.info("Fitting PCA (n_components=%d) ...", n_components)
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(clr_scaled_fit)
    
    log.info("Transforming full grid with PCA...")
    pca_scores = pca.transform(clr_scaled_full)
    
    # Factor Analysis
    log.info("Fitting Factor Analysis (n_components=%d, max_iter=100) ...", n_components)
    fa = FactorAnalysis(n_components=n_components, max_iter=100, random_state=42)
    fa.fit(clr_scaled_fit)
        
    log.info("Transforming full grid with Factor Analysis...")
    fa_scores = fa.transform(clr_scaled_full)
    
    # 6. Log PCA Variance Explained
    log.info("")
    log.info("PCA Explained Variance:")
    for i in range(n_components):
        log.info("  PC%d: %.2f%%  (cumulative: %.2f%%)", i+1, 
                 pca.explained_variance_ratio_[i] * 100, 
                 np.sum(pca.explained_variance_ratio_[:i+1]) * 100)
                 
    # 7. Print Loadings Table
    loadings_df = pd.DataFrame(index=ELEMENTS)
    for i in range(n_components):
        loadings_df[f"PC{i+1}"] = pca.components_[i]
    for i in range(n_components):
        loadings_df[f"FA{i+1}"] = fa.components_[i]
        
    log.info("\n" + "=" * 65)
    log.info(" ELEMENT LOADINGS (PCA & FA)")
    log.info("=" * 65)
    log.info(loadings_df.round(3).to_string())
    log.info("=" * 65)
    
    # Save loadings to CSV
    loadings_path = PROCESSED_DIR / f"geochem_pca_fa_loadings_{args.method}.csv"
    loadings_df.to_csv(loadings_path)
    log.info("Saved loadings matrix -> %s", loadings_path.name)
    
    # 8. Write score grids to GeoTIFFs
    meta.update(dtype=rasterio.float32, nodata=NODATA_VALUE, count=1)
    
    for i in range(n_components):
        # PCA PCi
        pca_out = np.full((height, width), NODATA_VALUE, dtype=np.float32)
        pca_out[valid_mask] = pca_scores[:, i]
        pca_path = RASTERS_DIR / f"geochem_pca_pc{i+1}_{args.method}.tif"
        
        with rasterio.open(pca_path, "w", **meta) as dst:
            dst.write(pca_out, 1)
        log.info("Saved PCA score grid  -> %s", pca_path.name)
        
        # FA Factori
        fa_out = np.full((height, width), NODATA_VALUE, dtype=np.float32)
        fa_out[valid_mask] = fa_scores[:, i]
        fa_path = RASTERS_DIR / f"geochem_fa_factor{i+1}_{args.method}.tif"
        
        with rasterio.open(fa_path, "w", **meta) as dst:
            dst.write(fa_out, 1)
        log.info("Saved FA score grid   -> %s", fa_path.name)
        
    log.info("=" * 60)
    log.info(" Geochemistry PCA & FA completed successfully!")
    log.info("=" * 60)

if __name__ == "__main__":
    main()
