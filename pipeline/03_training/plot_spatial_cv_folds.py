"""
plot_spatial_cv_folds.py
────────────────────────
Visualizes spatial cross-validation folds used in model training.

This script creates a map showing how the study area was partitioned into 
five geographically distinct spatial folds, with each fold differentiated 
by color. Points are further colored by class label (0=background, 1=VMS).

Steps:
  1. Load feature matrix (parquet)
  2. Extract coordinates from geometry_wkt
  3. Create spatial visualization with fold boundaries
  4. Display class distribution across folds
  5. Export spatial folds to GeoPackage for QGIS

Outputs:
  - spatial_cv_folds_map.png: Scatter plot of sample locations colored by fold
  - spatial_cv_distribution.png: Bar chart of fold sizes and class balance
  - spatial_cv_folds.gpkg: GeoPackage file with fold assignments and class labels (for QGIS)

Usage:
    python pipeline/03_training/plot_spatial_cv_folds.py
"""

from statistics import quantiles
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from shapely.wkt import loads
from shapely.geometry import Point

PIPELINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_DIR))
from config import (
    FEATURE_MATRIX_PQ, PROCESSED_DIR, TARGET_COLUMN,
    N_SPATIAL_FOLDS
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

OUTPUT_DIR = PROCESSED_DIR / "training_dataset"
OUTPUTS_DIR = Path(__file__).resolve().parent.parent.parent / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    """Load feature matrix and return labelled rows with geometry."""
    df = pd.read_parquet(path)
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)
    log.info(f"Loaded: {df.shape} labelled points")
    return df

def load_saved_folds() -> np.ndarray:
    """Load the exact spatial folds used during model training."""
    fold_file = OUTPUT_DIR / "spatial_folds.npy"

    if not fold_file.exists():
        raise FileNotFoundError(
            f"Could not find saved folds: {fold_file}"
    )

    folds = np.load(fold_file)

    # Convert to human-readable fold numbering (1–5)
    folds = folds + 1

    log.info(
        f"Loaded spatial folds from: {fold_file}"
    )

    return folds

def extract_coordinates(df: pd.DataFrame) -> np.ndarray:
    """Extract easting, northing from geometry_wkt."""
    coords = np.array([
        [loads(wkt).x, loads(wkt).y]
        for wkt in df["geometry_wkt"]
    ])
    return coords


def assign_spatial_blocks(coords: np.ndarray, n_blocks: int = 5) -> np.ndarray:
    """Assign spatial blocks based on quantiles of easting coordinate."""
    fold_ids = pd.qcut(coords[:, 0], q=n_blocks, labels=False, duplicates='drop')
    return fold_ids


def plot_spatial_folds(coords: np.ndarray,
                        fold_ids: np.ndarray,
                        labels: np.ndarray):
    """
    Create publication-ready fold map.

    Fold colour = CV fold
    Black stars = known VMS deposits
    """

    fig, ax = plt.subplots(figsize=(14, 10))

    fold_colors = [
        '#FF6B6B',
        '#4ECDC4',
        '#45B7D1',
        '#FFA07A',
        '#98D8C8'
    ]

    unique_folds = np.unique(fold_ids)

    # Plot folds
    for fold in unique_folds:

        mask = fold_ids == fold

        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=35,
            alpha=0.75,
            c=fold_colors[int(fold)-1],
            edgecolors='none',
            label=f"Fold {int(fold)}"
        )

    # Overlay VMS deposits
    vms_mask = labels == 1

    ax.scatter(
        coords[vms_mask, 0],
        coords[vms_mask, 1],
        marker='*',
        s=140,
        c='black',
        edgecolors='white',
        linewidth=0.7,
        zorder=10,
        label='Known VMS deposits'
    )

    # Add fold boundaries
    quantiles = np.quantile(
        coords[:, 0],
        [0.2, 0.4, 0.6, 0.8]
    )

    for boundary in quantiles:
        ax.axvline(
            boundary,
            color='gray',
            linestyle='--',
            alpha=0.4,
            linewidth=1
        )

    ax.set_xlabel(
        "Easting (m)",
        fontsize=12,
        fontweight="bold"
    )

    ax.set_ylabel(
        "Northing (m)",
        fontsize=12,
        fontweight="bold"
    )

    ax.set_title(
        "Spatial Cross-Validation Fold Assignment",
        fontsize=14,
        fontweight="bold"
    )

    ax.grid(True, alpha=0.2)

    handles, labels_ = ax.get_legend_handles_labels()

    ax.legend(
        handles,
        labels_,
        loc="best",
        framealpha=0.95
    )

    plt.tight_layout()

    output_path = (
        OUTPUTS_DIR /
        "spatial_cv_folds_map.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    log.info(
        f"✅ Spatial folds map saved: {output_path}"
    )

    plt.close()

def plot_fold_distribution(fold_ids: np.ndarray, labels: np.ndarray):
    """Create bar chart of class distribution across folds."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    unique_folds = np.unique(fold_ids)
    fold_counts = []
    fold_vms = []
    fold_bg = []
    
    for fold in unique_folds:
        fold_mask = fold_ids == fold
        total = fold_mask.sum()
        vms = ((fold_ids == fold) & (labels == 1)).sum()
        bg = ((fold_ids == fold) & (labels == 0)).sum()
        
        fold_counts.append(total)
        fold_vms.append(vms)
        fold_bg.append(bg)
    
    # Bar chart 1: Total samples per fold
    ax1.bar(unique_folds, fold_counts, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Fold ID', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Sample Count', fontsize=11, fontweight='bold')
    ax1.set_title('Total Samples per Fold', fontsize=12, fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.3)
    for i, (fold, count) in enumerate(zip(unique_folds, fold_counts)):
        ax1.text(fold, count + 5, str(count), ha='center', va='bottom', fontweight='bold')
    
    # Bar chart 2: Stacked bar showing class balance
    x_pos = np.arange(len(unique_folds))
    width = 0.6
    ax2.bar(x_pos, fold_bg, width, label='Background (0)', color='#A0A0A0', alpha=0.8, edgecolor='black')
    ax2.bar(x_pos, fold_vms, width, bottom=fold_bg, label='VMS (1)', color='#FF6B6B', alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Fold ID', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Sample Count', fontsize=11, fontweight='bold')
    ax2.set_title('Class Distribution per Fold', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(unique_folds)
    ax2.legend(loc='best', framealpha=0.95)
    ax2.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUTS_DIR / "spatial_cv_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    log.info(f"✅ Fold distribution chart saved: {output_path}")
    plt.close()


def print_fold_summary(fold_ids: np.ndarray, labels: np.ndarray):
    """Print detailed summary of fold assignments."""
    log.info("\n" + "="*70)
    log.info("SPATIAL CROSS-VALIDATION FOLD SUMMARY")
    log.info("="*70)
    
    unique_folds = np.unique(fold_ids)
    for fold in unique_folds:
        fold_mask = fold_ids == fold
        total = fold_mask.sum()
        vms = ((fold_ids == fold) & (labels == 1)).sum()
        bg = ((fold_ids == fold) & (labels == 0)).sum()
        vms_ratio = (vms / total * 100) if total > 0 else 0
        
        log.info(f"\nFold {fold}:")
        log.info(f"  Total samples  : {total:6d}")
        log.info(f"  Background (0) : {bg:6d} ({100*bg/total:5.1f}%)")
        log.info(f"  VMS sites (1)  : {vms:6d} ({vms_ratio:5.1f}%)")
    
    log.info("\n" + "="*70 + "\n")

def export_fold_polygons(coords: np.ndarray, fold_ids: np.ndarray):
    """
    Export spatial cross-validation fold polygons as a GeoPackage.
    Fold boundaries are derived directly from the sample locations
    assigned to each fold, ensuring exact consistency with the
    cross-validation procedure used during model training.

    Parameters
    ----------
    coords : np.ndarray
        Array of point coordinates with shape (n_samples, 2)
        where column 0 = easting and column 1 = northing.

    fold_ids : np.ndarray
        Spatial fold assignments for each sample.
    """

    from shapely.geometry import Polygon

    # Separate coordinates for clarity
    x = coords[:, 0]
    y = coords[:, 1]

    # Full north-south extent of study area
    ymin = y.min()
    ymax = y.max()

    polygons = []

    unique_folds = np.unique(fold_ids)

    log.info("\nSpatial Fold Extents")
    log.info("-" * 60)

    for fold in unique_folds:
        fold_mask = fold_ids == fold

        # Actual fold boundaries from assigned samples
        xmin = x[fold_mask].min()
        xmax = x[fold_mask].max()
        
        width_m = xmax - xmin

        log.info(
            f"Fold {int(fold)+1}: "
            f"Xmin={xmin:.2f}, "
            f"Xmax={xmax:.2f}, "
            f"Width={width_m:.2f} m"
        )

        poly = Polygon([
            (xmin, ymin),
            (xmax, ymin),
            (xmax, ymax),
            (xmin, ymax),
            (xmin, ymin)
        ])

        polygons.append({
            "fold_id": int(fold) + 1, # Display folds as 1–5
            "xmin": xmin,
            "xmax": xmax,
            "width_m": width_m,
            "geometry": poly
        })

    gdf_poly = gpd.GeoDataFrame(
        polygons,
        crs="EPSG:2953"
    )

    output_path = OUTPUTS_DIR / "spatial_cv_fold_polygons.gpkg"

    gdf_poly.to_file(
        output_path,
        driver="GPKG",
        index=False
    )

    log.info(
        f"\n✅ Spatial fold polygons exported: {output_path}"
    )

    return gdf_poly

def export_spatial_folds_gpkg(df: pd.DataFrame, fold_ids: np.ndarray):
    """Export spatial folds to GeoPackage format for QGIS."""
    try:
        # Create geometries from WKT
        geometries = [loads(wkt) for wkt in df["geometry_wkt"]]
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            {
                'fold_id': fold_ids.astype(int),
                'label': df["label"].values,
                'label_name': df["label"].map({0: 'Background', 1: 'VMS'}).values,
            },
            geometry=geometries,
            crs='EPSG:2953'
        )
        
        # Save to GeoPackage
        output_path = OUTPUTS_DIR / "spatial_cv_folds.gpkg"
        gdf.to_file(output_path, driver='GPKG', index=False)
        log.info(f"✅ GeoPackage exported: {output_path}")
        log.info(f"   Features: {len(gdf)}, CRS: {gdf.crs}")
        
    except Exception as e:
        log.error(f"Failed to export GeoPackage: {e}")
        log.warning("Ensure geopandas is installed: pip install geopandas")


def main():
    log.info("═══ Spatial CV Fold Visualization ═══\n")
    
    # Load data
    df = load_data(FEATURE_MATRIX_PQ)
    coords = extract_coordinates(df)
    labels = df["label"].values
    
    # Assign spatial blocks
    log.info(
        "Loading saved spatial folds used during model training..."
    )
    fold_ids = load_saved_folds()

    unique, counts = np.unique(
        fold_ids,
        return_counts=True
    )

    for fold, count in zip(pd.unique(fold_ids), counts):
        log.info(
            f" Fold {int(fold)}: {count} samples"
        )
    
    # Print summary
    print_fold_summary(fold_ids, labels)
    
    # Create visualizations
    log.info("Creating visualizations...\n")
    plot_spatial_folds(coords, fold_ids, labels)
    plot_fold_distribution(fold_ids, labels)
    
    # Export to GeoPackage
    log.info("\nExporting spatial folds to GeoPackage...")
    export_spatial_folds_gpkg(df, fold_ids)

    # Export fold polygons
    log.info("\nExporting fold polygons...")
    export_fold_polygons(coords, fold_ids)


if __name__ == "__main__":
    main()
