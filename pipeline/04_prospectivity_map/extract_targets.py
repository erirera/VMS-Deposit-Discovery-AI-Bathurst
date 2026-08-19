"""
extract_targets.py

Converts high-prospectivity raster cells into ranked target polygons.
"""

import sys
import argparse
from pathlib import Path

# Add pipeline directory to path so we can import config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rasterio
import geopandas as gpd
import numpy as np

from rasterio.features import shapes
from rasterio.mask import mask
from shapely.geometry import shape, mapping

from config import (
    RF_PROSPECTIVITY_TIFF,
    OUTPUTS_DIR
)

PI_THRESHOLD = 0.70

DEFAULT_OUT_GPKG = OUTPUTS_DIR / "rf_targets.gpkg"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract ranked target polygons from a prospectivity GeoTIFF."
    )
    parser.add_argument(
        "--input", "-i", type=Path, default=RF_PROSPECTIVITY_TIFF,
        help=f"Prospectivity GeoTIFF (default: {RF_PROSPECTIVITY_TIFF.name})"
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=DEFAULT_OUT_GPKG,
        help=f"Output GeoPackage (default: {DEFAULT_OUT_GPKG.name})"
    )
    parser.add_argument(
        "--threshold", type=float, default=PI_THRESHOLD,
        help=f"Minimum prospectivity probability (default: {PI_THRESHOLD})"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Prospectivity raster not found: {args.input}")
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be between 0 and 1")

    with rasterio.open(args.input) as src:

        prob = src.read(1)

        # Extract high-prospectivity cells
        target_mask = prob > args.threshold

        geoms = []

        for geom, value in shapes(
            target_mask.astype(np.uint8),
            transform=src.transform
        ):

            if value == 1:
                geoms.append(shape(geom))

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            geometry=geoms,
            crs=src.crs
        )

        # Area statistics
        gdf["area_m2"] = gdf.area
        gdf["area_km2"] = gdf.area / 1_000_000

        # Calculate prospectivity statistics
        mean_pi = []
        max_pi = []

        for geom in gdf.geometry:

            out_image, _ = mask(
                src,
                [mapping(geom)],
                crop=True,
                filled=False
            )

            vals = out_image[0]
            vals = vals[np.isfinite(vals)]

            if len(vals) > 0:
                mean_pi.append(float(np.mean(vals)))
                max_pi.append(float(np.max(vals)))
            else:
                mean_pi.append(np.nan)
                max_pi.append(np.nan)

        gdf["mean_pi"] = mean_pi
        gdf["max_pi"] = max_pi

        # Rank targets by maximum prospectivity while also creating a
        # complementary mean-prospectivity ranking for prioritisation.
        gdf = (
            gdf.sort_values(
                "max_pi",
                ascending=False
            )
            .reset_index(drop=True)
        )

        gdf["rank_by_max_pi"] = np.arange(
            1,
            len(gdf) + 1
        )

        # Target IDs (kept consistent with the max-prospectivity ordering)
        gdf["target_id"] = [
            f"TGT_{i:03d}"
            for i in range(1, len(gdf) + 1)
        ]

        mean_rank_lookup = {
            row["target_id"]: idx + 1
            for idx, row in (
                gdf.sort_values(
                    "mean_pi",
                    ascending=False
                )
                .reset_index(drop=True)
                .iterrows()
            )
        }

        gdf["rank_by_mean_pi"] = gdf["target_id"].map(mean_rank_lookup)

        # Backward-compatible alias: legacy code expected a single rank column.
        gdf["rank"] = gdf["rank_by_max_pi"]

        # Export GeoPackage
        args.output.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(
            args.output,
            driver="GPKG"
        )

    print(
        f"Saved: {args.output}"
    )


if __name__ == "__main__":
    main()