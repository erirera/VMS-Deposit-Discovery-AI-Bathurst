import sys
import logging
from pathlib import Path
import geopandas as gpd
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("compile_geochem")

elements = [
    "Ag", "As", "Ba", "Bi", "Cd", "Co", "Cu", "Fe", "In", "Mn", "Mo", "Ni", "Pb", "Sb", "Sn", "Tl", "Zn"
]

def main():
    log.info("=== Compiling 17 Geochemistry Elements ===")
    
    workspace = Path(__file__).resolve().parents[2]
    rasters_dir = workspace / "data" / "raw" / "rasters"
    output_path = workspace / "data" / "raw" / "nb_till_geochemistry.gpkg"
    
    data = {}
    for el in elements:
        gpkg_path = rasters_dir / f"bmc_{el}.gpkg"
        if not gpkg_path.exists():
            log.error(f"Required element file not found: {gpkg_path}")
            sys.exit(1)
            
        log.info(f"Loading {el}...")
        gdf = gpd.read_file(gpkg_path, engine="pyogrio")
        
        # Round coordinates to match locations
        gdf['x_rnd'] = gdf.geometry.x.round(0)
        gdf['y_rnd'] = gdf.geometry.y.round(0)
        gdf['coord_key'] = gdf['x_rnd'].astype(str) + '_' + gdf['y_rnd'].astype(str)
        
        # Deduplicate
        gdf = gdf.drop_duplicates(subset=['coord_key'])
        data[el] = gdf[['coord_key', 'VALUE', 'geometry']]
        
    # Gather all unique coordinate keys
    all_keys = set()
    for el in elements:
        all_keys.update(data[el]['coord_key'])
        
    log.info(f"Total unique geochemistry sample locations: {len(all_keys)}")
    
    # Merge outer
    merged = pd.DataFrame({'coord_key': list(all_keys)})
    for el in elements:
        df_el = data[el].rename(columns={'VALUE': f'{el.lower()}_ppm', 'geometry': f'geom_{el}'})
        merged = merged.merge(df_el, on='coord_key', how='left')
        
    # Extract non-null geometry for each row
    log.info("Aligning geometries...")
    geoms = []
    for idx, row in merged.iterrows():
        g = None
        for el in elements:
            val = row[f'geom_{el}']
            if pd.notna(val):
                g = val
                break
        geoms.append(g)
        
    merged['geometry'] = geoms
    merged = gpd.GeoDataFrame(merged, geometry='geometry', crs='EPSG:2953')
    
    # Select columns
    cols_to_keep = [f'{el.lower()}_ppm' for el in elements] + ['geometry']
    merged = merged[cols_to_keep]
    
    log.info(f"Saving compiled GPKG to: {output_path}")
    merged.to_file(output_path, driver="GPKG", layer="till_geochemistry")
    log.info("Successfully compiled all 17 elements into nb_till_geochemistry.gpkg!")

if __name__ == "__main__":
    main()
