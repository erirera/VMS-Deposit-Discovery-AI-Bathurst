"""
download_nrcan_mag.py
─────────────────────
Downloads NRCan / NB DNRED airborne geophysical survey data for the Bathurst
Mining Camp region. Attempts known direct-download URLs first; prints detailed
step-by-step manual instructions if automated download is not possible.

Primary sources (verified April 2026, no registration required):
  ① NB DNRED — Compiled Aeromagnetic Data (province-wide, recommended)
     https://www2.gnb.ca/content/gnb/en/departments/erd/open-data/geophysical-data/aeromagnetic-data-compiled.html

  ② NB DNRED — Individual Aeromagnetic Surveys (higher local resolution)
     https://www2.gnb.ca/content/gnb/en/departments/erd/open-data/geophysical-data/aeromagnetic-data-individual.html

  ③ NB DNRED — Radiometric Data (Compiled)
     https://www2.gnb.ca/content/gnb/en/departments/erd/open-data/geophysical-data/radiometric-data-compiled.html

  ④ NRCan Geoscience Data Repository (national archive)
     https://gdr.agg.nrcan.gc.ca/gdrdap/dap/portal

See pipeline/DATA_ACQUISITION_GUIDE.md for full step-by-step instructions.

Usage:
    python pipeline/01_data_download/download_nrcan_mag.py
"""

import sys
import logging
from pathlib import Path
import requests
from tqdm import tqdm

# Ensure UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PIPELINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_DIR))
from config import RASTERS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ── Candidate Automated Download URLs ─────────────────────────────────────────
# NB DNRED and NRCan portals deliver raster files through authenticated download
# sessions — direct URL automation is not reliably supported.
# These FTP paths are provided as a best-effort attempt only.
CANDIDATE_DOWNLOADS: dict[str, dict] = {
    # NRCan FTP mirror — browse for NB surveys at:
    # https://ftp.maps.canada.ca/pub/nrcan_rncan/Mining-industry_Industrie-miniere/
    # geophysics-geophysique/surveys/
    #
    # Uncomment and update URL if you locate a direct GeoTIFF link:
    # "mag_tmi_nb_compiled.tif": {
    #     "url": "https://ftp.maps.canada.ca/pub/.../TMI_grid.tif",
    #     "description": "Total Magnetic Intensity — NB compiled",
    # },
}

# ── Manual Download Instructions ──────────────────────────────────────────────
MANUAL_INSTRUCTIONS = """
==============================================================================
     MANUAL DOWNLOAD REQUIRED -- Aeromagnetic & Geophysical Raster Data      
==============================================================================

The NB DNRED and NRCan portals deliver raster grids through session-based
download buttons that cannot be reliably automated. Follow the steps below,
then place all files in:  data/raw/rasters/

------------------------------------------------------------------------------
 OPTION A (Recommended) -- NB DNRED Compiled Grids   [No account needed]
------------------------------------------------------------------------------

 START HERE:
   https://www2.gnb.ca/content/gnb/en/departments/erd/open-data/geophysical-data.html

 STEP 1 -- AEROMAGNETICS
   Click -> "Aeromagnetic Data - Compiled"
   URL: https://www2.gnb.ca/content/gnb/en/departments/erd/open-data/geophysical-data/aeromagnetic-data-compiled.html
   Download:
     * RMI (Residual Magnetic Intensity) grid  -> rename to: mag_rmi_bmc_compiled.tif 
     * FVD (First Vertical Derivative) grid -> rename to: mag_fvd_bmc_compiled.tif
   Format: GeoTIFF preferred; ASCII grid (.asc) also accepted
   Place in:  data/raw/rasters/

 STEP 2 -- RADIOMETRICS
   Click -> "Radiometric Data - Compiled"
   URL: https://www2.gnb.ca/content/gnb/en/departments/erd/open-data/geophysical-data/radiometric-data-compiled.html
   Download (separate file for each channel):
     * Potassium K%   -> rename to: rad_k_bmc_compiled.tif
     * Thorium Th ppm -> rename to: rad_th_bmc_compiled.tif
     * Uranium U ppm  -> rename to: rad_u_bmc_compiled.tif
   Place in:  data/raw/rasters/

 STEP 3 -- GRAVITY GRADIOMETRY
   Click -> "Gravity Gradiometry Data - Compiled"
   URL: https://www2.gnb.ca/content/gnb/en/departments/erd/open-data/geophysical-data/gravity-gradiometry-compiled.html
   Download the Bouguer anomaly (or Gz / vertical gradient) grid
   -> rename to: gravity_bouguer_bmc.tif
   Place in:  data/raw/rasters/

 STEP 4 -- FOR HIGHER RESOLUTION (optional, individual surveys)
   Aeromagnetic: https://www2.gnb.ca/content/gnb/en/departments/erd/open-data/geophysical-data/aeromagnetic-data-individual.html
   Radiometric:  https://www2.gnb.ca/content/gnb/en/departments/erd/open-data/geophysical-data/radiometric-data-individual.html
   Gravity:      https://www2.gnb.ca/content/gnb/en/departments/erd/open-data/geophysical-data/gravity-gradiometry-individual.html
   -> Look for surveys in the Bathurst, Nepisiguit, Tetagouche, Restigouche area
   -> Lat: 47.0 N - 48.0 N  |  Lon: 65.5 W - 67.0 W

------------------------------------------------------------------------------
 OPTION B -- NRCan Geoscience Data Repository (national archive)
------------------------------------------------------------------------------

 1. Open: https://gdr.agg.nrcan.gc.ca/gdrdap/dap/portal
    (alternative: https://geophysical-data.canada.ca/)

 2. Click "Search by Province" -> select "New Brunswick"
    OR click "Search by Map", draw bounding box over Bathurst area:
    Lat 47.0 N - 48.0 N  |  Lon 65.5 W - 67.0 W

 3. Browse survey list -- look for surveys covering:
    Keywords: Bathurst, Tetagouche, Nepisiguit, Gloucester, Restigouche

 4. Click each relevant survey -> Download
    Preferred format: GeoTIFF (.tif)
    Alternative: ASCII grid (.grd, .asc)

 5. If only Geosoft binary format (.grd / .ers) is available:
    -> Convert to GeoTIFF using QGIS:
        Raster -> Miscellaneous -> Translate (Convert Format)
    -> Or with GDAL command line (free, no Oasis Montaj license needed):
        gdal_translate -of GTiff input.grd output.tif

 6. Rename and place in: data/raw/rasters/

------------------------------------------------------------------------------
 OPTION C -- NRCan Open Government Portal (alternative search)
------------------------------------------------------------------------------

 1. Open: https://open.canada.ca/en/open-data
 2. Search: "New Brunswick aeromagnetic geophysical survey"
 3. Filter: Organisation = "Natural Resources Canada"
 4. Download relevant GeoTIFF packages

------------------------------------------------------------------------------
 EXPECTED FILE STRUCTURE after manual download:
------------------------------------------------------------------------------

   data/raw/rasters/
   +-- mag_rmi_bmc_compiled.tif    <- Residual RMI BMC
   +-- rad_k_bmc_compiled.tif      <- Potassium K% radiometric BMC
   +-- rad_th_bmc_compiled.tif     <- Thorium ppm radiometric BMC
   +-- rad_u_bmc_compiled.tif      <- Uranium ppm radiometric BMC
   +-- gravity_bouguer_bmc.tif     <- Bouguer anomaly / gravity gradiometry BMC

 After placing files, run:
   python pipeline/02_preprocessing/reproject_grids.py

------------------------------------------------------------------------------
 CONTACT -- If portals are down or datasets not found:
------------------------------------------------------------------------------

 NB Geological Survey Branch
   Phone: 506-453-2206  |  Email: minerals@gnb.ca
   Web: https://www2.gnb.ca/content/gnb/en/departments/erd/energy/content/minerals.html

 NRCan GDR Support
   Email: gdr@nrcan.gc.ca
   Web: https://gdr.agg.nrcan.gc.ca/

 Full guide: pipeline/DATA_ACQUISITION_GUIDE.md
==============================================================================
"""


def check_existing_rasters() -> list[str]:
    """List any rasters already present in the rasters directory."""
    if not RASTERS_DIR.exists():
        return []
    return [p.name for p in RASTERS_DIR.glob("*.tif")]


def try_download(name: str, url: str, dest: Path) -> bool:
    """Attempt to download a file. Returns True on success."""
    if dest.exists():
        log.info(f"  Already exists — skipping: {dest.name}")
        return True
    log.info(f"  Trying: {url}")
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=name
        ) as bar:
            for chunk in r.iter_content(8192):
                f.write(chunk)
                bar.update(len(chunk))
        log.info(f"  ✅ Downloaded: {dest}")
        return True
    except Exception as e:
        log.warning(f"  ✗  Failed ({e})")
        if dest.exists():
            dest.unlink()
        return False


def main():
    log.info("═══ Geophysical Raster Download ═══")
    RASTERS_DIR.mkdir(parents=True, exist_ok=True)

    # Check what's already there
    existing = check_existing_rasters()
    if existing:
        log.info(f"  Found {len(existing)} raster(s) already in {RASTERS_DIR}:")
        for name in existing:
            log.info(f"    ✓ {name}")

    # Attempt any configured automated downloads
    success_count = 0
    for name, meta in CANDIDATE_DOWNLOADS.items():
        dest = RASTERS_DIR / name
        log.info(f"\n{meta['description']}")
        ok = try_download(name, meta["url"], dest)
        if ok:
            success_count += 1

    # Expected files
    EXPECTED = [
        "mag_tmi_nb_compiled.tif",
        "mag_fvd_nb_compiled.tif",
        "rad_k_nb_compiled.tif",
        "rad_th_nb_compiled.tif",
        "rad_u_nb_compiled.tif",
        "gravity_bouguer_nb.tif",
    ]
    missing = [f for f in EXPECTED if not (RASTERS_DIR / f).exists()]

    if missing:
        log.warning(f"\n⚠️  {len(missing)} expected raster(s) not yet present:")
        for name in missing:
            log.warning(f"    missing: {name}")
        print(MANUAL_INSTRUCTIONS)
    else:
        log.info(
            "\n✅ All expected rasters are present in data/raw/rasters/\n"
            "   Run: python pipeline/02_preprocessing/reproject_grids.py"
        )


if __name__ == "__main__":
    main()
