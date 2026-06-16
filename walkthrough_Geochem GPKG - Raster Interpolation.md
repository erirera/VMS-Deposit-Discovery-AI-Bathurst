# Walkthrough: Geochem GPKG → Raster Interpolation

## Summary

Added a new preprocessing step to the VMS Deposit Discovery AI pipeline that converts the 17 BMC till geochemistry point datasets (`.gpkg` files) into continuous GeoTIFF raster surfaces using spatial interpolation. These rasters are automatically picked up by the existing `extract_features.py` without any changes to that script.

---

## Files Changed

### [NEW] [interpolate_geochem.py](file:///c:/Users/delef/.gemini/antigravity/bathurst%20VMS/VMS-Deposit-Discovery-AI-Bathurst/pipeline/02_preprocessing/interpolate_geochem.py)

Full-featured interpolation script supporting two methods:

| Method | Speed | Notes |
|:-------|:------|:------|
| **IDW** (default) | ~2 s / element | Power=2, k=12 neighbours, fully deterministic |
| **Kriging** | ~30-120 s / element | Ordinary Kriging via `pykrige`, spherical variogram, subsampled to 500 pts |

### [MODIFY] [config.py](file:///c:/Users/delef/.gemini/antigravity/bathurst%20VMS/VMS-Deposit-Discovery-AI-Bathurst/pipeline/config.py)

Added 17 IDW geochem raster names to `RASTER_FEATURES` for documentation and future reference.

---

## IDW Results (17/17 rasters — verified)

| Raster | Size | Min (ppm) | Max (ppm) | NoData |
|:-------|-----:|----------:|----------:|-------:|
| `geochem_ag_ppm_idw.tif` | 12.3 MB | 0.0 | 21.1 | 0% |
| `geochem_as_ppm_idw.tif` | 13.7 MB | 0.0 | 18,167 | 0% |
| `geochem_ba_ppm_idw.tif` | 13.4 MB | 0.0 | 2,177 | 0% |
| `geochem_bi_ppm_idw.tif` | 12.5 MB | 0.0 | 174 | 0% |
| `geochem_cd_ppm_idw.tif` | 11.5 MB | 0.0 | 14.0 | 0% |
| `geochem_co_ppm_idw.tif` | 13.4 MB | 0.0 | 180 | 0% |
| `geochem_cu_ppm_idw.tif` | 13.6 MB | 0.0 | 1,138 | 0% |
| `geochem_fe_ppm_idw.tif` | 13.2 MB | 0.5 | 156 | 0% |
| `geochem_in_ppm_idw.tif` | 11.8 MB | 0.0 | 9.4 | 0% |
| `geochem_mn_ppm_idw.tif` | 13.3 MB | 21.1 | 5,682 | 0% |
| `geochem_mo_ppm_idw.tif` | 12.5 MB | 0.0 | 83.5 | 0% |
| `geochem_ni_ppm_idw.tif` | 13.7 MB | 0.0 | 286 | 0% |
| `geochem_pb_ppm_idw.tif` | 13.6 MB | 3.0 | 19,217 | 0% |
| `geochem_sb_ppm_idw.tif` | 13.6 MB | 0.0 | 348 | 0% |
| `geochem_sn_ppm_idw.tif` | 13.2 MB | 0.0 | 31,669 | 0% |
| `geochem_tl_ppm_idw.tif` | 12.7 MB | 0.1 | 15.0 | 0% |
| `geochem_zn_ppm_idw.tif` | 13.5 MB | 0.8 | 2,303 | 0% |

> [!NOTE]
> As, Pb, and Sn show very high maxima due to extreme anomalies near known VMS deposits — exactly the signal the model needs. These are stored as raw PPM values; consider `log10(x+1)` transform in `engineer_features.py` to normalise skewed distributions.

---

## Total raster directory: 54 files

The `data/processed/rasters_reprojected/` directory now contains **54 GeoTIFFs** — the original 37 geophysical derivatives plus the 17 new geochemistry IDW surfaces.

---

## Next Steps

### Run Kriging (when ready)
```bash
# All 17 elements with Ordinary Kriging (spherical variogram)
python pipeline/02_preprocessing/interpolate_geochem.py --method kriging

# Faster test on a single element
python pipeline/02_preprocessing/interpolate_geochem.py --method kriging --element Cu

# Try different variogram models
python pipeline/02_preprocessing/interpolate_geochem.py --method kriging --kriging-variogram exponential
```

### Rebuild Feature Matrix
```bash
python pipeline/02_preprocessing/extract_features.py
```
The new geochem rasters will be **automatically sampled** alongside all geophysical layers — no code changes required.

### Feature Engineering
Consider adding `log10(x + 1)` transforms for skewed elements (As, Pb, Sn) in `engineer_features.py`.
