# Walkthrough: Geochem GPKG → Raster Interpolation (IDW + Kriging)

## Summary

Added a new preprocessing step that converts all 17 BMC till geochemistry point
datasets into continuous GeoTIFF raster surfaces using two interpolation methods.
Both sets of rasters land in `data/processed/rasters_reprojected/` and are
automatically sampled by the existing `extract_features.py`.

---

## Files Changed

### [NEW] [interpolate_geochem.py](file:///c:/Users/delef/.gemini/antigravity/bathurst%20VMS/VMS-Deposit-Discovery-AI-Bathurst/pipeline/02_preprocessing/interpolate_geochem.py)

Supports two methods via `--method {idw,kriging}`:

| Method | RAM / element | Time / element | Notes |
|:-------|:-------------|:--------------|:------|
| **IDW** | ~200 MB | ~2 s | Power=2, k=12 neighbours, deterministic |
| **Kriging** | ~40 MB/chunk | ~50 s | Spherical variogram, 200 control pts, 25 k cell chunks |

**OOM fix applied:** pykrige's default vectorised executor tried to allocate 11 GB.
Fixed by switching to chunked `ok.execute("points", ...)` calls (25,000 cells/chunk → ~40 MB peak).

### [MODIFY] [config.py](file:///c:/Users/delef/.gemini/antigravity/bathurst%20VMS/VMS-Deposit-Discovery-AI-Bathurst/pipeline/config.py)

Added 17 IDW geochem raster names to `RASTER_FEATURES`.

---

## IDW vs Kriging Results

| Element | IDW max (ppm) | IDW MB | Kriging max (ppm) | Kriging MB |
|:--------|-------------:|-------:|------------------:|-----------:|
| Ag | 21.1 | 12.3 | 1.3 | 3.4 |
| As | 18,167 | 13.7 | 77.6 | 0.1 |
| Ba | 2,177 | 13.4 | 567.1 | 11.2 |
| Bi | 174 | 12.5 | 1.0 | 0.2 |
| Cd | 14.0 | 11.5 | 2.0 | 12.0 |
| Co | 180 | 13.4 | 15.8 | 0.1 |
| Cu | 1,138 | 13.6 | 39.9 | 0.1 |
| Fe | 156 | 13.2 | 6.2 | 4.8 |
| In | 9.4 | 11.8 | 1.0 | 0.4 |
| Mn | 5,682 | 13.3 | 868.7 | 12.0 |
| Mo | 83.5 | 12.5 | 4.5 | 2.7 |
| Ni | 286 | 13.7 | 29.8 | 11.1 |
| Pb | 19,217 | 13.6 | 99.5 | 0.1 |
| Sb | 348 | 13.6 | 4.7 | 1.6 |
| Sn | 31,669 | 13.2 | 63.2 | 11.9 |
| Tl | 15.0 | 12.7 | 3.9 | 5.5 |
| Zn | 2,303 | 13.5 | 429.8 | 10.4 |

**Total rasters in dir: 71** (37 geophysics + 17 IDW geochem + 17 Kriging geochem)

---

## ⚠️ Kriging Quality Warning

Several Kriging outputs show suspiciously **narrow value ranges** compared to IDW:

| Element | IDW range | Kriging range | Issue |
|:--------|----------:|:-------------|:------|
| Co | 0–180 | 15.8–15.8 | Near-constant (flat) |
| Cu | 0–1,138 | 36.7–39.9 | Near-constant |
| Bi | 0–174 | 1.0–1.0 | Constant |
| As | 0–18,167 | 40.6–77.6 | Severely smoothed |
| Pb | 3–19,217 | 55.0–99.5 | Severely smoothed |

**Root cause:** With only 200 subsampled control points, the fitted variogram range
often exceeds the entire study area extent, making the kriging weights nearly equal
for all grid cells → near-constant prediction (mean of the subsample).

**Recommendation for ML training: use the IDW rasters.** They preserve spatial
anomaly contrast (which is exactly the VMS pathfinder signal), while the Kriging
outputs at this configuration over-smooth the anomalies into the background.

**To improve Kriging quality (future work):**
```bash
# Use more control points (slower, more memory — test on a machine with 32+ GB RAM)
python pipeline/02_preprocessing/interpolate_geochem.py \
    --method kriging --kriging-max-points 1000 --element Cu

# Or use a coarser resolution (10x fewer cells, much faster)
python pipeline/02_preprocessing/interpolate_geochem.py \
    --method kriging --resolution 200 --overwrite
```

---

## Next Steps

### 1. Rebuild the feature matrix (uses IDW rasters automatically)
```bash
python pipeline/02_preprocessing/extract_features.py
```

### 2. Feature engineering — log-transform skewed elements
In `engineer_features.py`, consider `log10(x + 1)` for As, Pb, Sn, Mn which
span 3–5 orders of magnitude.

### 3. Improve Kriging (optional)
Run at 200 m resolution to get statistically valid Kriging surfaces without
needing subsampling:
```bash
python pipeline/02_preprocessing/interpolate_geochem.py \
    --method kriging --resolution 200 --kriging-max-points 500 --overwrite
```
