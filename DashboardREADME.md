# 🪨 VMS Deposit Discovery AI — Interactive Dashboard

> **Research Proposal**  
> Machine Learning Prospectivity Mapping in the Bathurst Mining Camp, New Brunswick, Canada

---

## Overview

This repository contains an interactive web dashboard visualising the research proposal for using **machine learning to predict undiscovered Volcanogenic Massive Sulphide (VMS) mineral deposits** in the [Bathurst Mining Camp (BMC)](https://en.wikipedia.org/wiki/Bathurst_Mining_Camp) — one of Canada's most intensively studied, yet still largely underexplored, mineral districts.

The proposal calls for integrating open geophysical and geochemical data from **NRCan** and the **NB Geological Survey** into a Random Forest / XGBoost classifier, producing a spatially continuous prospectivity map with SHAP-based explainability.

### Research Gap

> No peer-reviewed, open-science ML prospectivity paper exists for the Bathurst Mining Camp using the full NRCan geophysical compilation + NB till geochemistry. This is the open-science gap. ✨

---

## Dashboard Features

| Feature | Description |
|---|---|
| 🗺️ Interactive Map | Leaflet map centered on the BMC (~47.5°N, 66°W) |
| 🟠 VMS Deposits | 45 known deposit locations (positive training labels), pulsing markers |
| ⚫ Barren Drill Holes | 250+ negative training labels reflecting real-world class imbalance |
| 🟦 BMC Boundary | ~3,800 km² camp outline |
| 🌡️ AI Prospectivity Heatmap | Toggle-able heatmap layer simulating model output (blue → red) |
| 📊 Stats Panel | Live stats: total area, known deposits, AI target zone count |
| 🎨 Dark Glassmorphism UI | Smooth GSAP animations, custom toggle controls, responsive layout |

---

## Getting Started

**No build tools or server needed.** Just open the file in any modern browser:

```bash
# Clone the repo
git clone https://github.com/erirera/vms-discovery-dashboard.git
cd vms-discovery-dashboard

# Open the dashboard
start index.html       # Windows
open index.html        # macOS
xdg-open index.html    # Linux
```

> ⚠️ **Internet connection required** — the dashboard loads Leaflet.js, GSAP, and CartoDB map tiles from CDNs.

---

## Data Sources

All data sources referenced in the underlying research are freely available:

| Dataset | Source |
|---|---|
| Till Geochemistry (Zn, Pb, Cu, Ag, Au, As) | [NB Geological Survey Open Data](https://www2.gnb.ca/content/gnb/en/departments/erd/energy/content/minerals/content/geology_data.html) |
| Airborne Magnetics (TMI, FVD) | [NRCan Geoscience Repository](https://geoscan.nrcan.gc.ca/) |
| Airborne Radiometrics (K%, Th, U) | NRCan Airborne Geophysical Surveys |
| Airborne EM (Conductivity) | NRCan Airborne Geophysical Surveys |
| Gravity (Bouguer Anomaly) | NRCan Gravity Programme |
| VMS Deposit Locations | [NB Metallic Minerals Database](https://www2.gnb.ca/content/gnb/en/departments/erd/energy/content/minerals.html) |
| Barren Drill Records | NB GSB / SEDAR |

---

## Proposed Methodology

```
Phase 1 (Months 6–7): Data Preparation
  └── Reproject all grids → NAD83 / NB Double Stereographic (EPSG:2953) @ 100m
  └── Extract geophysical values at till geochemistry sample points (~20 features)
  └── Engineer derived features (TMI gradient, analytic signal, K/Th ratio, EM ratios)
  └── Label: positive = within 500m of known VMS, negative = within 500m of barren hole
  
Phase 2 (Months 7–9): Model Training
  └── Random Forest + XGBoost with 5-fold spatial cross-validation
  └── Class imbalance handling: SMOTE + class-weighted loss
  └── Hyperparameter tuning via Bayesian optimisation (Optuna)
  └── SHAP TreeExplainer for feature importance & per-pixel explanation maps

Phase 3 (Months 9–10): Map Production & Validation
  └── Run model over full ~3,800 km² BMC extent
  └── Validate against 10 held-out known deposits
  └── Compare AI targets against historical drill density
```

---

## Expected Outputs

- 📄 **Primary Paper** — *"Machine learning prospectivity mapping for VMS deposits in the Bathurst Mining Camp using integrated NRCan geophysical and till geochemical data"*  
  Target: *Ore Geology Reviews* or *Journal of Geochemical Exploration*
- 🗺️ **Open Prospectivity Map** — GeoTIFF + PDF shared with NB Geological Survey Branch
- 💻 **Reproducible Python Pipeline** — GitHub repository: raw data download → final map

---

## Tech Stack (Dashboard)

- **Map:** [Leaflet.js](https://leafletjs.com/) + Leaflet.heat
- **Animations:** [GSAP](https://gsap.com/)
- **Fonts:** [Outfit](https://fonts.google.com/specimen/Outfit) via Google Fonts
- **Basemap:** [CartoDB Dark Matter](https://carto.com/basemaps/)
- **Python (Research):** scikit-learn, XGBoost, SHAP, GeoPandas, Rasterio, Optuna

---

## Project Timeline

| Phase | Months | Activity |
|---|---|---|
| 1 | 6–7 | Data download, harmonisation, feature engineering |
| 2 | 7–9 | Model training & SHAP explainability |
| 3 | 9–10 | Prospectivity map production & validation |
| Write-up | 10–12 | Manuscript preparation & submission |

*Estimated total duration: 6–8 months. Data cost: $0 — all datasets are open access.*

---

## License

CC(LICENSE)  
Research data references: open government data, subject to respective source licences.

---
## 🤖 Acknowledgements

This was built with the assistance of the **Antigravity Agent**, (Claude Sonnet 4.6) an advanced AI coding assistant developed by Google DeepMind.

---

*Prepared March 2026 · New Brunswick Geoscience AI Research · Solo Researcher Edition*
