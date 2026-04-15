# 🪨 VMS Deposit Discovery — ML Prospectivity Research Design
### Bathurst Mining Camp, New Brunswick, Canada

> **Status: Research Design & Prototype Visualisation**
> This repository contains the full research design, proposed methodology, and interactive proof-of-concept dashboard for a machine learning prospectivity mapping study of the Bathurst Mining Camp. The Python modelling pipeline is under active development — see the [Roadmap](#roadmap) for current progress.

---

## Overview

The **Bathurst Mining Camp (BMC)** in northern New Brunswick is one of Canada's most important base-metal districts, hosting over 45 known Volcanogenic Massive Sulphide (VMS) deposits including the world-class Brunswick No. 12 mine. Despite intensive historical exploration, significant portions of the camp remain underexplored at depth and along strike.

This project proposes and designs a rigorous, open-science machine learning workflow to predict undiscovered VMS mineralisation across the ~3,800 km² camp by integrating freely available geophysical and geochemical datasets from NRCan and the NB Geological Survey.

**The research gap this addresses:** No peer-reviewed, open-science ML prospectivity study exists for the Bathurst Mining Camp using the full NRCan airborne geophysical compilation combined with NB till geochemistry. This project aims to fill that gap.

---

## What's in This Repository

| File / Folder | Description |
|---|---|
| `index.html` + `main.js` + `style.css` | Interactive proof-of-concept dashboard visualising the proposed model architecture, training label locations (45 known VMS deposits + 250+ barren drill holes), and a simulated prospectivity heatmap |
| `README.md` | Full research design, methodology, data sources, and implementation roadmap |

The **dashboard is a research communication tool**, not a trained model output. It visualises the proposed spatial framework — deposit locations, camp boundary, and simulated heatmap — to communicate the study design and solicit feedback. Live at: [erirera.github.io/VMS-Deposit-Discovery-AI-Bathurst](https://erirera.github.io/VMS-Deposit-Discovery-AI-Bathurst/)

---

## Scientific Background

VMS deposits form at or near the seafloor through hydrothermal circulation driven by volcanic activity. In the BMC, they are spatially associated with specific geophysical and geochemical signatures:

- **Airborne magnetics** (TMI, first vertical derivative): structural controls, magnetic lows over sulphide-rich zones
- **Airborne radiometrics** (K%, Th, U): alteration halos, potassic and sericitic zones around VMS systems
- **Airborne EM** (apparent conductivity): direct detection of conductive sulphide bodies at depth
- **Gravity** (Bouguer anomaly): density contrasts associated with massive sulphide lenses
- **Till geochemistry** (Zn, Pb, Cu, Ag, Au, As): glacially dispersed pathfinder elements downice of mineralisation

The proposed model integrates these layers as features for a supervised classification problem: predict VMS mineralisation probability at any point within the camp.

---

## Proposed Methodology

```
Phase 1 — Data Preparation (Months 1–2)
├── Download NRCan airborne geophysical grids (magnetics, radiometrics, EM, gravity)
├── Download NB GSB till geochemistry point data
├── Reproject all data → NAD83 / NB Double Stereographic (EPSG:2953) at 100m resolution
├── Extract geophysical grid values at till sample locations (~20 features)
└── Engineer derived features: TMI gradient, analytic signal, K/Th ratio, EM decay ratios

Phase 2 — Label Construction
├── Positive labels: 45 known VMS deposits (NB Metallic Minerals Database), buffered 500m
├── Negative labels: 250+ barren drill intercepts (NB GSB / SEDAR), buffered 500m
└── Class imbalance strategy: SMOTE oversampling + class-weighted loss functions

Phase 3 — Model Training & Evaluation (Months 2–4)
├── Algorithms: Random Forest + XGBoost (ensemble comparison)
├── Validation: 5-fold spatial cross-validation (spatially blocked to prevent data leakage)
├── Hyperparameter tuning: Bayesian optimisation via Optuna
└── Explainability: SHAP TreeExplainer for feature importance + per-pixel explanation maps

Phase 4 — Prospectivity Map Production (Months 4–5)
├── Run trained model across full 3,800 km² BMC extent
├── Validate against 10 held-out known deposits (not used in training)
├── Compare AI-generated targets against historical drill density
└── Export: GeoTIFF + PDF prospectivity map shared with NB Geological Survey Branch

Phase 5 — Publication (Months 5–8)
└── Target journal: Ore Geology Reviews or Journal of Geochemical Exploration
```

---

## Data Sources

All data is freely available — total data cost: **$0**.

| Dataset | Source | Status |
|---|---|---|
| Till Geochemistry (Zn, Pb, Cu, Ag, Au, As) | [NB Geological Survey Open Data](https://www2.gnb.ca/content/gnb/en/departments/erd/energy/content/minerals/content/geology_data.html) | Identified |
| Airborne Magnetics (TMI, FVD) | [NRCan Geoscience Repository](https://geoscan.nrcan.gc.ca/) | Identified |
| Airborne Radiometrics (K%, Th, U) | NRCan Airborne Geophysical Surveys | Identified |
| Airborne EM (Conductivity) | NRCan Airborne Geophysical Surveys | Identified |
| Gravity (Bouguer Anomaly) | NRCan Gravity Programme | Identified |
| VMS Deposit Locations | [NB Metallic Minerals Database](https://www2.gnb.ca/content/gnb/en/departments/erd/energy/content/minerals.html) | Identified |
| Barren Drill Records | NB GSB / SEDAR | Identified |

---

## Python Stack (Implementation)

```python
# Core pipeline (in development)
geopandas        # Spatial data handling
rasterio         # Raster grid I/O and resampling
scikit-learn     # Random Forest, cross-validation, SMOTE
xgboost          # Gradient boosting classifier
shap             # Model explainability
optuna           # Bayesian hyperparameter optimisation
matplotlib       # Visualisation
```

---

## Roadmap

- [x] Research design completed
- [x] Interactive dashboard (proof-of-concept visualisation)
- [x] Data sources identified and access confirmed
- [ ] Data download and preprocessing pipeline (in progress)
- [ ] Feature engineering notebook
- [ ] Model training and spatial cross-validation
- [ ] SHAP explainability maps
- [ ] Full prospectivity map production
- [ ] Manuscript preparation

---

## Why This Matters

Geological surveys, KoBold Metals, Kennecott, and other data-driven exploration companies are demonstrating that integrating ML into prospectivity mapping can dramatically reduce the number of dry holes drilled. The BMC offers an ideal test case: abundant open data, well-characterised training labels (45 known deposits), and a district with strong evidence of remaining undiscovered resources.

This project is designed to be fully reproducible, open-science, and publishable — contributing to the growing body of applied ML literature in mineral exploration.

---

## Author

**Dele Falebita, PhD** — GIT APEGNB & Data Scientist  
[github.com/erirera](https://github.com/erirera) | Moncton, New Brunswick, Canada  
20+ years experience in geo-resource exploration, geostatistics, and geophysical data analysis.

---

*Research design completed March 2026. Python implementation in progress.*  
*License: CC0-1.0 — all referenced datasets are open government data subject to respective source licences.*
