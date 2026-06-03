# 🪨 VMS Deposit Discovery — ML Prospectivity Research Design
### Bathurst Mining Camp, New Brunswick, Canada

> **Status: Research Design & Prototype Visualisation**
> This repository contains the full research design, proposed methodology, and interactive proof-of-concept dashboard for a machine learning prospectivity mapping study of the Bathurst Mining Camp. The Python modelling pipeline is under active development — see the [Roadmap](#roadmap) for current progress.

---

## Overview

### Research Gap

>The Bathurst Mining Camp (BMC) of northern New Brunswick, Canada, is one of the world's most intensively studied volcanogenic massive sulphide (VMS) districts, hosting more than 45 known Zn-Pb-Cu-Ag deposits and representing a cornerstone of Canada's historic base metal production (Goodfellow and McCutcheon, 2003). Despite this legacy, recent exploration has yielded limited new discoveries, and an estimated 70% of the camp's prospective ground remains untested (McCutcheon and Walker, 2020). This disconnect between data density and exploration success motivates a reassessment of how available geoscientific datasets are being leveraged. Machine learning (ML)-based mineral prospectivity mapping (MPM) has emerged as a powerful framework for integrating multi-source datasets to delineate high-priority exploration targets (e.g., Rodriguez-Galiano et al., 2015; Carranza and Laborte, 2015b; Zuo et al., 2023). Within the BMC, Parsa et al. (2023) demonstrated the viability of ML-based MPM for VHMS deposits along the northeastern Brunswick belt, employing an ensemble regularization approach to address class imbalance and overfitting inherent to deposit-scale training data. However, that study, like others in the region, was limited to a subregional extent and did not incorporate a camp-scale integration of NRCan airborne geophysical data — spanning magnetics, radiometrics, and full-tensor gravity gradiometry — alongside NB till geochemistry, which records glacially dispersed pathfinder element signatures that geophysics alone cannot capture. To date, no peer-reviewed, open-science ML prospectivity study has operated at camp scale with this combination of inputs. This study addresses that gap by presenting the first camp-scale, open-science ML prospectivity model for the BMC, jointly leveraging NRCan airborne geophysical data and NB till geochemistry, with the aim of producing reproducible, publicly available targeting outputs for the broader exploration community.

### Reference:

1. Goodfellow, W.D., and McCutcheon, S.R., 2003. Massive Sulphide Deposits of the Bathurst Mining Camp, New Brunswick, Canada. Economic Geology Monograph 11.
2. McCutcheon, S.R., and Walker, J.A., 2020. Great Mining Camps of Canada 8. The Bathurst Mining Camp, New Brunswick, Part 2. Geoscience Canada, 47, 143–166.
3. Rodriguez-Galiano, V., Sanchez-Castillo, M., Chica-Olmo, M., and Chica-Rivas, M., 2015. Machine learning predictive models for mineral prospectivity: an evaluation of neural networks, random forest, regression trees and support vector machines. Ore Geology Reviews, 71, 804–818.
4. Carranza, E.J.M., and Laborte, A.G., 2015b. Random forest predictive modeling of mineral prospectivity with small number of prospects and data with missing values in Abra (Philippines). Computers & Geosciences, 74, 60–70.
5. Zuo, R., Xiong, Y., Wang, Z., Wang, J., and Kreuzer, O.P., 2023. A new generation of artificial intelligence algorithms for mineral prospectivity mapping. Natural Resources Research, 32(5), 1859–1869.
6. Parsa, M., Lentz, D.R., and Walker, J.A., 2023. Predictive modeling of prospectivity for VHMS mineral deposits, northeastern Bathurst Mining Camp, NB, Canada, using an ensemble regularization technique. Natural Resources Research, 32, 19–36. ✨

---

## What's in This Repository

| File / Folder | Description |
|---|---|
| `index.html` + `main.js` + `style.css` | Interactive proof-of-concept dashboard visualising the proposed model architecture, training label locations (45 known VMS deposits + 250+ barren drill holes), and a simulated prospectivity heatmap |
| `README.md` | Full research design, methodology, data sources, and implementation roadmap |
| `pipeline/` | Full Python ML pipeline: data download, preprocessing, feature engineering, model training (RF + XGBoost), spatial cross-validation, SHAP explainability, and prospectivity map export |

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
- [x] Python pipeline scaffold (`pipeline/` directory — all scripts written)
- [ ] Data download: run `pipeline/01_data_download/` scripts to fetch real data
- [ ] Preprocessing: reproject rasters, extract features, engineer derived bands
- [ ] Model training: Random Forest + XGBoost with 5-fold spatial cross-validation
- [ ] SHAP explainability maps
- [ ] Full prospectivity map production (GeoTIFF)
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
