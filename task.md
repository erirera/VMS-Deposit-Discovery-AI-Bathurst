# VMS Python ML Pipeline — Task Checklist

## Scaffold & Environment
- [x] Create `pipeline/` directory structure
- [x] `requirements.txt`
- [x] `config.py` (paths, CRS constants)

## Phase 1 — Data Download
- [x] `01_data_download/download_nb_geochemistry.py`
- [x] `01_data_download/download_vms_labels.py`
- [x] `01_data_download/download_nrcan_mag.py`

## Phase 2 — Preprocessing
- [x] `02_preprocessing/reproject_grids.py`
- [x] `02_preprocessing/extract_features.py`
- [x] `02_preprocessing/engineer_features.py`

## Phase 3 — Training
- [x] `03_training/build_dataset.py`
- [x] `03_training/train_rf.py`
- [x] `03_training/train_xgb.py`
- [x] `03_training/evaluate_models.py`

## Phase 4 — Prospectivity Map
- [x] `04_prospectivity_map/predict_full_extent.py`
- [x] `04_prospectivity_map/export_map.py`

## Phase 5 — Explainability
- [x] `05_explainability/shap_analysis.py`

## Documentation
- [x] README roadmap updated (pipeline scaffold marked complete)
- [x] `config.py` documents all paths and CRS constants

## Next Actions (User Runs)
- [ ] `pip install -r pipeline/00_environment/requirements.txt`
- [ ] `python pipeline/01_data_download/download_vms_labels.py` (works immediately — no internet needed)
- [ ] `python pipeline/01_data_download/download_nb_geochemistry.py` (downloads real NB data)
- [ ] `python pipeline/01_data_download/download_nrcan_mag.py` (attempts download; may need manual step)
- [ ] Continue through preprocessing → training → map pipeline
