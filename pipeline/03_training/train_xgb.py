"""
train_xgb.py
─────────────
Trains an XGBoost classifier for VMS deposit prospectivity.
Mirrors the structure of train_rf.py with XGBoost-specific hyperparameters.

XGBoost advantages over RF for this task:
  • Better calibrated probability outputs (useful for prospectivity mapping)
  • scale_pos_weight handles class imbalance without SMOTE
  • Early stopping prevents overfitting on small positive class
  • SHAP values are especially clean for tree boosters

Usage:
    python pipeline/03_training/train_xgb.py
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import xgboost as xgb

PIPELINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_DIR))
from config import (
    PROCESSED_DIR, MODELS_DIR, XGB_MODEL_PATH,
    RANDOM_STATE, N_SPATIAL_FOLDS, N_OPTUNA_TRIALS
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

DATASET_DIR  = PROCESSED_DIR / "training_dataset"
METRICS_PATH = MODELS_DIR / "xgb_cv_metrics.csv"


def load_dataset():
    data   = np.load(DATASET_DIR / "training_data.npz")
    X, y   = data["X"], data["y"]
    folds  = np.load(DATASET_DIR / "spatial_folds.npy")
    names  = pd.read_csv(DATASET_DIR / "feature_names.csv", header=None).squeeze().tolist()
    neg, pos = (y == 0).sum(), (y == 1).sum()
    log.info(f"Loaded: X={X.shape}  pos={pos}  neg={neg}  ratio={neg/pos:.1f}:1")
    return X, y, folds, names, neg / pos


class SpatialBlockCV:
    """Same spatial CV splitter as in train_rf.py."""
    def __init__(self, fold_ids, n_splits):
        self.fold_ids = fold_ids
        self.n_splits = n_splits

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_total    = len(X)
        n_original = len(self.fold_ids)
        for fold in range(self.n_splits):
            test_idx  = np.where(self.fold_ids == fold)[0]
            train_idx = np.where(self.fold_ids != fold)[0]
            if n_total > n_original:
                train_idx = np.concatenate([train_idx, np.arange(n_original, n_total)])
            yield train_idx, test_idx


def build_xgb_objective(X, y, cv, scale_pos_weight):
    def objective(trial):
        params = dict(
            n_estimators      = trial.suggest_int("n_estimators", 100, 1000),
            max_depth         = trial.suggest_int("max_depth", 3, 10),
            learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample         = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight  = trial.suggest_int("min_child_weight", 1, 20),
            gamma             = trial.suggest_float("gamma", 0.0, 5.0),
            reg_alpha         = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda        = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            scale_pos_weight  = scale_pos_weight,
            eval_metric       = "auc",
            use_label_encoder = False,
            random_state      = RANDOM_STATE,
            n_jobs            = -1,
        )
        aucs = []
        for train_idx, test_idx in cv.split(X, y):
            clf = xgb.XGBClassifier(**params)
            clf.fit(X[train_idx], y[train_idx], verbose=False)
            y_prob = clf.predict_proba(X[test_idx])[:, 1]
            aucs.append(roc_auc_score(y[test_idx], y_prob))
        return np.mean(aucs)
    return objective


def evaluate(clf, X, y, cv, feature_names):
    aucs, aps, baccs = [], [], []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        clf.fit(X[train_idx], y[train_idx], verbose=False)
        y_prob = clf.predict_proba(X[test_idx])[:, 1]
        y_pred = clf.predict(X[test_idx])
        auc    = roc_auc_score(y[test_idx], y_prob)
        ap     = average_precision_score(y[test_idx], y_prob)
        bacc   = balanced_accuracy_score(y[test_idx], y_pred)
        aucs.append(auc); aps.append(ap); baccs.append(bacc)
        log.info(f"  Fold {fold}: AUC={auc:.4f}  AP={ap:.4f}  BalAcc={bacc:.4f}")

    metrics = dict(
        roc_auc_mean=np.mean(aucs), roc_auc_std=np.std(aucs),
        avg_prec_mean=np.mean(aps), avg_prec_std=np.std(aps),
        balanced_acc_mean=np.mean(baccs), balanced_acc_std=np.std(baccs),
    )
    log.info(f"\n  ROC-AUC : {metrics['roc_auc_mean']:.4f} ± {metrics['roc_auc_std']:.4f}")
    log.info(f"  AvgPrec : {metrics['avg_prec_mean']:.4f} ± {metrics['avg_prec_std']:.4f}")
    log.info(f"  BalAcc  : {metrics['balanced_acc_mean']:.4f} ± {metrics['balanced_acc_std']:.4f}")

    clf.fit(X, y, verbose=False)
    importances = pd.Series(clf.feature_importances_, index=feature_names)
    log.info("\n  Top 10 Feature Importances (XGBoost gain):")
    for name, imp in importances.nlargest(10).items():
        bar = "█" * int(imp * 50)
        log.info(f"    {name:30s}: {imp:.4f} {bar}")

    return metrics, importances


def main():
    log.info("═══ XGBoost Training ═══")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X, y, folds, feature_names, ratio = load_dataset()
    cv = SpatialBlockCV(folds, N_SPATIAL_FOLDS)

    log.info(f"\n[Bayesian HPO — {N_OPTUNA_TRIALS} trials]")
    study = optuna.create_study(direction="maximize", study_name="xgb_vms")
    study.optimize(
        build_xgb_objective(X, y, cv, scale_pos_weight=ratio),
        n_trials=N_OPTUNA_TRIALS,
        show_progress_bar=True
    )
    log.info(f"  Best AUC: {study.best_value:.4f}")
    log.info(f"  Best params: {study.best_params}")

    best_clf = xgb.XGBClassifier(
        **study.best_params,
        scale_pos_weight=ratio,
        eval_metric="auc",
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    log.info("\n[Final Evaluation (Spatial CV)]")
    metrics, importances = evaluate(best_clf, X, y, cv, feature_names)

    metrics_df = pd.DataFrame([metrics])
    metrics_df["best_params"] = str(study.best_params)
    metrics_df.to_csv(METRICS_PATH, index=False)

    fi_path = MODELS_DIR / "xgb_feature_importances.csv"
    importances.sort_values(ascending=False).to_csv(fi_path, header=["importance"])

    best_clf.fit(X, y, verbose=False)
    joblib.dump(best_clf, XGB_MODEL_PATH)
    log.info(f"\n✅ XGBoost model saved → {XGB_MODEL_PATH}")
    log.info(f"✅ CV metrics saved   → {METRICS_PATH}")
    log.info("   Run: python pipeline/03_training/evaluate_models.py")


if __name__ == "__main__":
    main()
