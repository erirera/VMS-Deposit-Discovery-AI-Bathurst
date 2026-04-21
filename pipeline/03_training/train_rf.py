"""
train_rf.py
────────────
Trains a Random Forest classifier for VMS deposit prospectivity.

Key design decisions:
  • Spatial cross-validation: uses pre-computed spatial fold indices to
    prevent train/test leakage via spatial autocorrelation.
  • class_weight='balanced': handles any residual imbalance after SMOTE.
  • Bayesian hyperparameter tuning via Optuna (50 trials default).
  • Saves best model + CV metrics to disk.

Metrics:
  Primary: ROC-AUC (spatial CV)
  Secondary: Average Precision, Balanced Accuracy, Precision@K

Usage:
    python pipeline/03_training/train_rf.py
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    balanced_accuracy_score, classification_report
)
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

PIPELINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_DIR))
from config import (
    PROCESSED_DIR, MODELS_DIR, RF_MODEL_PATH,
    RANDOM_STATE, N_SPATIAL_FOLDS, N_OPTUNA_TRIALS,
    CLASS_WEIGHT
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

DATASET_DIR   = PROCESSED_DIR / "training_dataset"
METRICS_PATH  = MODELS_DIR / "rf_cv_metrics.csv"


def load_dataset():
    """Load training arrays and spatial fold indices."""
    data = np.load(DATASET_DIR / "training_data.npz")
    X, y = data["X"], data["y"]
    spatial_folds = np.load(DATASET_DIR / "spatial_folds.npy")

    feature_names = pd.read_csv(
        DATASET_DIR / "feature_names.csv", header=None
    ).squeeze().tolist()

    log.info(f"Dataset loaded: X={X.shape}, y distribution={dict(zip(*np.unique(y, return_counts=True)))}")
    log.info(f"Features: {feature_names}")
    return X, y, spatial_folds, feature_names


class SpatialBlockCV:
    """
    Custom cross-validator that splits data by pre-assigned spatial fold IDs.
    Ensures test points are spatially separated from training points.
    Only applies to the original (pre-SMOTE) points; SMOTE points are always
    included in the training folds.
    """
    def __init__(self, fold_ids: np.ndarray, n_splits: int):
        self.fold_ids = fold_ids
        self.n_splits = n_splits

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_total = len(X)
        n_original = len(self.fold_ids)

        for fold in range(self.n_splits):
            test_idx  = np.where(self.fold_ids == fold)[0]
            train_idx = np.where(self.fold_ids != fold)[0]

            # If SMOTE added extra rows, include them in training
            if n_total > n_original:
                smote_idx = np.arange(n_original, n_total)
                train_idx = np.concatenate([train_idx, smote_idx])

            yield train_idx, test_idx


def build_objective(X: np.ndarray, y: np.ndarray, cv) -> callable:
    """Return an Optuna objective function for Random Forest."""
    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 800),
            "max_depth":        trial.suggest_int("max_depth", 3, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features":     trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
            "class_weight":     CLASS_WEIGHT,
            "n_jobs":           -1,
            "random_state":     RANDOM_STATE,
        }
        clf = RandomForestClassifier(**params)
        scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
        return scores.mean()
    return objective


def evaluate_final_model(
    clf: RandomForestClassifier,
    X: np.ndarray,
    y: np.ndarray,
    cv,
    feature_names: list
) -> dict:
    """Run full evaluation suite on the best model."""
    log.info("\n[Final Model Evaluation]")

    aucs, aps, baccs = [], [], []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        clf.fit(X[train_idx], y[train_idx])
        y_prob = clf.predict_proba(X[test_idx])[:, 1]
        y_pred = clf.predict(X[test_idx])
        y_true = y[test_idx]

        auc  = roc_auc_score(y_true, y_prob)
        ap   = average_precision_score(y_true, y_prob)
        bacc = balanced_accuracy_score(y_true, y_pred)

        aucs.append(auc)
        aps.append(ap)
        baccs.append(bacc)
        log.info(
            f"  Fold {fold}: ROC-AUC={auc:.4f}  AvgPrecision={ap:.4f}  "
            f"BalancedAcc={bacc:.4f}"
        )

    metrics = {
        "roc_auc_mean":  np.mean(aucs),
        "roc_auc_std":   np.std(aucs),
        "avg_prec_mean": np.mean(aps),
        "avg_prec_std":  np.std(aps),
        "balanced_acc_mean": np.mean(baccs),
        "balanced_acc_std":  np.std(baccs),
    }

    log.info(f"\n  ROC-AUC  : {metrics['roc_auc_mean']:.4f} ± {metrics['roc_auc_std']:.4f}")
    log.info(f"  AvgPrec  : {metrics['avg_prec_mean']:.4f} ± {metrics['avg_prec_std']:.4f}")
    log.info(f"  BalAcc   : {metrics['balanced_acc_mean']:.4f} ± {metrics['balanced_acc_std']:.4f}")

    # Feature importances
    clf.fit(X, y)  # Refit on all data for importances
    importances = pd.Series(clf.feature_importances_, index=feature_names)
    log.info("\n  Top 10 Feature Importances (Gini):")
    for name, imp in importances.nlargest(10).items():
        bar = "█" * int(imp * 50)
        log.info(f"    {name:30s}: {imp:.4f} {bar}")

    return metrics, importances


def main():
    log.info("═══ Random Forest Training ═══")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    X, y, spatial_folds, feature_names = load_dataset()

    # ── Spatial CV ───────────────────────────────────────────────────────────
    cv = SpatialBlockCV(spatial_folds, n_splits=N_SPATIAL_FOLDS)
    log.info(f"Cross-validation: Spatial Block ({N_SPATIAL_FOLDS} folds)")

    # ── Bayesian HPO ─────────────────────────────────────────────────────────
    log.info(f"\n[Bayesian Hyperparameter Optimisation — {N_OPTUNA_TRIALS} trials]")
    study = optuna.create_study(direction="maximize", study_name="rf_vms")
    study.optimize(
        build_objective(X, y, cv),
        n_trials=N_OPTUNA_TRIALS,
        show_progress_bar=True
    )

    log.info(f"  Best ROC-AUC: {study.best_value:.4f}")
    log.info(f"  Best params : {study.best_params}")

    # ── Final model ───────────────────────────────────────────────────────────
    best_params = study.best_params | {
        "class_weight": CLASS_WEIGHT,
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    }
    best_clf = RandomForestClassifier(**best_params)

    metrics, feature_importances = evaluate_final_model(
        best_clf, X, y, cv, feature_names
    )

    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df["best_params"] = str(study.best_params)
    metrics_df.to_csv(METRICS_PATH, index=False)

    # Save feature importances
    fi_path = MODELS_DIR / "rf_feature_importances.csv"
    feature_importances.sort_values(ascending=False).to_csv(fi_path, header=["importance"])

    # Save final model (refit on all data)
    best_clf.fit(X, y)
    joblib.dump(best_clf, RF_MODEL_PATH)
    log.info(f"\n✅ Best RF model saved → {RF_MODEL_PATH}")
    log.info(f"✅ CV metrics saved   → {METRICS_PATH}")
    log.info("   Run: python pipeline/03_training/train_xgb.py")
    log.info("   Run: python pipeline/03_training/evaluate_models.py")
    log.info("   Run: python pipeline/04_prospectivity_map/predict_full_extent.py")


if __name__ == "__main__":
    main()
