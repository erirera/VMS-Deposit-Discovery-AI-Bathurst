"""
evaluate_models.py
───────────────────
Comparative evaluation of the trained RF and XGBoost models.

Produces:
  • Side-by-side CV metric table (ROC-AUC, Average Precision, Balanced Accuracy)
  • ROC curve comparison plot (outputs/roc_curves.png)
  • Precision-Recall curve plot (outputs/pr_curves.png)
  • Feature importance comparison (outputs/feature_importance_comparison.png)

Usage:
    python pipeline/03_training/evaluate_models.py
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import xgboost as xgb

PIPELINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_DIR))
from config import (
    PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR,
    RF_MODEL_PATH, XGB_MODEL_PATH, RANDOM_STATE, N_SPATIAL_FOLDS
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

DATASET_DIR = PROCESSED_DIR / "training_dataset"

# ── Plotting Aesthetics ───────────────────────────────────────────────────────
STYLE = {
    "figure.facecolor":  "#0f172a",
    "axes.facecolor":    "#1e293b",
    "axes.edgecolor":    "#334155",
    "axes.labelcolor":   "#cbd5e1",
    "xtick.color":       "#94a3b8",
    "ytick.color":       "#94a3b8",
    "text.color":        "#e2e8f0",
    "grid.color":        "#334155",
    "grid.alpha":        0.6,
    "font.family":       "DejaVu Sans",
}


class SpatialBlockCV:
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


def load_data():
    data   = np.load(DATASET_DIR / "training_data.npz")
    folds  = np.load(DATASET_DIR / "spatial_folds.npy")
    names  = pd.read_csv(DATASET_DIR / "feature_names.csv", header=None).squeeze().tolist()
    return data["X"], data["y"], folds, names


def compute_roc_pr(clf, X, y, cv):
    """Collect per-fold ROC and PR curves."""
    all_probs, all_true = [], []
    for train_idx, test_idx in cv.split(X, y):
        clf.fit(X[train_idx], y[train_idx])
        probs = clf.predict_proba(X[test_idx])[:, 1]
        all_probs.extend(probs)
        all_true.extend(y[test_idx])
    return np.array(all_true), np.array(all_probs)


def plot_roc_pr(rf_true, rf_probs, xgb_true, xgb_probs):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(
            "VMS Prospectivity — Model Comparison\nBathurst Mining Camp, NB",
            color="#f8fafc", fontsize=14, fontweight="bold"
        )

        # ── ROC Curve ────────────────────────────────────────────────────────
        ax = axes[0]
        for true, probs, label, color in [
            (rf_true,  rf_probs,  "Random Forest", "#f59e0b"),
            (xgb_true, xgb_probs, "XGBoost",       "#3b82f6"),
        ]:
            fpr, tpr, _ = roc_curve(true, probs)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2.5,
                    label=f"{label} (AUC = {roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.4)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves (Spatial CV)", color="#94a3b8")
        ax.legend(loc="lower right", framealpha=0.2)
        ax.grid(True, alpha=0.3)

        # ── Precision-Recall Curve ────────────────────────────────────────────
        ax = axes[1]
        for true, probs, label, color in [
            (rf_true,  rf_probs,  "Random Forest", "#f59e0b"),
            (xgb_true, xgb_probs, "XGBoost",       "#3b82f6"),
        ]:
            prec, rec, _ = precision_recall_curve(true, probs)
            pr_auc = auc(rec, prec)
            ax.plot(rec, prec, color=color, lw=2.5,
                    label=f"{label} (AP = {pr_auc:.3f})")

        baseline = (rf_true == 1).mean()
        ax.axhline(baseline, color="white", lw=1, ls="--", alpha=0.4,
                   label=f"Baseline (prevalence={baseline:.2f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves (Spatial CV)", color="#94a3b8")
        ax.legend(loc="upper right", framealpha=0.2)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        out_path = OUTPUTS_DIR / "roc_pr_curves.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        log.info(f"  ✅ Saved: {out_path}")
        plt.close(fig)


def plot_feature_importance(names, rf_model, xgb_model):
    rf_imp  = pd.Series(rf_model.feature_importances_,  index=names).nlargest(15)
    xgb_imp = pd.Series(xgb_model.feature_importances_, index=names).nlargest(15)
    all_feats = rf_imp.index.union(xgb_imp.index)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(
            "Feature Importances (Top 15 per model)",
            color="#f8fafc", fontsize=14, fontweight="bold"
        )

        for ax, imp, label, color in [
            (axes[0], rf_imp,  "Random Forest", "#f59e0b"),
            (axes[1], xgb_imp, "XGBoost",       "#3b82f6"),
        ]:
            bars = ax.barh(imp.index[::-1], imp.values[::-1], color=color, alpha=0.85)
            ax.set_title(label, color="#94a3b8")
            ax.set_xlabel("Importance")
            ax.grid(True, axis="x", alpha=0.3)

        fig.tight_layout()
        out_path = OUTPUTS_DIR / "feature_importances.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        log.info(f"  ✅ Saved: {out_path}")
        plt.close(fig)


def print_metric_table(rf_metrics: pd.DataFrame, xgb_metrics: pd.DataFrame):
    log.info("\n══════════════════════════════════════════════════")
    log.info("  Model Comparison — Spatial Cross-Validation")
    log.info("══════════════════════════════════════════════════")
    log.info(f"  {'Metric':<25} {'Random Forest':>16} {'XGBoost':>12}")
    log.info(f"  {'─'*25} {'─'*16} {'─'*12}")
    for col in ["roc_auc_mean", "avg_prec_mean", "balanced_acc_mean"]:
        rf_val  = rf_metrics[col].values[0]
        xgb_val = xgb_metrics[col].values[0]
        winner  = "◀ RF" if rf_val > xgb_val else "◀ XGB"
        log.info(f"  {col:<25} {rf_val:>16.4f} {xgb_val:>12.4f}  {winner}")
    log.info("══════════════════════════════════════════════════")


def main():
    log.info("═══ Model Evaluation & Comparison ═══")

    X, y, folds, feature_names = load_data()
    cv = SpatialBlockCV(folds, N_SPATIAL_FOLDS)

    # Load trained models
    rf_model  = joblib.load(RF_MODEL_PATH)
    xgb_model = joblib.load(XGB_MODEL_PATH)
    log.info("  Loaded RF and XGBoost models")

    # Load saved metrics CSVs
    rf_metrics  = pd.read_csv(MODELS_DIR / "rf_cv_metrics.csv")
    xgb_metrics = pd.read_csv(MODELS_DIR / "xgb_cv_metrics.csv")
    print_metric_table(rf_metrics, xgb_metrics)

    # Collect predictions for plotting
    log.info("\n[Collecting predictions for ROC/PR curves ...]")
    rf_true, rf_probs   = compute_roc_pr(rf_model,  X, y, cv)
    xgb_true, xgb_probs = compute_roc_pr(xgb_model, X, y, cv)

    log.info("\n[Generating plots ...]")
    plot_roc_pr(rf_true, rf_probs, xgb_true, xgb_probs)
    plot_feature_importance(feature_names, rf_model, xgb_model)

    log.info("\n✅ Evaluation complete. Check outputs/ for figures.")
    log.info("   Run next: python pipeline/04_prospectivity_map/predict_full_extent.py")
    log.info("   Run next: python pipeline/05_explainability/shap_analysis.py")


if __name__ == "__main__":
    main()
