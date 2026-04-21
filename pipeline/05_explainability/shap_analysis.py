"""
shap_analysis.py
─────────────────
SHAP TreeExplainer analysis for the best-performing VMS prospectivity model.

Produces:
  • SHAP summary plot (beeswarm) — overall feature importance + direction
  • SHAP bar chart — mean absolute SHAP values
  • SHAP dependence plots — top 4 features vs. deposit probability
  • SHAP per-point values saved to CSV for spatial mapping

Why SHAP for this project:
  In mineral exploration, understanding *why* the model predicts high
  prospectivity is as important as the prediction itself. SHAP values
  answer "which features pushed this pixel toward high VMS probability?"
  at every grid cell — producing geologically interpretable explanation maps.

Usage:
    python pipeline/05_explainability/shap_analysis.py
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

PIPELINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_DIR))
from config import (
    PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, RF_MODEL_PATH, XGB_MODEL_PATH
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

DATASET_DIR = PROCESSED_DIR / "training_dataset"
SHAP_DIR = OUTPUTS_DIR / "shap"
SHAP_DIR.mkdir(parents=True, exist_ok=True)

STYLE = {
    "figure.facecolor": "#0f172a",
    "axes.facecolor":   "#1e293b",
    "text.color":       "#e2e8f0",
    "axes.labelcolor":  "#cbd5e1",
    "xtick.color":      "#94a3b8",
    "ytick.color":      "#94a3b8",
}


def load_data():
    data  = np.load(DATASET_DIR / "training_data.npz")
    names = pd.read_csv(
        DATASET_DIR / "feature_names.csv", header=None
    ).squeeze().tolist()
    return data["X"], data["y"], names


def compute_shap_values(model, X: np.ndarray, feature_names: list):
    """Compute SHAP values using TreeExplainer (fast, exact for trees)."""
    log.info("  Initialising SHAP TreeExplainer ...")
    explainer = shap.TreeExplainer(model)

    log.info(f"  Computing SHAP values for {X.shape[0]} samples ...")
    shap_values = explainer.shap_values(X)

    # For binary classifiers, shap_values may be a list [class0, class1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]   # Use class 1 (VMS positive)

    log.info(f"  SHAP values shape: {shap_values.shape}")
    return shap_values, explainer.expected_value


def plot_summary(shap_values, X, feature_names, model_name):
    """Beeswarm summary plot."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(
            shap_values, X,
            feature_names=feature_names,
            plot_type="dot",
            show=False,
            color_bar=True,
            max_display=15
        )
        plt.title(
            f"SHAP Feature Importance — {model_name}\n"
            "VMS Prospectivity · Bathurst Mining Camp",
            color="#f8fafc", fontsize=12, fontweight="bold"
        )
        out = SHAP_DIR / f"shap_summary_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f172a")
        plt.close()
        log.info(f"  ✅ Saved: {out}")


def plot_bar(shap_values, feature_names, model_name):
    """Mean absolute SHAP bar chart."""
    mean_shap = np.abs(shap_values).mean(axis=0)
    imp = pd.Series(mean_shap, index=feature_names).sort_values()

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(
            imp.index[-15:], imp.values[-15:],
            color="#3b82f6", alpha=0.85, edgecolor="#1e40af"
        )
        ax.set_xlabel("Mean |SHAP value|  (impact on VMS probability)")
        ax.set_title(
            f"SHAP Feature Ranking — {model_name}\nBathurst Mining Camp VMS Prospectivity",
            color="#f8fafc"
        )
        ax.grid(True, axis="x", alpha=0.3)
        out = SHAP_DIR / f"shap_bar_{model_name.lower().replace(' ', '_')}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f172a")
        plt.close()
        log.info(f"  ✅ Saved: {out}")


def plot_dependence(shap_values, X, feature_names, model_name, top_n=4):
    """Dependence plots for top-N features."""
    mean_shap = np.abs(shap_values).mean(axis=0)
    top_features = np.argsort(mean_shap)[::-1][:top_n]

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        for i, feat_idx in enumerate(top_features):
            feat_name = feature_names[feat_idx]
            shap.dependence_plot(
                feat_idx, shap_values, X,
                feature_names=feature_names,
                ax=axes[i],
                show=False,
                dot_size=20,
                alpha=0.7
            )
            axes[i].set_title(
                f"SHAP dependence: {feat_name}", color="#94a3b8", fontsize=10
            )
        fig.suptitle(
            f"SHAP Dependence Plots — {model_name}\nBathurst Mining Camp",
            color="#f8fafc", fontsize=13, fontweight="bold"
        )
        fig.tight_layout()
        out = SHAP_DIR / f"shap_dependence_{model_name.lower().replace(' ', '_')}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f172a")
        plt.close()
        log.info(f"  ✅ Saved: {out}")


def save_shap_csv(shap_values, feature_names, model_name):
    """Save raw SHAP values to CSV for spatial mapping in GIS."""
    df = pd.DataFrame(shap_values, columns=[f"shap_{c}" for c in feature_names])
    df["shap_sum"] = shap_values.sum(axis=1)
    out = SHAP_DIR / f"shap_values_{model_name.lower().replace(' ', '_')}.csv"
    df.to_csv(out, index=False)
    log.info(f"  ✅ Saved SHAP CSV: {out}")


def run_for_model(model, X, y, feature_names, model_name):
    log.info(f"\n─── {model_name} ───")
    log.info("  Fitting final model on full dataset ...")
    model.fit(X, y)
    shap_vals, base_val = compute_shap_values(model, X, feature_names)
    log.info(f"  Expected value (base rate): {base_val:.4f}")

    plot_summary(shap_vals, X, feature_names, model_name)
    plot_bar(shap_vals, feature_names, model_name)
    plot_dependence(shap_vals, X, feature_names, model_name)
    save_shap_csv(shap_vals, feature_names, model_name)


def main():
    log.info("═══ SHAP Explainability Analysis ═══")

    X, y, feature_names = load_data()

    # Run SHAP for both models
    rf_model  = joblib.load(RF_MODEL_PATH)
    xgb_model = joblib.load(XGB_MODEL_PATH)

    run_for_model(rf_model,  X, y, feature_names, "Random Forest")
    run_for_model(xgb_model, X, y, feature_names, "XGBoost")

    log.info(f"\n✅ All SHAP outputs saved to: {SHAP_DIR}")
    log.info(
        "   Geological interpretation:\n"
        "   • High |SHAP| for mag_hgm → structural control on mineralisation\n"
        "   • High |SHAP| for rad_k_th → alteration halo detected\n"
        "   • High |SHAP| for zn_ppm / as_ppm → pathfinder anomaly\n"
        "   These should align with known geology — if not, investigate data quality."
    )
    log.info("   Run next: python pipeline/04_prospectivity_map/predict_full_extent.py")


if __name__ == "__main__":
    main()
