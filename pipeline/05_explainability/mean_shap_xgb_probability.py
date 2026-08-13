"""
mean_shap_xgb_probability.py
─────────────────────────────
Compute Mean SHAP values for XGBoost in probability space.

This script:
  1. Loads the trained XGBoost model and training data
  2. Computes SHAP values in probability space (after sigmoid transformation)
  3. Calculates Mean |SHAP| for each feature across all samples
  4. Generates visualizations and exports results to CSV

Why probability space?
  • Probability space is more interpretable to stakeholders
  • Directly represents the model's predicted VMS prospectivity (0-1)
  • Each SHAP value represents contribution to deposit probability
  • Mean SHAP aggregates global feature importance in probability units

Usage:
    python pipeline/05_explainability/mean_shap_xgb_probability.py
"""

import sys
import logging
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

PIPELINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_DIR))
from config import (
    PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, XGB_MODEL_PATH
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
    """Load training data and feature names."""
    data = np.load(DATASET_DIR / "training_data.npz")
    names = pd.read_csv(
        DATASET_DIR / "feature_names.csv", header=None
    ).squeeze().tolist()
    return data["X"], data["y"], names


def compute_mean_shap_xgb_probability(model, X: np.ndarray, feature_names: list):
    """
    Compute SHAP values for XGBoost in probability space.
    
    Approach:
      1. Create background dataset from a sample of training data
      2. Use TreeExplainer with model_output="probability" for direct probability-space SHAP values
      3. Use interventional feature perturbation for robust explanations
      4. Calculate mean absolute SHAP for each feature
    
    This ensures SHAP contributions are directly in probability space:
    - SHAP values represent changes in predicted probability (0 to 1)
    - Base value is the expected probability from background data
    - All values are naturally bounded in probability space
    
    Args:
        model: Fitted XGBoost classifier
        X: Feature matrix (n_samples, n_features)
        feature_names: List of feature names
    
    Returns:
        mean_shap: Mean |SHAP| per feature in probability space (n_features,)
        shap_values_prob: SHAP values in probability space (n_samples, n_features)
        base_prob: Base probability (expected value in probability space)
    """
    # Create background dataset from training data (subsample for efficiency)
    log.info("  Creating background dataset ...")
    background = shap.sample(
        X,
        min(100, X.shape[0])
    )
)
    log.info(f"  Background dataset shape: {background.shape}")
    
    log.info("  Initialising SHAP TreeExplainer ...")
    explainer = shap.TreeExplainer(
        model,
        model_output="probability",
        data=background,
        feature_perturbation="interventional"
    )

    log.info(f"  Computing SHAP values for {X.shape[0]} samples ...")
    shap_values_prob = explainer.shap_values(X)

    # Handle different SHAP value formats
    if isinstance(shap_values_prob, list):
        shap_values_prob = shap_values_prob[1]  # Use class 1 (VMS positive)
    elif isinstance(shap_values_prob, np.ndarray) and shap_values_prob.ndim == 3:
        shap_values_prob = shap_values_prob[:, :, 1]

    log.info(f"  SHAP values shape: {shap_values_prob.shape}")

    # Get base value (expected value in probability space)
    base_prob = explainer.expected_value
    if isinstance(base_prob, (list, np.ndarray)):
        base_prob = base_prob[1] if len(base_prob) > 1 else base_prob[0]
    
    base_prob = float(base_prob)
    log.info(f"  Base probability (expected value): {base_prob:.6f}")

    # Get predictions in probability space
    y_pred_prob = model.predict_proba(X)[:, 1]
    
    log.info(f"  Probability predictions range: [{y_pred_prob.min():.4f}, {y_pred_prob.max():.4f}]")
    log.info(f"  SHAP values range: [{shap_values_prob.min():.4f}, {shap_values_prob.max():.4f}]")

    # Compute mean absolute SHAP in probability space
    mean_shap_abs = np.abs(shap_values_prob).mean(axis=0)

    return {
        'mean_shap': mean_shap_abs,
        'mean_shap_df': pd.DataFrame({
            'feature': feature_names,
            'mean_shap_abs': mean_shap_abs
        }).sort_values('mean_shap_abs', ascending=False),
        'shap_values': shap_values_prob,
        'base_prob': base_prob,
        'y_pred_prob': y_pred_prob,
        'explainer': explainer
    }


def save_mean_shap_table(mean_shap_df, model_name="XGBoost"):
    """Save Mean SHAP table to CSV."""
    out = SHAP_DIR / f"mean_shap_{model_name.lower()}_probability.csv"
    mean_shap_df.to_csv(out, index=False)
    log.info(f"  ✅ Saved Mean SHAP table: {out}")
    return out


def plot_mean_shap_bar(mean_shap_df, model_name="XGBoost", top_n=15):
    """Create bar chart of Mean SHAP values."""
    top_features = mean_shap_df.head(top_n)
    
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(
            top_features['feature'],
            top_features['mean_shap_abs'],
            color="#10b981", alpha=0.85, edgecolor="#059669"
        )
        ax.set_xlabel("Mean |SHAP| (Impact on VMS Probability)", fontsize=11)
        ax.set_title(
            f"Mean SHAP Feature Importance — {model_name} (Probability Space)\n"
            "Bathurst Mining Camp VMS Prospectivity",
            color="#f8fafc", fontsize=12, fontweight="bold"
        )
        ax.grid(True, axis="x", alpha=0.3)
        ax.invert_yaxis()
        
        fig.tight_layout()
        out = SHAP_DIR / f"mean_shap_bar_{model_name.lower()}_probability.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f172a")
        plt.close()
        log.info(f"  ✅ Saved Mean SHAP bar chart: {out}")
        return out


def plot_mean_shap_stats(mean_shap_df, y_pred_prob, model_name="XGBoost"):
    """Create statistical summary plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Top 10 features
    top_10 = mean_shap_df.head(10)
    axes[0].barh(top_10['feature'], top_10['mean_shap_abs'], color="#3b82f6", alpha=0.85)
    axes[0].set_xlabel("Mean |SHAP|", fontsize=10)
    axes[0].set_title("Top 10 Features by Mean SHAP", fontsize=11, fontweight="bold")
    axes[0].grid(True, axis="x", alpha=0.3)
    axes[0].invert_yaxis()
    
    # Right: Distribution of predicted probabilities
    axes[1].hist(y_pred_prob, bins=40, color="#8b5cf6", alpha=0.7, edgecolor="#6d28d9")
    axes[1].set_xlabel("Predicted VMS Probability", fontsize=10)
    axes[1].set_ylabel("Frequency (# samples)", fontsize=10)
    axes[1].set_title("Distribution of Model Predictions", fontsize=11, fontweight="bold")
    axes[1].grid(True, axis="y", alpha=0.3)
    
    # Apply style
    for ax in axes:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_color("#334155")
    
    fig.patch.set_facecolor("#0f172a")
    fig.suptitle(
        f"XGBoost Mean SHAP Summary (Probability Space)",
        color="#f8fafc", fontsize=13, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    out = SHAP_DIR / f"mean_shap_stats_{model_name.lower()}_probability.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    plt.close()
    log.info(f"  ✅ Saved Mean SHAP stats plot: {out}")
    return out


def generate_summary_report(mean_shap_df, base_prob, model_name="XGBoost"):
    """Generate text summary report."""
    report = []
    report.append("=" * 80)
    report.append(f"MEAN SHAP ANALYSIS — {model_name.upper()} (PROBABILITY SPACE)")
    report.append("=" * 80)
    report.append("")
    
    report.append(f"Base Probability (Expected Value):  {base_prob:.6f}")
    report.append(f"Total Features:                      {len(mean_shap_df)}")
    report.append("")
    
    report.append("TOP 10 FEATURES BY MEAN |SHAP|:")
    report.append("-" * 80)
    for idx, row in mean_shap_df.head(10).iterrows():
        report.append(f"  {idx+1:2d}. {row['feature']:40s}  Mean SHAP: {row['mean_shap_abs']:10.6f}")
    
    report.append("")
    report.append("INTERPRETATION:")
    report.append("-" * 80)
    report.append("• Mean |SHAP| quantifies each feature's average contribution to VMS probability")
    report.append("• Values are directly computed in probability space (0-1 range)")
    report.append("• All contributions are naturally bounded within [0, 1]")
    report.append("• Interpretation: 'Feature X changes deposit probability by ±Y on average'")
    report.append("")
    report.append("TECHNICAL DETAILS:")
    report.append("-" * 80)
    report.append("• TreeExplainer with model_output='probability' directly provides probability-space SHAP")
    report.append("• Interventional feature perturbation ensures robust explanations")
    report.append("• Background dataset enables proper baseline probability calculation")
    report.append("• SHAP values are additive and sum to the difference between prediction and base")
    
    return "\n".join(report)


def main():
    log.info("═══ Mean SHAP Analysis for XGBoost (Probability Space) ═══")
    log.info("")

    # Load data
    X, y, feature_names = load_data()
    log.info(f"Loaded data: X.shape={X.shape}, y.shape={y.shape}")
    log.info(f"Feature names: {len(feature_names)} features")

    # Load model
    log.info(f"\nLoading XGBoost model from {XGB_MODEL_PATH} ...")
    model = joblib.load(XGB_MODEL_PATH)
    log.info("✅ Model loaded successfully")

    # Compute Mean SHAP
    log.info("\n─── Computing Mean SHAP (Probability Space) ───")
    results = compute_mean_shap_xgb_probability(model, X, feature_names)
    
    # Extract results
    mean_shap_df = results['mean_shap_df']
    base_prob = results['base_prob']
    y_pred_prob = results['y_pred_prob']

    # Save results
    log.info("\n─── Saving Results ───")
    save_mean_shap_table(mean_shap_df, model_name="XGBoost")
    
    # Generate visualizations
    log.info("\n─── Generating Visualizations ───")
    plot_mean_shap_bar(mean_shap_df, model_name="XGBoost", top_n=15)
    plot_mean_shap_stats(mean_shap_df, y_pred_prob, model_name="XGBoost")

    # Print summary
    log.info("\n─── Summary ───")
    summary = generate_summary_report(mean_shap_df, base_prob, model_name="XGBoost")
    log.info(summary)
    
    # Save summary report
    report_out = SHAP_DIR / "mean_shap_xgb_probability_report.txt"
    with open(report_out, "w", encoding="utf-8") as f:
        f.write(summary)
    log.info(f"\n✅ Report saved: {report_out}")
    
    log.info("\n" + "=" * 80)
    log.info("✅ Mean SHAP analysis complete!")
    log.info("=" * 80)


if __name__ == "__main__":
    main()
