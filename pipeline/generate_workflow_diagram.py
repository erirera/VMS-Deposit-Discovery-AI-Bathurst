"""
generate_workflow_diagram.py
──────────────────────────────
Generates a crisp, publication-quality workflow flowchart (workflow.png)
for the VMS Deposit Prospectivity Mapping pipeline in the Bathurst Mining Camp.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_workflow():
    fig, ax = plt.subplots(figsize=(14, 18), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 18)
    ax.axis("off")

    # Colors
    c_blue = "#1f77b4"
    c_green = "#2ca02c"
    c_orange = "#ff7f0e"
    c_purple = "#9467bd"
    c_red = "#d62728"
    c_dark = "#222222"
    
    bg_box = "#f8f9fa"
    border_color = "#cccccc"

    # Title
    ax.text(7, 17.4, "Machine Learning VMS Prospectivity Mapping Pipeline", 
            ha="center", va="center", fontsize=18, fontweight="bold", color="#111111")
    ax.text(7, 17.0, "Bathurst Mining Camp (BMC), New Brunswick, Canada", 
            ha="center", va="center", fontsize=13, fontstyle="italic", color="#555555")

    # Pipeline Phase Container Boxes
    phases = [
        ("Phase 1: Multi-Source Geoscience Data Compilation", 13.5, 3.0, "#eef4fb", c_blue),
        ("Phase 2: Mathematical Preprocessing & Feature Engineering", 10.1, 3.0, "#eef9ee", c_green),
        ("Phase 3: Hybrid Label Strategy & Spatial Dataset Assembly", 6.7, 3.0, "#fff5ea", c_orange),
        ("Phase 4: Spatial Block Cross-Validation & Model Training", 3.3, 3.0, "#f6efff", c_purple),
        ("Phase 5: Prospectivity Mapping & SHAP Explainability", -0.1, 3.0, "#fdeeee", c_red),
    ]

    for title, y_top, height, bg_color, header_color in phases:
        rect = patches.FancyBboxPatch(
            (0.5, y_top - height), 13.0, height,
            boxstyle="round,pad=0.2,rounding_size=0.3",
            facecolor=bg_color, edgecolor=header_color, linewidth=1.8, zorder=1
        )
        ax.add_patch(rect)
        
        # Phase Header Banner
        banner = patches.FancyBboxPatch(
            (0.7, y_top - 0.5), 12.6, 0.45,
            boxstyle="round,pad=0.1,rounding_size=0.15",
            facecolor=header_color, edgecolor="none", zorder=2
        )
        ax.add_patch(banner)
        ax.text(7.0, y_top - 0.28, title, ha="center", va="center", 
                fontsize=12, fontweight="bold", color="white", zorder=3)

    # Box Helper Function
    def add_card(x, y, w, h, text, title="", color="#ffffff", border="#444444", font_size=9.5):
        card = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.1,rounding_size=0.2",
            facecolor=color, edgecolor=border, linewidth=1.2, zorder=4
        )
        ax.add_patch(card)
        if title:
            ax.text(x + w/2, y + h - 0.25, title, ha="center", va="center", 
                    fontsize=font_size, fontweight="bold", color="#111111", zorder=5)
            ax.text(x + w/2, y + (h - 0.3)/2, text, ha="center", va="center", 
                    fontsize=font_size - 1, color="#333333", zorder=5, multialignment="center")
        else:
            ax.text(x + w/2, y + h/2, text, ha="center", va="center", 
                    fontsize=font_size, color="#111111", zorder=5, multialignment="center")

    # Arrow Helper Function
    def add_arrow(x1, y1, x2, y2, color="#555555"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0, mutation_scale=15),
                    zorder=6)

    # ── Phase 1 Content ──
    add_card(0.9, 11.0, 3.6, 1.8, "TMI, Bouguer Gravity,\nRadiometrics (%K, eTh, eU)\n(NRCan & NBDNRED)", "Geophysics Grids", "#ffffff", c_blue)
    add_card(5.2, 11.0, 3.6, 1.8, "2,753 Sample Locations\n17 Pathfinder Elements\n(Ag, As, Cu, Pb, Zn, etc.)", "Till Geochemistry", "#ffffff", c_blue)
    add_card(9.5, 11.0, 3.6, 1.8, "45 VMS Deposits (van Staal 2003)\n10,377 GeoNB Drill Collars\n(100% Inside BMC Grid)", "Mineral Labels & Drilling", "#ffffff", c_blue)

    add_arrow(2.7, 10.9, 2.7, 10.0)
    add_arrow(7.0, 10.9, 7.0, 10.0)
    add_arrow(11.3, 10.9, 11.3, 10.0)

    # ── Phase 2 Content ──
    add_card(0.9, 7.6, 3.6, 1.9, "Fourier-Domain Derivatives\non Native Grids:\n• FVD, THG, TDR, AS\n• Radioelement Ratios (K/Th)", "Native Grid Derivatives", "#ffffff", c_green)
    add_card(5.2, 7.6, 3.6, 1.9, "CoDA Transformation:\n• CLR Transformation\n• Compositional PCA & FA\n• IDW Interpolated Surfaces", "Geochem CoDA & IDW", "#ffffff", c_green)
    add_card(9.5, 7.6, 3.6, 1.9, "Multi-Element Anomaly Score\n(MEAS Cu-Zn-Pb-As)\nMaster BMC Grid Alignment\n(EPSG:2953, 100m Res)", "MEAS & Grid Alignment", "#ffffff", c_green)

    add_arrow(2.7, 7.5, 2.7, 6.6)
    add_arrow(7.0, 7.5, 7.0, 6.6)
    add_arrow(11.3, 7.5, 11.3, 6.6)

    # ── Phase 3 Content ──
    add_card(0.9, 4.2, 3.6, 1.9, "45 Positive VMS Deposits\n100% Raster Contained\n500m Buffer Geometry", "Positive VMS Labels", "#ffffff", c_orange)
    add_card(5.2, 4.2, 3.6, 1.9, "125 GeoNB Barren Collars\n125 Mahalanobis Feature\nDissimilarity Pseudo-Absences\n(Parsa & Cumani, 2025)", "Hybrid Negative Labels", "#ffffff", c_orange)
    add_card(9.5, 4.2, 3.6, 1.9, "SMOTE Class Balancing:\n45 -> 250 Positive Samples\n(Total n = 500, 1:1 Ratio)\n59 Feature Columns", "Balanced Feature Matrix", "#ffffff", c_orange)

    add_arrow(2.7, 4.1, 2.7, 3.2)
    add_arrow(7.0, 4.1, 7.0, 3.2)
    add_arrow(11.3, 4.1, 11.3, 3.2)

    # ── Phase 4 Content ──
    add_card(0.9, 0.8, 3.6, 1.9, "5-Fold Spatial Block CV\n(BlockKFold)\nEliminates Autocorrelation\nData Leakage", "Spatial Cross-Validation", "#ffffff", c_purple)
    add_card(5.2, 0.8, 3.6, 1.9, "Optuna Hyperparameter Search\n• Random Forest (50 Trials)\n• XGBoost (50 Trials)\nBalanced Class Weights", "Classifier Optimization", "#ffffff", c_purple)
    add_card(9.5, 0.8, 3.6, 1.9, "Spatial CV Performance:\n• RF ROC-AUC: 0.927 ± 0.047\n• XGB ROC-AUC: 0.915 ± 0.056\n• RF PR-AUC: 0.740 ± 0.135", "Spatial CV Evaluation", "#ffffff", c_purple)

    add_arrow(2.7, 0.7, 2.7, -0.2)
    add_arrow(7.0, 0.7, 7.0, -0.2)
    add_arrow(11.3, 0.7, 11.3, -0.2)

    # ── Phase 5 Content ──
    add_card(0.9, -2.6, 3.6, 1.9, "Full-Extent Prediction:\n1,194,109 Grid Cells (100m)\nGeoTIFF & High-Res Maps\n(PI > 0.7 Priority Targets)", "Full-Extent Mapping", "#ffffff", c_red)
    add_card(5.2, -2.6, 3.6, 1.9, "Success Rate AUC:\n• RF SR-AUC = 0.9679\n• XGB SR-AUC = 0.9395\nTop 10% Area -> 91.1% VMS", "Discovery Efficiency", "#ffffff", c_red)
    add_card(9.5, -2.6, 3.6, 1.9, "TreeSHAP Explainability:\n• Mo & Pb Pathfinder Footprints\n• Th/K Potassic Alteration Halos\n• Gravity Horizontal Gradient", "SHAP Feature Attribution", "#ffffff", c_red)

    # Adjust plot limits for negative Y
    ax.set_ylim(-3.0, 18.0)

    plt.tight_layout()
    plt.savefig("workflow.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Successfully generated workflow.png")

if __name__ == "__main__":
    draw_workflow()
