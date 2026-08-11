"""
generate_workflow_diagram.py
──────────────────────────────
Generates an interconnected, publication-quality workflow flowchart (workflow.png)
for the VMS Deposit Prospectivity Mapping pipeline in the Bathurst Mining Camp.
Demonstrates clear data flow, cross-phase dependencies, and process interconnectivity.
"""

import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_workflow():
    fig, ax = plt.subplots(figsize=(15, 21), dpi=300)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 21)
    ax.axis("off")

    # Palette
    c_blue = "#1f77b4"
    c_green = "#2ca02c"
    c_orange = "#ff7f0e"
    c_purple = "#9467bd"
    c_red = "#d62728"
    
    # Title & Subtitle
    ax.text(7.5, 20.4, "Machine Learning VMS Prospectivity Mapping Interconnected Pipeline", 
            ha="center", va="center", fontsize=16.5, fontweight="bold", color="#111111")
    ax.text(7.5, 20.0, "Process Interconnectivity & Data-Flow Architecture — Bathurst Mining Camp (BMC)", 
            ha="center", va="center", fontsize=11.5, fontstyle="italic", color="#555555")

    # Phase Container Box Specifications
    phases = [
        ("Phase 1: Multi-Source Geoscience Data Compilation & Alignment", 19.3, 2.5, "#eef4fb", c_blue),
        ("Phase 2: Mathematical Preprocessing & Feature Synthesis", 15.8, 2.5, "#eef9ee", c_green),
        ("Phase 3: Hybrid Label Assembly & Dataset Pre-conditioning", 12.3, 2.5, "#fff5ea", c_orange),
        ("Phase 4: Spatial Block Cross-Validation & Model Training", 8.8, 2.5, "#f6efff", c_purple),
        ("Phase 5: Camp-Scale Prediction, GIS Mapping & SHAP Explainability", 5.3, 2.5, "#fdeeee", c_red),
    ]

    # Draw Phase Banners & Containers
    for title, y_top, height, bg_color, header_color in phases:
        rect = patches.FancyBboxPatch(
            (0.4, y_top - height), 14.2, height,
            boxstyle="round,pad=0.15,rounding_size=0.25",
            facecolor=bg_color, edgecolor=header_color, linewidth=1.6, zorder=1
        )
        ax.add_patch(rect)
        
        banner = patches.FancyBboxPatch(
            (0.6, y_top - 0.4), 13.8, 0.35,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            facecolor=header_color, edgecolor="none", zorder=2
        )
        ax.add_patch(banner)
        ax.text(7.5, y_top - 0.225, title, ha="center", va="center", 
                fontsize=11.0, fontweight="bold", color="white", zorder=3)

    # Card Helper Function
    def add_card(x, y, w, h, text, title="", border="#444444", font_size=8.5):
        card = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.1,rounding_size=0.18",
            facecolor="#ffffff", edgecolor=border, linewidth=1.2, zorder=4
        )
        ax.add_patch(card)
        if title:
            ax.text(x + w/2, y + h - 0.25, title, ha="center", va="center", 
                    fontsize=font_size + 0.5, fontweight="bold", color="#111111", zorder=5)
            ax.text(x + w/2, y + (h - 0.32)/2, text, ha="center", va="center", 
                    fontsize=font_size - 0.5, color="#333333", zorder=5, multialignment="center", linespacing=1.25)
        else:
            ax.text(x + w/2, y + h/2, text, ha="center", va="center", 
                    fontsize=font_size, color="#111111", zorder=5, multialignment="center", linespacing=1.25)

    # Arrow Connection Helper
    def draw_arrow(x1, y1, x2, y2, color="#444444", style="-|>", lw=1.5, ls="-"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw, linestyle=ls,
                                    mutation_scale=12),
                    zorder=6)

    # Multi-segment Orthogonal Path Helper
    def draw_path(points, color="#444444", style="-|>", lw=1.5, ls="-"):
        for i in range(len(points) - 2):
            ax.plot([points[i][0], points[i+1][0]], [points[i][1], points[i+1][1]],
                    color=color, lw=lw, linestyle=ls, zorder=6)
        draw_arrow(points[-2][0], points[-2][1], points[-1][0], points[-1][1],
                   color=color, style=style, lw=lw, ls=ls)

    # Label on Arrow Helper
    def add_arrow_label(x, y, text, color="#222222", bg="#ffffff", border="#aaaaaa", font_size=7.5):
        ax.text(x, y, text, ha="center", va="center", fontsize=font_size, fontweight="bold",
                color=color, zorder=7,
                bbox=dict(boxstyle="round,pad=0.15", facecolor=bg, edgecolor=border, lw=0.8, alpha=0.95))

    # ── CARDS POSITIONING ──
    w_card = 4.0
    h_card = 1.6
    
    # Phase 1 Cards (y_top=19.3, cards y=17.05)
    y_p1 = 17.05
    add_card(0.7, y_p1, w_card, h_card, "• TMI, Bouguer Gravity, & Radiometrics\n• Airborne Geophysical Datasets\n• Reprojected to EPSG:2953", "1A. Airborne Geophysics", c_blue)
    add_card(5.5, y_p1, w_card, h_card, "• 17 Pathfinder Elements (Ag, Cu, Pb, Zn...)\n• Regional Till Sampling Database\n• Coordinate Standardisation", "1B. Till Geochemistry", c_blue)
    add_card(10.3, y_p1, w_card, h_card, "• Known VMS Deposits (van Staal 2003)\n• GeoNB Mineral Drilling Records\n• Confirmed Barren Drill Intercepts", "1C. Deposit & Drill Database", c_blue)

    # Phase 2 Cards (y_top=15.8, cards y=13.55)
    y_p2 = 13.55
    add_card(0.7, y_p2, w_card, h_card, "• Native-Grid Fourier Derivatives\n• FVD, THG, TDR, & AS Filters\n• Radioelement Ratios (K/Th, U/Th)", "2A. Grid Derivatives", c_green)
    add_card(5.5, y_p2, w_card, h_card, "• Centered Log-Ratio (CLR) Transform\n• IDW 50m Surface Interpolation\n• Compositional PCA & FA Factor Scores", "2B. Compositional Geochem", c_green)
    add_card(10.3, y_p2, w_card, h_card, "• Master 100m Raster Alignment\n• Log-Transformed Raw Geochem\n• Multi-Element Anomaly Score (MEAS)", "2C. Master Raster Stack", c_green)

    # Phase 3 Cards (y_top=12.3, cards y=10.05)
    y_p3 = 10.05
    add_card(0.7, y_p3, w_card, h_card, "• 45 Confirmed VMS Deposit Centroids\n• Master Raster Extent Contained\n• 500m Buffer Geometry", "3A. Positive Deposit Labels", c_orange)
    add_card(5.5, y_p3, w_card, h_card, "• 125 Barren Exploration Drill Collars\n• 125 Mahalanobis Distance Candidates\n• Multi-Dimensional Feature Dissimilarity", "3B. Hybrid Negative Labels", c_orange)
    add_card(10.3, y_p3, w_card, h_card, "• Raster Sampling at Label Points\n• Exclusion of >75% Missing Features\n• SMOTE Minority Class Balancing (1:1)", "3C. Balanced Feature Matrix", c_orange)

    # Phase 4 Cards (y_top=8.8, cards y=6.55)
    y_p4 = 6.55
    add_card(0.7, y_p4, w_card, h_card, "• 5-Fold Spatial BlockKFold Split\n• Spatially Disjoint Block Partitioning\n• Eliminates Autocorrelation Leakage", "4A. Spatial Block CV", c_purple)
    add_card(5.5, y_p4, w_card, h_card, "• Random Forest & XGBoost Classifiers\n• Optuna Hyperparameter Optimization\n• Balanced Class Weighting", "4B. Model Optimization", c_purple)
    add_card(10.3, y_p4, w_card, h_card, "• Out-of-Fold ROC-AUC & AP Evaluation\n• Threshold-Free Balanced Accuracy\n• Area-Normalized Success Rate (SR-AUC)", "4C. Performance Evaluation", c_purple)

    # Phase 5 Cards (y_top=5.3, cards y=3.05)
    y_p5 = 3.05
    add_card(0.7, y_p5, w_card, h_card, "• Model Inference across Master Grid\n• 1,194,109 Master Raster Cells (100m)\n• Continuous Prospectivity Index (PI)", "5A. Camp-Scale Inference", c_red)
    add_card(5.5, y_p5, w_card, h_card, "• Georeferenced GeoTIFF Export\n• High-Priority Target Filtering (PI>0.7)\n• Cumulative Area Capture Analysis", "5B. Target Delineation & GIS", c_red)
    add_card(10.3, y_p5, w_card, h_card, "• TreeSHAP Feature Attribution\n• Global & Local Importance Rankings\n• Partial Dependence Analysis (PDP)", "5C. TreeSHAP Explainability", c_red)

    # ── CLEAN ORTHOGONAL INTERCONNECTIVITY ROUTING ──

    # 1. Geophysics (1A) -> Grid Derivatives (2A)
    draw_arrow(2.7, y_p1, 2.7, y_p2 + h_card, color=c_blue, lw=1.8)

    # 2. Till Geochemistry (1B) -> Compositional Geochem (2B)
    draw_arrow(7.5, y_p1, 7.5, y_p2 + h_card, color=c_blue, lw=1.8)

    # 3. Derivatives (2A) & Geochem (2B) -> Master Raster Stack (2C)
    draw_arrow(4.7, y_p2 + h_card*0.7, 10.3, y_p2 + h_card*0.7, color=c_green, lw=1.5)
    draw_arrow(9.5, y_p2 + h_card*0.3, 10.3, y_p2 + h_card*0.3, color=c_green, lw=1.5)
    add_arrow_label(7.5, y_p2 + h_card*0.7 + 0.22, "Raster Derivatives & Factor Scores", color=c_green, border=c_green)

    # 4. Deposit & Drill DB (1C) -> Positive (3A) & Negative Labels (3B)
    # Route through whitespace midpoint y = 12.8 (between Phase 2 container & Phase 3 banner)
    draw_path([(12.3, y_p1), (12.3, 16.3), (9.9, 16.3), (9.9, 12.8), (2.7, 12.8), (2.7, y_p3 + h_card)], color=c_blue, lw=1.5)
    add_arrow_label(5.0, 12.8, "Deposit Coordinates", color=c_blue, border=c_blue)
    
    draw_path([(9.9, 12.8), (7.5, 12.8), (7.5, y_p3 + h_card)], color=c_blue, lw=1.5)
    add_arrow_label(8.7, 12.8, "Barren Intercepts", color=c_blue, border=c_blue)

    # 5. Master Raster Stack (2C) -> Hybrid Negative Labels (3B) [Mahalanobis Feature Space]
    # Route through y = 13.0 above Phase 3 container
    draw_path([(10.3, y_p2 + 0.2), (5.1, y_p2 + 0.2), (5.1, 12.8), (7.5, 12.8), (7.5, y_p3 + h_card)], color=c_green, lw=1.4, ls="--")
    add_arrow_label(5.1, 13.0, "Mahalanobis Feature Space", color=c_green, border=c_green)

    # 6. Master Raster Stack (2C) -> Balanced Feature Matrix (3C) [Raster Feature Sampling]
    draw_arrow(12.3, y_p2, 12.3, y_p3 + h_card, color=c_green, lw=1.8)
    add_arrow_label(12.3, 12.8, "Raster Feature Sampling", color=c_green, border=c_green)

    # 7. Labels (3A & 3B) -> Balanced Feature Matrix (3C)
    draw_arrow(4.7, y_p3 + h_card/2, 5.5, y_p3 + h_card/2, color=c_orange, lw=1.5)
    draw_arrow(9.5, y_p3 + h_card/2, 10.3, y_p3 + h_card/2, color=c_orange, lw=1.6)
    add_arrow_label(9.9, y_p3 + h_card/2 + 0.22, "Labels (Y=1, Y=0)", color=c_orange, border=c_orange)

    # 8. Balanced Feature Matrix (3C) -> Spatial Block CV (4A)
    # Route through whitespace midpoint y = 9.3
    draw_path([(12.3, y_p3), (12.3, 9.3), (2.7, 9.3), (2.7, y_p4 + h_card)], color=c_orange, lw=1.8)
    add_arrow_label(7.5, 9.3, "Spatial Feature Matrix (X, Y)", color=c_orange, border=c_orange)

    # 9. Spatial Block CV (4A) -> Model Optimization (4B) -> Performance Evaluation (4C)
    draw_arrow(4.7, y_p4 + h_card/2, 5.5, y_p4 + h_card/2, color=c_purple, lw=1.8)
    add_arrow_label(5.1, y_p4 + h_card/2 + 0.22, "CV Folds", color=c_purple, border=c_purple)

    draw_arrow(9.5, y_p4 + h_card/2, 10.3, y_p4 + h_card/2, color=c_purple, lw=1.8)
    add_arrow_label(9.9, y_p4 + h_card/2 + 0.22, "OOF Predictions", color=c_purple, border=c_purple)

    # 10. Model Optimization (4B) [Trained Classifiers] -> Camp-Scale Inference (5A) & TreeSHAP (5C)
    # Route through whitespace midpoint y = 5.8
    draw_path([(7.5, y_p4), (7.5, 5.8), (2.7, 5.8), (2.7, y_p5 + h_card)], color=c_purple, lw=1.7)
    add_arrow_label(5.1, 5.8, "Trained Model (RF/XGB)", color=c_purple, border=c_purple)

    draw_path([(7.5, 5.8), (12.3, 5.8), (12.3, y_p5 + h_card)], color=c_purple, lw=1.7)
    add_arrow_label(9.9, 5.8, "Model Trees & Weights", color=c_purple, border=c_purple)

    # 11. Master Raster Stack (2C) -> Camp-Scale Inference (5A) [Full Master Grid Rasters]
    # Route down the right margin (x = 14.7)
    draw_path([(14.3, y_p2 + h_card/2), (14.7, y_p2 + h_card/2), (14.7, 2.4), (2.7, 2.4), (2.7, y_p5)], color=c_green, lw=1.6)
    add_arrow_label(8.5, 2.4, "Full 100m Master Grid Rasters", color=c_green, border=c_green)

    # 12. Camp-Scale Inference (5A) -> Target Delineation & GIS (5B)
    draw_arrow(4.7, y_p5 + h_card/2, 5.5, y_p5 + h_card/2, color=c_red, lw=1.8)
    add_arrow_label(5.1, y_p5 + h_card/2 + 0.25, "Prospectivity Index", color=c_red, border=c_red)

    plt.tight_layout()
    output_path = "workflow.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Successfully generated {output_path}")

    # Copy to target directories if they exist
    destinations = [
        r"c:\Users\delef\.gemini\antigravity\brain\fee87920-985b-47de-af6f-868d6a640943\workflow.png",
        r"C:\Users\delef\.gemini\antigravity-ide\brain\8ecc03ec-6bbe-4e1d-a5e3-3772be6ca400\workflow.png"
    ]
    for dest in destinations:
        dest_dir = os.path.dirname(dest)
        if os.path.exists(dest_dir):
            shutil.copy(output_path, dest)
            print(f"Copied workflow.png to {dest}")

if __name__ == "__main__":
    draw_workflow()
