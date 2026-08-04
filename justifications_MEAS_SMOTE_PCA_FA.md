# Methodological Justifications: MEAS, SMOTE, PCA, and FA
## VMS Deposit Discovery – Bathurst Mining Camp ML Prospectivity Pipeline

---

## 1. Multi-Element Anomaly Score (MEAS)

### Problem it Solves
The Bathurst Mining Camp (BMC) VMS system generates a **multi-element geochemical footprint** around each deposit — a diagnostic pathfinder signature driven by hydrothermal plume dispersal of Zn, Pb, Cu, Ag, As, Sb, Cd, and other elements simultaneously. No single element is a reliable standalone discriminator: Zn can be elevated by mafic host rocks; Pb can reflect background sedimentary sources; and Cu dispersal is highly sensitive to pH and distance from the vent. Relying on individual element concentrations as separate model features means the classifier must independently learn their joint significance — a difficult task with only 45 positive training examples.

### Why It Is Needed Here
The training dataset is **severely data-sparse** (45 positive labels; ~295 total points). With a high-dimensional raw geochemical feature space (17 elements + 17 IDW-interpolated surfaces), individual element signals are noisy, especially in the BMC's glaciated setting where dispersal trains are shortened and diluted. A MEAS compresses the correlated pathfinder signal into a single geologically meaningful scalar that:

- **Amplifies the collective anomaly** at VMS locations by summing normalized, geologically weighted concentrations of the diagnostic pathfinder suite (Zn, Pb, Cu, Ag, As, Sb, Cd).
- **Suppresses background noise** from non-pathfinder elements that dominate individual IDW rasters.
- **Provides an interpretable feature** for SHAP analysis, directly linking model predictions to the mineralizing hydrothermal system rather than individual element noise.
- **Reduces effective dimensionality** before model training, which is critical given the small sample size relative to the number of features.

### Geological Basis
BMC VMS deposits display characteristic bimodal-siliciclastic alteration footprints: a proximal Zn–Pb–Ag–Cd massive sulphide lens overlies a Cu–As–Sb stockwork zone, and distal Fe–Mn formation marks the oxidized plume (Goodfellow, 2007). MEAS formalizes this multi-element association into a prospectivity-weighted index, consistent with established pathfinder geochemistry practice in glaciated VMS exploration terrain (Parkhill and Doiron, 2003). IDW interpolation spatially smooths individual elements, suppressing the local high-amplitude anomalies near deposits. MEAS, applied at the point scale before smoothing, preserves those diagnostic local contrasts.

### Mathematical Form
$$\text{MEAS} = \sum_{i \in \mathcal{P}} w_i \cdot \tilde{x}_i$$

where $\mathcal{P}$ is the set of VMS pathfinder elements, $\tilde{x}_i$ is the standardized concentration (zero mean, unit variance), and $w_i$ is the geological weight reflecting the element's diagnostic association strength with BMC VMS mineralization.

---

## 2. Synthetic Minority Over-Sampling Technique (SMOTE)

### Problem it Solves
The BMC training dataset has a fundamental **class imbalance**: 45 positive VMS labels versus 250 negative (barren/background) labels — a ratio of approximately **1:5.5**. When a classifier is trained on imbalanced data, it is incentivized to minimize overall loss by predicting the majority class (barren) for all observations. The result is a model with high accuracy but poor sensitivity: it correctly identifies barren areas but fails to flag VMS-prospective zones — which is the exact task this pipeline needs to accomplish.

### Why It Is Needed Here
This is not a simple over/under-representation problem — the imbalance reflects the physical reality of mineral exploration: VMS deposits are genuinely rare relative to the regional background. There are only 45 known deposits in the entire 3,800 km² camp. Options considered:

| Strategy | Issue |
|---|---|
| Do nothing (imbalanced training) | Classifier biased toward majority class; very low VMS recall |
| Random duplication of positives | Overfits to exact feature values of 45 deposits; no generalization |
| Class-weighted loss | Helps but does not enrich the minority manifold in feature space |
| **SMOTE** | Synthesizes *new* plausible VMS feature vectors by interpolating between real positive nearest neighbours |

SMOTE (Chawla et al., 2002) addresses this by generating **synthetic positive examples** through linear interpolation between k-nearest VMS neighbours in the multi-dimensional feature space. This:
- Expands the positive class manifold, giving the classifier a richer representation of the VMS feature space.
- Balances the dataset to 250 positive and 250 negative samples ($n = 500$) without duplicating information.
- Prevents the model from memorizing the 45 exact deposit locations while still learning the geophysical-geochemical signature associated with VMS mineralization.

### Critical Note: SMOTE Applied Within CV Folds
To prevent **data leakage**, SMOTE is applied only to the training folds within each spatial block cross-validation iteration — never to the validation fold. This ensures that synthetic samples never contaminate the held-out evaluation set, and that reported AUC metrics reflect true model generalization.

### Relevance to Spatial Block CV
The spatial block cross-validation further exacerbates effective class imbalance: when an entire geographic block is held out, the number of positive samples in the training folds can drop to as few as 30–35 deposits. SMOTE applied within each training fold stabilizes the training class ratio regardless of which deposits fall into which block, improving consistency of gradient signals across folds.

---

## 3. Principal Component Analysis (PCA)

### Problem it Solves
The 17-element geochemical dataset is **highly collinear**. VMS-related elements (Zn, Pb, Cu, Ag, Cd, In) co-occur in ore zones; mafic-hosted elements (Fe, Co, Ni, Mn) co-vary with lithological background; and alteration elements (As, Sb, Tl) correlate with proximity to hydrothermal conduits. These correlations violate the effective independence assumption of many feature importance metrics, inflate the apparent number of informative dimensions, and make it difficult for the model to separate the VMS signal from background geology.

### The Compositional Problem Requiring CLR-PCA
Geochemical concentration data are **compositional** — they sum to a constant (10⁶ ppm), which introduces the Aitchison simplex constraint. In raw concentration space, correlations between elements are mathematically forced by the closure effect and do not reflect true geochemical relationships (Aitchison, 1986). Applying standard PCA to raw concentrations produces **spurious negative correlations** among unrelated elements and mixes the lithological-hydrothermal signal with closure-induced mathematical artefacts.

The Centered Log-Ratio (CLR) transformation lifts the data from the simplex onto unconstrained Euclidean space:
$$\text{clr}(\mathbf{x}) = \left[\ln\frac{x_1}{g(\mathbf{x})},\ \ln\frac{x_2}{g(\mathbf{x})},\ \dots,\ \ln\frac{x_D}{g(\mathbf{x})}\right]$$

CLR-PCA then captures orthogonal axes of **genuine** geochemical variance (Filzmoser et al., 2009):
- **PC1 (Lithology):** Expected to separate mafic (Fe, Co, Ni, Mn) from felsic (Ba) background, reflecting the bimodal volcanic stratigraphy of the BMC.
- **PC2 (VMS Hydrothermal):** Expected to concentrate Zn, Pb, Cu, Ag, As, Sb, Cd, Tl — the hydrothermal pathfinder suite.
- **PC3–PC5:** Secondary alteration or sedimentary controls (Mo, Sn, Bi).

### Why It Is Needed as a Feature
When IDW-interpolated geochemical rasters are used as model features, they carry 17 correlated inputs for the same information. Including CLR-PCA scores:
1. **Reduces dimensionality** from 17 correlated elements to 4–5 orthogonal components, reducing the curse of dimensionality given the small training set.
2. **Provides spatially continuous surfaces** (mapped back to 50 m rasters) that summarize regional lithological and hydrothermal variation across the full camp extent.
3. **Improves SHAP interpretability**: a single "VMS Hydrothermal PC" feature is more geologically meaningful than eight individual correlated elements in a feature importance plot.
4. **Reduces multicollinearity** that degrades Random Forest's ability to correctly partition feature importance among correlated predictors.

---

## 4. Factor Analysis (FA) with Varimax Rotation

### How FA Differs from PCA — and Why Both Are Used
PCA finds orthogonal directions of **maximum variance** in the dataset, regardless of whether that variance is geologically meaningful or noise-driven. Factors from FA model the **latent structure** underlying the correlations — specifically, they ask: "What unobserved processes (factors) could produce the observed correlation matrix?" FA separates the variance of each element into:
- **Common variance** (shared with other elements; explained by latent factors).
- **Unique variance** (specific to that element; treated as measurement noise or local anomaly).

This distinction is geologically important: in geochemical data, each element's concentration is driven by a combination of regional geology (mafic vs. felsic lithology), hydrothermal alteration (VMS footprint), and local dispersion (glacial transport, background sediment). FA explicitly models these shared geological processes as latent factors.

### Varimax Rotation
Varimax rotation maximizes the variance of squared loadings within each factor, driving loadings toward 0 or 1. This produces **simpler, more interpretable factor structures** than unrotated PCA: each element loads strongly on one factor and weakly on others. In geochemical applications, varimax-rotated factors correspond more cleanly to distinct geological processes — one factor for VMS pathfinders, one for mafic lithology, one for sedimentary background — than unrotated PCs which blend these signals (Filzmoser et al., 2009; Carranza, 2008).

### Complementarity with PCA
| Property | CLR-PCA | CLR-FA (varimax) |
|---|---|---|
| Objective | Maximum variance | Common factor structure |
| Rotation | Orthogonal (unrotated) | Varimax (simplified loadings) |
| Unique variance | Retained in components | Explicitly separated |
| Interpretability | Moderate | High |
| Spatial sensitivity | Global variance | Shared process signal |

Including **both** PCA and FA scores as model features provides:
1. **PCA scores** capture the maximum-variance geochemical gradients across the camp (including background lithological variation).
2. **FA scores** capture the shared alteration-hydrothermal signal with cleaner element-to-factor assignment, reducing noise from unique element variance.

The two sets of components together give the RF/XGBoost classifiers complementary views of the geochemical landscape — one emphasizing total variance, the other emphasizing shared alteration processes. This redundancy is intentional: it ensures that the model has access to the hydrothermal footprint signature regardless of whether a particular decomposition method better resolves it in a given spatial block.

### Geological Basis for the BMC
Given the BMC's bimodal volcanic stratigraphy and polyphase deformation, the geochemical landscape reflects at least three distinct processes:
1. **Regional lithological background** (mafic vs. felsic volcanic units → Fe, Co, Ni, Mn vs. Ba, Mo).
2. **VMS hydrothermal footprint** (Zn, Pb, Cu, Ag, As, Sb, Tl, Cd, In) — the primary exploration target.
3. **Sedimentary / structural overprint** (Sn, Bi — trace critical minerals with localized source controls).

FA with varimax rotation is optimally suited to cleanly resolve these three geological processes as separate factors, directly supporting the physical interpretation of model predictions via SHAP.

---

## Summary Table

| Method | Primary Problem Addressed | Specific Context in This Study |
|---|---|---|
| **MEAS** | Sparse positive labels + noisy individual pathfinder signals | Compress 17-element VMS signature into one geologically weighted score; preserve local high-amplitude anomalies lost during IDW interpolation |
| **SMOTE** | 1:5.5 class imbalance (45 VMS vs. 250 negatives) | Synthesize minority-class feature vectors within each CV training fold to prevent bias toward barren prediction; stabilize gradient signal across spatial blocks |
| **PCA (CLR)** | Compositional closure bias + high collinearity among 17 elements | Lift data off the simplex; produce orthogonal variance-maximizing components mapping regional lithology vs. hydrothermal gradients across the 3,800 km² camp |
| **FA (CLR + varimax)** | Need for geologically interpretable latent factor structure | Separate shared alteration-hydrothermal signal from unique element variance; produce clean element-to-process loadings for SHAP interpretation and spatial mapping |

---

## References

- Aitchison, J. (1986). *The Statistical Analysis of Compositional Data*. Chapman and Hall.
- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *JAIR*, 16, 321–357.
- Carranza, E. J. M. (2008). *Geochemical Anomaly and Mineral Prospectivity Mapping in GIS*. Elsevier.
- Filzmoser, P., Hron, K., & Reimann, C. (2009). Principal component analysis for compositional data with outliers. *Environmetrics*, 20(6), 621–632.
- Goodfellow, W. D. (2007). Metallogeny of the Bathurst Mining Camp. *Economic Geology Monograph 11*.
- Parkhill, M. A., & Doiron, A. (2003). Till geochemistry of the Bathurst Mining Camp. *Economic Geology Monograph 11*.
- Parsa, M., Lentz, D. R., & Walker, J. A. (2023). Predictive modeling of VHMS deposits, Bathurst Mining Camp. *Natural Resources Research*, 32, 19–36.
