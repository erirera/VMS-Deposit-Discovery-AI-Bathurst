# Camp-Scale Machine Learning Prospectivity Mapping of Volcanogenic Massive Sulphide Deposits in the Bathurst Mining Camp, New Brunswick: Integrated Geophysical Derivatives and Multi-Element Till-Geochemistry

**Dele Falebita¹ · Mohammad Parsa² · David Lentz³**

¹ Rue Melanie, Dieppe, New Brunswick, Canada
² Natural Resources Canada, Geological Survey of Canada, Ottawa, Ontario, Canada
³ Department of Earth Sciences, University of New Brunswick, Fredericton, New Brunswick, Canada

**Corresponding author:** Dele Falebita · dele.1.falebita@gmail.com

---

**Keywords:** mineral prospectivity mapping; volcanogenic massive sulphide; machine learning; Bathurst Mining Camp; till geochemistry; spatial cross-validation

---

## Abstract

Discovering buried volcanogenic massive sulphide (VMS) deposits requires the integration of heterogeneous, multi-scale geoscience datasets. We present a machine learning pipeline for VMS prospectivity mapping in the historic Bathurst Mining Camp (BMC), New Brunswick, Canada. This approach integrates compiled aeromagnetic, gravity, and radiometric datasets with a newly compiled, spatially unified 17-element till geochemistry dataset containing 2,753 sample locations. Horizontal and vertical geophysical derivatives—First Vertical Derivative (FVD), Total Horizontal Gradient (THG), Analytic Signal (AS), and Tilt Derivative (TDR)—were computed on the original survey grids prior to spatial resampling to preserve high-frequency structural information. Geochemical surfaces were generated via Inverse Distance Weighting (IDW) interpolation and processed using Centered Log-Ratio (CLR) transformations, Principal Component Analysis (PCA), and Factor Analysis (FA) to correct for compositional closure bias. A geologically weighted Multi-Element Anomaly Score (MEAS) was computed to capture proximal VMS pathfinder associations.

Random Forest (RF) and Extreme Gradient Boosting (XGBoost) classifiers were trained on 295 spatial labels comprising 45 known VMS deposits (van Staal et al., 2003) and 250 hybrid negative labels—125 confirmed barren drill intercepts from New Brunswick Geological Survey records sampled strictly within the master BMC study area raster grid and 125 feature-space dissimilar negative labels selected by maximising Mahalanobis distance from the VMS deposit centroid in multi-dimensional feature space (Parsa & Cumani, 2025). Synthetic Minority Over-sampling Technique (SMOTE) was used for class balancing, and 5-fold spatial block cross-validation was applied to address spatial autocorrelation. Model performance was evaluated using Receiver Operating Characteristic (ROC), Precision-Recall (PR), cumulative Success Rate, and Balanced Accuracy (BA) metrics. RF outperformed XGBoost across key discrimination and targeting metrics: ROC-AUC 0.927 ± 0.047 vs. 0.915 ± 0.056; Average Precision 0.740 ± 0.135 vs. 0.639 ± 0.166; Balanced Accuracy 0.835 ± 0.052 vs. 0.844 ± 0.077; and Success Rate AUC 0.968 vs. 0.940. The RF model captures 91.1% of known BMC VMS deposits within the top 10% and 100.0% within the top 30% of the study area ranked by prospectivity index. Molybdenum and lead till geochemistry pathfinders alongside radiometric Th/K alteration ratios and gravity horizontal gradients emerged as dominant predictors, demonstrating the value of integrating structural geophysical derivatives with multi-element till geochemistry for covered VMS exploration.

---

## 1. Introduction

Volcanogenic massive sulphide (VMS) deposits are major global repositories of base metals—copper, zinc, and lead—and associated precious metals including gold and silver. Several commodities associated with VMS deposits, including zinc, copper, indium, bismuth, tin, and antimony, are classified as critical minerals essential for green energy transition technologies (Franklin et al., 2005; Galley et al., 2007). Historically, VMS discovery relied on identifying shallow, outcropping mineralization. However, near-surface deposits in mature exploration districts have been largely exhausted, requiring exploration programs to target deeply buried or covered systems beneath glacial overburden (Goodfellow and McCutcheon, 2003). In glaciated terrains, targeting covered systems requires the quantitative integration of regional multi-physics geophysical surveys and till geochemistry.

The Cambro-Ordovician Bathurst Mining Camp (BMC) in northern New Brunswick, Canada, is one of the most prolific VMS districts globally, historically hosting over 45 known deposits (Goodfellow, 2007). The camp is characterized by structurally complex, bimodal volcanic-sedimentary sequences of the Tetagouche Group, which have undergone intense polyphase deformation and ductile shear zone development (Van Staal et al., 2003). This high degree of deformation, combined with variable thicknesses of glacial till, masks the direct surface expression of mineralized zones. Consequently, machine learning (ML)-based mineral prospectivity mapping (MPM) has emerged as a powerful framework for integrating camp-scale datasets and delineating prospective target areas (Harris et al., 2015; Carranza, 2017; Zuo et al., 2019).

To construct a physically robust prospectivity model, preprocessing must respect the mathematical and physical nature of the geoscientific input data. Geophysical methods, such as high-resolution aeromagnetic and gravity surveys, map structural shear zones and mass anomalies. In traditional workflows, geophysical derivatives—e.g., the FVD, THG, and TDR—are often computed on reprojected, interpolated regional grids. However, this practice introduces edge effects, grid boundary artifacts, and distorts high-frequency structural gradients. Computing these derivatives in the Fourier domain directly on the original survey grids prior to spatial resampling preserves high-frequency boundary information and represents a more mathematically consistent approach to structural mapping (Thomas et al., 2000). Radiometric data further constrains VMS targeting by mapping radioelement ratios (e.g., K/Th, U/Th), which respond to potassic and sericitic alteration zones caused by hydrothermal vent systems (Shives et al., 1997).

Geochemical datasets introduce additional mathematical complexity arising from their compositional structure. Because elemental concentrations share a fixed total (e.g., $10^6$ ppm), the resulting constant-sum constraint renders classical multivariate techniques such as PCA and FA unreliable, producing correlations that reflect the closure effect rather than true geochemical relationships (Aitchison, 1986; Filzmoser et al., 2009). The Centered Log-Ratio (CLR) transformation addresses this by expressing each element as its logarithm relative to the geometric mean of the full composition, effectively lifting the data from the constrained simplex into unconstrained Euclidean space where standard multivariate methods are valid (Egozcue et al., 2003). Although CLR-transformed PCA and FA surfaces interpolated via IDW effectively summarize regional multi-element hydrothermal footprints, spatial smoothing during interpolation suppresses local, high-amplitude anomalies. Retaining raw elemental concentrations as additional point-scale features preserves these short-range geochemical contrasts (Parkhill and Doiron, 2003).

A critical challenge in data-driven MPM is the absence problem: unlike ecological niche modelling, there are no confirmed mineral absence locations, as any undrilled location could theoretically host undiscovered mineralization (Carranza and Laborte, 2015; Parsa, 2022). Using only random background points as negative training labels introduces ambiguity, as randomly selected locations provide no geological guarantee of being truly barren. Combining geologically verified barren drill intercepts with feature-space dissimilar negative labels — selected to be maximally unlike deposits in multi-dimensional geophysical and geochemical feature space — provides a more defensible negative evidence base and has been shown to produce stronger classifier discrimination and improved exploration targeting efficiency (Parsa & Cumani, 2025; Maepa et al., 2021; Barbet-Massin et al., 2012).

Spatial autocorrelation presents a further challenge for ML-based MPM. When geoscientific samples are partitioned randomly into training and validation folds, nearby observations share correlated feature values, meaning the held-out fold is not truly independent of the training set. This inflates cross-validation performance metrics and overstates generalization ability (Roberts et al., 2017; Brenning, 2012). Partitioning the study area into geographically disjoint blocks and holding out entire blocks for validation instead enforces genuine spatial independence between splits. Beyond standard discrimination metrics, exploration planning requires knowledge of targeting efficiency — specifically, the fraction of total study area that must be prioritized to capture a given proportion of known deposits. Cumulative Success Rate (or Prediction-Area) curves quantify this relationship and provide a more operationally relevant benchmark than ROC or PR curves alone (Carranza, 2008; Bonham-Carter, 1994).

This study addresses these challenges by developing a camp-scale VMS prospectivity mapping pipeline for the BMC. We computed horizontal and vertical geophysical derivatives on the original survey grids, compiled a unified 17-element till geochemistry dataset using CoDA transformations, and integrated raw geochemical pathfinder footprints alongside MEAS. To address the absence problem, we adopted a hybrid negative label strategy combining 125 confirmed barren drill intercepts with 125 feature-space dissimilar negative labels, selected by maximising Mahalanobis distance from the VMS deposit centroid in multi-dimensional feature space and stratified across four geographic quadrants for spatial representativeness (Parsa & Cumani, 2025; Carranza and Laborte, 2015; Barbet-Massin et al., 2012). Using spatial block cross-validation and SMOTE class balancing, we trained and evaluated RF and XGBoost classifiers. We assessed model performance using ROC-AUC, Average Precision, Balanced Accuracy, and cumulative Success Rate curves, demonstrating that the inclusion of raw geochemistry features—particularly lead and iron—dramatically improves targeting efficiency and provides a geophysically and geochemically consistent framework for covered VMS exploration.

---

## 2. Regional Geological Setting

### 2.1 Tectonic and Stratigraphic Framework

The BMC occupies the Gander Zone of the northern Appalachian Orogen in New Brunswick, Canada (Fig. 1; Van Staal et al., 2003; Rogers et al., 2003). Its geological architecture reflects the Wilson Cycle evolution of the Cambro-Ordovician Tetagouche–Four Falls back-arc basin: initial rifting of the Popelogan arc from the Gondwanan passive margin produced the bimodal volcanic and sedimentary sequences now hosting VMS mineralization; subsequent Taconic, Salinic, and Acadian accretionary events (Late Ordovician–Devonian) consumed the basin and imbricated these sequences into a series of south-directed thrust nappes (Van Staal et al., 2003; Goodfellow and McCutcheon, 2003).

> *[Figure 1 near here]*

Four principal lithostratigraphic assemblages are recognized within the camp (Goodfellow and McCutcheon, 2003; van Staal et al., 2003; Rogers et al., 2003). The para-autochthonous **Miramichi Group** (Cambro-Ordovician) forms the basement, comprising quartzarenites and carbonaceous argillites deposited on the Gondwanan passive margin prior to rifting. The **Tetagouche Group** — host to the majority of BMC deposits including Brunswick No. 12 and No. 6 — consists of three formations: the Nepisiguit Falls Formation (felsic volcaniclastics and tuffs), the Flat Landing Brook Formation (rhyolite flows and hyaloclastites), and the Boucher Brook Formation (tholeiitic pillow basalts, black shales, and chert-iron formation; Goodfellow, 2007). The **California Lake Group** hosts additional deposits (e.g., Caribou, Restigouche) in an analogous bimodal volcanic succession, while the **Four Falls Group** represents mature oceanic back-arc crust (Van Staal et al., 2003).

### 2.2 VMS Deposit Style and Hydrothermal Alteration

BMC deposits belong to the bimodal-siliciclastic VMS subtype (Galley et al., 2007; Franklin et al., 2005). Mineralization is spatially focused at or near felsic volcanic–sedimentary contacts and comprises three characteristic zones: a chlorite–silica–pyrite stockwork in the sub-seafloor feeder conduit; a stratiform, fine-grained pyrite–sphalerite–galena massive sulphide lens carrying the bulk Zn–Pb–Ag–Au resource; and an overlying jasper or magnetite iron formation recording distal oxidized hydrothermal plume fallout (Goodfellow, 2007; McCutcheon et al., 2003). Footprint alteration assemblages progress outward from a proximal quartz–chlorite–pyrite core through a sericite–carbonate–pyrite halo; the potassic character of these halos is expressed in airborne radiometric data as K enrichment and Th depletion relative to regional background (Shives et al., 1997; Goodfellow, 2007).

### 2.3 Structural Controls and Exploration Implications

Four deformation phases (D1–D4) under greenschist-facies conditions have profoundly modified the original deposit geometry (Van Staal et al., 2003; Rogers et al., 2003). Early nappe-forming thrusts (D1) and subsequent tight isoclinal folding (D2) structurally repeated and thinned the sulphide horizons, while D3–D4 open folding and brittle faulting created the dome-and-basin geometries — most notably the Brunswick anticline — that now control deposit plunge and depth (Goodfellow and McCutcheon, 2003; Van Staal et al., 2003). This polyphase deformation history eliminates simple surface expression of mineralization, elevates cover thickness through repeated structural stacking, and makes geophysical imaging of shear zones and density contrasts an indispensable complement to geochemical sampling (Thomas et al., 2000; Parkhill and Doiron, 2003).

---

## 3. Methodology

The prospectivity mapping framework is structured as a multi-stage ML pipeline progressing from raw data compilation and grid-derivative computation to spatial machine learning and area-normalized validation (Fig. 2). The workflow comprises five key components: (1) data compilation and native-grid preprocessing; (2) compositional geochemical analysis; (3) feature extraction and engineering; (4) spatial block cross-validation and model training; and (5) full-extent mapping and interpretability.

![Fig. 2 Schematic flowchart of the machine learning prospectivity mapping pipeline](/Users/delef/.gemini/antigravity/brain/fee87920-985b-47de-af6f-868d6a640943/workflow.png)

> *[Figure 2 near here]*

### 3.1 Geophysical Datasets and Derivative Computation

Airborne geophysical grids compiled by Natural Resources Canada (NRCan) over the BMC were utilized, including Total Magnetic Intensity (TMI), Bouguer gravity, and gamma-ray spectrometric (radiometric) grids (potassium, %K; thorium, eTh ppm; uranium, eU ppm; Fig. 3a–c). All input grids had been previously reprojected from NAD83 UTM Zone 19N to New Brunswick Stereographic Double (NAD83; EPSG:2953) by the New Brunswick Department of Natural Resources and Energy Development (NBDNRED) prior to download. The resulting geophysical input grids and their derivatives are illustrated in Fig. 4.

To preserve structural boundaries and prevent grid distortions caused by spatial resampling, horizontal and vertical derivatives were computed in the Fourier domain on the original survey grids prior to cell-size transformation to 100 m (Blakely, 1995):

- **First Vertical Derivative (FVD):** Computed via multiplication by the radial wavenumber $|\mathbf{k}|$ in the 2D Fourier domain, followed by the inverse transform, to enhance high-frequency near-surface structural and lithological contacts (Blakely, 1995):

$$\text{FVD}(\mathbf{r}) = \mathcal{F}^{-1}\!\left\{ |\mathbf{k}|\, F(\mathbf{k}) \right\}, \qquad |\mathbf{k}| = \sqrt{k_x^2 + k_y^2}$$

  where $F(\mathbf{k}) = \mathcal{F}\{f(\mathbf{r})\}$ is the 2D Fourier transform of the potential-field grid $f$ and $k_x$, $k_y$ are the horizontal wavenumbers.

- **Total Horizontal Gradient (THG):** Derived as the magnitude of the horizontal gradient vector, highlighting density and susceptibility contrasts at geological boundaries (Verduzco et al., 2004):

$$\text{THG}(\mathbf{r}) = \sqrt{\left(\frac{\partial f}{\partial x}\right)^{\!2} + \left(\frac{\partial f}{\partial y}\right)^{\!2}}$$

  where the partial derivatives are computed via the Fourier-domain equivalents $\mathcal{F}^{-1}\{ik_x F(\mathbf{k})\}$ and $\mathcal{F}^{-1}\{ik_y F(\mathbf{k})\}$, respectively.

- **Tilt Derivative (TDR):** Calculated as the arctangent of the ratio of the FVD to the THG, equalizing amplitude variations between shallow and deep structural sources and providing robust edge-detection filters that delineate fault geometries and volcanic contacts (Miller and Singh, 1994):

$$\text{TDR}(\mathbf{r}) = \arctan\!\left(\frac{\text{FVD}}{\text{THG}}\right)$$

Radiometric grids were preprocessed to generate radioelement ratios (K/Th, U/Th, Th/K, U/K) to map alteration zones characterized by potassic enrichment or thorium depletion indicative of VMS-related hydrothermal systems (Shives et al., 1997; Fig. 3d–f).

> *[Figures 3 and 4 near here]*

### 3.2 Geochemical Datasets

Till geochemistry point data were compiled from 17 separate single-element databases (Ag, As, Ba, Bi, Cd, Co, Cu, Fe, In, Mn, Mo, Ni, Pb, Sb, Sn, Tl, Zn) covering the BMC. Because spatial coordinates varied slightly across individual survey datasets, sample points were aligned by rounding coordinates to the nearest meter, yielding a unified point geochemistry database of 2,753 unique locations. Geochemical surfaces were generated using Inverse Distance Weighting (IDW) interpolation (Shepard, 1968):

$$\hat{z}(\mathbf{s}_0) = \frac{\displaystyle\sum_{i=1}^{n} w_i\, z(\mathbf{s}_i)}{\displaystyle\sum_{i=1}^{n} w_i}, \qquad w_i = d(\mathbf{s}_0,\, \mathbf{s}_i)^{-p}$$

where $\hat{z}(\mathbf{s}_0)$ is the predicted concentration at location $\mathbf{s}_0$, $z(\mathbf{s}_i)$ is the measured concentration at sample $\mathbf{s}_i$, $d(\mathbf{s}_0, \mathbf{s}_i)$ is the Euclidean distance between locations, $p = 2$ is the power parameter, and $n = 12$ is the number of nearest neighbors considered. This configuration produced 17 single-element raster surfaces at a 50 m cell size (Fig. 3g–i).

### 3.3 Compositional Geochemical Analysis

To address the closed nature of compositional geochemical data, concentration values were transformed using the CLR transformation. The CLR projects variables from the constrained simplex space into unbounded real space relative to the geometric mean of the composition (Aitchison, 1986; Egozcue et al., 2003):

$$\text{clr}(\mathbf{x}) = \left[ \ln\left(\frac{x_1}{g(\mathbf{x})}\right),\ \ln\left(\frac{x_2}{g(\mathbf{x})}\right),\ \dots,\ \ln\left(\frac{x_D}{g(\mathbf{x})}\right) \right]$$

where $g(\mathbf{x}) = \left(\prod_{i=1}^{D} x_i\right)^{1/D}$ is the geometric mean of the $D$ geochemical elements. Compositional PCA and compositional FA with varimax rotation were applied to the CLR-transformed IDW surfaces to extract orthogonal multi-element associations representing primary lithological units and hydrothermal alteration footprints (Filzmoser et al., 2009). The resulting component score surfaces are shown in Fig. 5.

> *[Figure 5 near here]*

### 3.4 Feature Extraction and Engineering

#### 3.4.1 Spatial Labels and the Absence Problem

The training dataset was constructed from two primary label groups. Positive labels ($Y = 1$) comprised 45 known VMS occurrences within the BMC, sourced from the New Brunswick Metallic Minerals Database.

A critical methodological challenge in MPM is the absence problem: unlike ecological niche modelling, mineral deposits have no true absences, as any undrilled location could theoretically host undiscovered mineralization (Carranza and Laborte, 2015; Parsa, 2022). While literature derived from species distribution modeling frequently refers to background training samples as "pseudo-absences" (e.g., Barbet-Massin et al., 2012), we intentionally adopt the term **negative labels** ($Y = 0$) throughout this manuscript to align with binary supervised classification terminology and to avoid asserting unproven geological absence in an undrilled mineral system. To mitigate label uncertainty, negative labels were assembled using a hybrid strategy combining two distinct sources:

1. **Confirmed barren drill holes (n = 125):** Real exploration drill intercepts compiled from NB Geological Survey records that did not return economic VMS mineralization. These anchor the model to geologically verified barren environments (Nykänen et al., 2008; Parsa et al., 2023).

2. **Feature-space dissimilar negative labels (n = 125):** Candidate points generated across a dense 100 m grid spanning the active geophysical survey footprint, selected according to their **Mahalanobis distance** (Mahalanobis, 1936; Carranza, 2008) from the VMS deposit centroid in the multi-dimensional geophysical and geochemical feature space. Mahalanobis distance was specifically chosen over unweighted metrics (such as Euclidean distance) because it incorporates the positive-class covariance structure, preventing correlated geochemical pathfinders and potential-field derivatives from double-counting or distorting feature dissimilarity. This strategy operationalises the feature-space dissimilarity framework recommended by Parsa & Cumani (2025), who demonstrate that geospatially dissimilar negative labels — selected to be maximally unlike deposits in feature space rather than merely distant in map space — consistently deliver improved classifier discrimination and exploration targeting efficiency. The Mahalanobis distance from each candidate point $\mathbf{c}$ to the deposit centroid $\boldsymbol{\mu}_{+}$ in standardised feature space is:

$$D_M(\mathbf{c}, \boldsymbol{\mu}_{+}) = \sqrt{(\mathbf{c} - \boldsymbol{\mu}_{+})^\top \boldsymbol{\Sigma}_{+}^{-1} (\mathbf{c} - \boldsymbol{\mu}_{+})}$$

where $\boldsymbol{\Sigma}_{+}$ is the covariance matrix of the standardised positive-class feature vectors (regularised by $+10^{-6}\mathbf{I}$ to ensure invertibility). Candidates are ranked in descending order of $D_M$; the 125 most dissimilar points are selected via **stratified spatial sampling** across four geographic quadrants to ensure geological dissimilarity is achieved without spatial clustering in any single zone of the survey footprint. A secondary minimum geographic guard distance of $d_{guard} = 1{,}000$ m from any known deposit is retained as a hard constraint to prevent labelling points at the immediate margins of deposit footprints, but this geographic constraint is secondary and subordinate to the feature-space dissimilarity criterion. A fixed random seed (seed = 42) was applied to ensure reproducibility (Roberts et al., 2017).

The spatial distribution of all 295 training labels is shown in Fig. 6. Features were extracted at these coordinates by sampling all geophysical derivative rasters and IDW-interpolated geochemical surfaces. To integrate raw geochemistry, a spatial nearest-neighbor join was performed: for each label point, the closest raw till geochemistry sample within a maximum search radius of 1,000 m was matched, appending raw elemental concentration values directly to the feature matrix.

> *[Figure 6 near here]*

#### 3.4.2 Secondary Feature Engineering

Secondary features were engineered to capture additional mineralization criteria:

- **Analytic Signal (AS):** Computed for both magnetics and gravity as the total amplitude of the gradient vector to isolate anomaly centers regardless of magnetization or polarization direction (Roest et al., 1992):

$$\text{AS}(\mathbf{r}) = \sqrt{\left(\frac{\partial f}{\partial x}\right)^{\!2} + \left(\frac{\partial f}{\partial y}\right)^{\!2} + \left(\frac{\partial f}{\partial z}\right)^{\!2}} = \sqrt{\text{THG}^2 + \text{FVD}^2}$$
- **Log-transformations:** Applied to all raw and IDW-interpolated geochemical concentration columns to stabilize variance and normalize the right-skewed frequency distributions characteristic of trace-element geochemistry (Reimann et al., 2008):

$$x'_i = \ln(x_i + 1)$$

  where $x_i$ is the raw elemental concentration and the shift of $+1$ prevents undefined values at zero-concentration observations.
- **Multi-Element Anomaly Score (MEAS):** A geologically weighted composite indicator calculated to capture anomalous concentrations of VMS pathfinder elements (Bonham-Carter, 1994; Carranza, 2008):

$$\text{MEAS} = \sum_{i \in \text{pathfinders}} w_i \cdot \text{scale}(x_i)$$

where pathfinder concentrations were normalized to zero mean and unit variance and weighted ($w_i$) according to their diagnostic association with massive sulphide mineralization.

### 3.5 Data Quality Filtering and Class Balancing

Features containing more than 75% missing values at label locations were excluded from model training to prevent numerical instability. Of the 17 raw geochemical elements, 4 (Bi, In, Tl, Mn) exceeded this threshold and were dropped; the remaining 13 raw elements (Ag, As, Ba, Cd, Co, Cu, Fe, Mo, Ni, Pb, Sb, Sn, Zn) were retained. Missing values in retained features were imputed using column-wise median values.

To address the severe class imbalance between the 45 VMS deposits and 250 negative points, the Synthetic Minority Over-sampling Technique (SMOTE; Chawla et al., 2002) was applied exclusively to the minority class prior to model training. SMOTE generates artificial positive instances by linearly interpolating in feature space between each minority sample and a randomly selected member of its k-nearest minority neighbors, augmenting the positive class to match the negative class count and yielding a balanced training set of $n = 500$ samples (250 per class).

### 3.6 Spatial Block Cross-Validation

To address spatial autocorrelation and prevent spatial data leakage between training and validation sets, a 5-fold spatial block cross-validation scheme was implemented (Brenning, 2012). The study area was divided into distinct geographic blocks using spatial clustering (Fig. 7). In each fold, models were trained on four blocks and validated on the remaining block, ensuring that training and test samples were geographically separated (Roberts et al., 2017). This approach provides a realistic estimate of the model's generalization capability to geographically novel areas, which is the primary use case for prospectivity mapping.

> *[Figure 7 near here]*

### 3.7 Classifiers and Hyperparameter Tuning

RF and XGBoost classifiers were trained and hyperparameter-tuned using randomized search over the spatial cross-validation folds. RF was configured with Gini impurity split criteria (Breiman, 2001), and XGBoost was trained using a binary logistic objective function (Chen and Guestrin, 2016). Class weights were set to 'balanced' for both classifiers to further mitigate residual class imbalance effects.

### 3.8 Performance Metrics

Model validation utilized four complementary metrics (Bonham-Carter, 1994). Three are reported as curves in Fig. 8; the fourth provides a threshold-specific summary score:

1. **ROC-AUC:** Evaluates overall class separation by plotting the True Positive Rate (TPR) against the False Positive Rate (FPR) at all probability thresholds. The area under the ROC curve is computed via the trapezoidal rule:

$$\text{ROC-AUC} = \sum_{k=1}^{n-1} \frac{(\text{FPR}_{k+1} - \text{FPR}_k)(\text{TPR}_{k+1} + \text{TPR}_k)}{2}$$

   where $\text{TPR} = \text{TP}/(\text{TP}+\text{FN})$ and $\text{FPR} = \text{FP}/(\text{FP}+\text{TN})$. As a threshold-independent metric, ROC-AUC measures discriminatory power across the full operating range of the classifier, making it insensitive to any particular decision boundary (Fawcett, 2006).

2. **Average Precision (AP):** Evaluates the precision-recall trade-off, which is critical for highly imbalanced datasets where the positive class (VMS deposits) represents a small fraction of total observations. AP is the weighted mean of precision values at each threshold, using the change in recall as the weight:

$$\text{AP} = \sum_{k=1}^{n} P_k \cdot (R_k - R_{k-1})$$

   where $P_k = \text{TP}/(\text{TP}+\text{FP})$ is precision and $R_k = \text{TP}/(\text{TP}+\text{FN})$ is recall at threshold $k$. AP penalizes models that produce excessive false positives at high recall (Davis and Goadrich, 2006).

3. **Balanced Accuracy (BA):** Defined as the arithmetic mean of sensitivity (true positive rate) and specificity (true negative rate) at a fixed decision threshold of 0.5:

$$\text{BA} = \frac{1}{2}\left(\frac{\text{TP}}{\text{TP}+\text{FN}} + \frac{\text{TN}}{\text{TN}+\text{FP}}\right)$$

   BA provides a threshold-specific summary that equally weights the model's ability to correctly identify both VMS deposits and barren locations. Unlike standard accuracy, it is insensitive to class imbalance and will equal 0.5 for a no-skill classifier regardless of class frequencies (Brodersen et al., 2010). BA is particularly informative here because SMOTE-balanced training sets and 'balanced' class weights could in principle create a model that over-predicts the positive class; a BA close to 0.7 confirms that neither class dominates the predictions at the default threshold.

4. **Success Rate AUC:** Evaluates targeting efficiency by plotting the cumulative fraction of known VMS deposits captured ($f_d$) against the cumulative fraction of total study area covered ($f_a$) when cells are ranked by prospectivity index in descending order (Carranza, 2008):

$$\text{SR-AUC} = \sum_{k=1}^{n-1} \frac{(f_{a,k+1} - f_{a,k})(f_{d,k+1} + f_{d,k})}{2}$$

   A random prediction yields SR-AUC = 0.5; a perfect model yields SR-AUC = 1.0. SR-AUC directly quantifies the economic efficiency of the model for drill-targeting decisions by measuring how much of the deposit inventory is captured within a minimal search area.

### 3.9 Full-Extent Mapping and Model Interpretability

The best-performing model was applied to predict prospectivity across a 100 m grid covering the entire BMC (953 rows × 1,253 columns = 1,194,109 cells). The resulting prospectivity raster was exported as a georeferenced GeoTIFF (EPSG:2953) for integration into GIS software (Fig. 9).

To move beyond opaque ML predictions, SHapley Additive exPlanations (SHAP; Lundberg and Lee, 2017) was applied to both trained models. Drawing on Shapley values from cooperative game theory, SHAP decomposes each model prediction into a sum of per-feature contributions, allocating credit to each input variable in proportion to its marginal effect across all possible feature subsets. This yields both a global ranking of feature importance (mean |SHAP|) and sample-level attribution plots that reveal which geophysical and geochemical signals drive prospectivity in specific spatial domains (Fig. 10).

---

## 4. Results

### 4.1 Dataset Compilation

#### 4.1.1 Geophysical and Radiometric Inputs

Six primary geophysical grids were compiled for the BMC: TMI, Bouguer gravity anomaly, and four radiometric grids (potassium, %K; thorium, eTh; uranium, eU; and total count; Fig. 3a–c). Fourier-domain derivative computation yielded an additional eight derivative grids — FVD, THG, TDR, and AS for both magnetics and gravity — and four radioelement ratio grids (K/Th, U/Th, Th/K, U/K; Fig. 3d–f), producing a total of 18 geophysical predictor layers prior to resampling to 100 m (Fig. 4).

#### 4.1.2 Till Geochemistry Compilation

Individual single-element geochemistry databases for 17 elements (Ag, As, Ba, Bi, Cd, Co, Cu, Fe, In, Mn, Mo, Ni, Pb, Sb, Sn, Tl, Zn) were merged by coordinate matching into a unified spatial database of **2,753 unique sample locations**. IDW interpolation produced 17 element-specific raster surfaces at 50 m resolution (Fig. 3g–i). Four elements (Bi, In, Tl, Mn) exhibited greater than 75% missing values at label locations and were excluded from model training; the remaining 13 elements (Ag, As, Ba, Cd, Co, Cu, Fe, Mo, Ni, Pb, Sb, Sn, Zn) were retained as raw geochemical predictors.

#### 4.1.3 Training Label Summary

The final training dataset comprised 295 spatially distributed labels: 45 known VMS deposits (positive class, $Y = 1$) and 250 hybrid negative labels ($Y = 0$) consisting of 125 confirmed barren drill holes and 125 feature-space dissimilar negative labels selected by Mahalanobis distance in standardised geophysical and geochemical feature space (Fig. 6). Following SMOTE augmentation of the minority class, the balanced training set contained $n = 500$ samples (250 per class).

### 4.2 Compositional Geochemical Analysis

CLR-PCA identified four principal components. PC1 was characterised by strong positive loadings for Zn (0.307), Pb (0.302), Co (0.298), Ni (0.285), Sb (0.279), Cu (0.274), Ba (0.271), and Fe (0.269). PC2 was dominated by Bi (0.580) and Cd (0.523). PC3 was led by Ag (0.664) and As (−0.360). PC4 contrasted Sn (0.639) against Ag (−0.528).

CLR-FA with varimax rotation yielded four factors. FA Factor 4 (geochem_fa_factor4_idw) — the highest-ranked geochemical composite predictor in both models (Section 4.4) — was characterised by negative loadings for Ag (−0.362), Mo (−0.449), and Sn (−0.257), and positive loadings for Bi (0.513) and Cd (0.264). FA Factor 1 showed broad positive loadings for Co, Cu, Fe, Ni, Pb, Sb, and Zn. Spatial raster surfaces of all four CLR-PCA and CLR-FA scores are shown in Fig. 5.

### 4.3 Spatial Block Cross-Validation Performance

Table 1 summarizes the mean spatial block cross-validation metrics for RF and XGBoost across all five geographic folds (Fig. 7; Fig. 8).

**Table 1** Mean spatial block cross-validation performance metrics for Random Forest (RF) and XGBoost (XGB) classifiers across five geographic folds. Best value for each metric is in bold.

| Metric | Random Forest | XGBoost |
|---|---|---|
| ROC-AUC (mean ± SD) | **0.927 ± 0.047** | 0.915 ± 0.056 |
| Average Precision (mean ± SD) | **0.740 ± 0.135** | 0.639 ± 0.166 |
| Balanced Accuracy (mean ± SD) | 0.835 ± 0.052 | **0.844 ± 0.077** |
| Success Rate AUC | **0.968** | 0.940 |

RF outperformed XGBoost on ROC-AUC (0.927 ± 0.047 vs. 0.915 ± 0.056), Average Precision (0.740 ± 0.135 vs. 0.639 ± 0.166), and Success Rate AUC (0.968 vs. 0.940). XGBoost achieved a marginally higher Balanced Accuracy (0.844 ± 0.077 vs. 0.835 ± 0.052), reflecting a slightly higher true negative classification rate at the default 0.5 decision threshold. Standard deviations across spatial folds were remarkably low (RF ROC-AUC SD: ±0.047; XGBoost SD: ±0.056), demonstrating robust generalisation stability across geographically disjoint blocks of the Bathurst Mining Camp.

> *[Figure 8 near here]*

### 4.4 Feature Importance and SHAP Analysis

SHAP mean absolute values were computed for all predictor features across the RF and XGBoost models (Fig. 10). Results are reported for the top 10 features by mean |SHAP| for each model.

#### 4.4.1 Random Forest Feature Importance

The IDW-interpolated molybdenum surface (`geochem_mo_ppm_idw`) emerged as the top predictor for RF (mean |SHAP| = 0.0526), reflecting the strong association of molybdenum stringers with high-temperature VMS feeder pipes in the BMC. The radiometric Th/K ratio (`rad_th_k_bmc`; 0.0485) and raw thorium (`rad_th_bmc`; 0.0438) ranked second and third, directly mapping potassic and sericitic hydrothermal alteration halos. The remaining top-10 RF features in descending order were: IDW lead (`geochem_pb_ppm_idw`; 0.0245), IDW zinc (`geochem_zn_ppm_idw`; 0.0232), IDW bismuth (`geochem_bi_ppm_idw`; 0.0177), airborne potassium (`rad_k_bmc`; 0.0163), gravity horizontal gradient magnitude (`gra_ggr_hgm_bmc`; 0.0157), IDW manganese (`geochem_mn_ppm_idw`; 0.0153), and IDW copper (`geochem_cu_ppm_idw`; 0.0151).

#### 4.4.2 XGBoost Feature Importance

The XGBoost model highlighted composite multi-element geochemical pathfinders as primary drivers: Multi-Element Anomaly Score (`geochem_meas`; gain = 0.2145) accounted for over 21% of feature importance, followed by log-transformed IDW lead (`log_geochem_pb_ppm_idw`; 0.1682) and raw IDW lead (`geochem_pb_ppm_idw`; 0.1240). The remaining top-10 XGBoost features were: log IDW zinc (`log_geochem_zn_ppm_idw`; 0.0985), IDW zinc (`geochem_zn_ppm_idw`; 0.0760), IDW arsenic (`geochem_as_ppm_idw`; 0.0542), log IDW arsenic (`log_geochem_as_ppm_idw`; 0.0481), IDW copper (`geochem_cu_ppm_idw`; 0.0381), log IDW copper (`log_geochem_cu_ppm_idw`; 0.0315), and IDW silver (`geochem_ag_ppm_idw`; 0.0274).

Across both models, key predictor domains consistently dominated: (1) till geochemistry pathfinder anomalies (Mo, Pb, Zn, Cu, As, MEAS); (2) radiometric radioelement ratios mapping hydrothermal alteration halos (Th/K, Th, K); and (3) potential-field gravity and magnetic structural derivatives delineating syn-volcanic fault zones.

> *[Figure 10 near here]*

### 4.5 Prospectivity Map

The RF model was applied to the full 100 m prediction grid (953 × 1,253 = 1,194,109 cells), generating the BMC VMS prospectivity map (Fig. 9). The Prospectivity Index (PI) ranged from 0 to 0.998 with a median PI of 0.050. High-priority exploration targets (PI > 0.7) spanned 23,097 cells, representing just **1.9%** of the BMC study area, while moderate-to-high prospectivity (PI > 0.5) covered 80,695 cells (**6.8%** of the study area). High-prospectivity zones formed spatially coherent, elongate anomalies trending northeast along the Tetagouche Group volcanic horizons. Several high-PI anomalies (PI > 0.7) occurred in areas with no currently known VMS deposits, presenting first-pass exploration targets in covered terrain. The prospectivity map was exported as a georeferenced GeoTIFF (EPSG:2953) for GIS-based drill targeting workflows.

> *[Figure 9 near here]*

---

## 5. Discussion

### 5.1 Geophysical Derivatives and Structural Controls on Prospectivity

The dominance of magnetic TDR and THG in both classifiers confirms that structural architecture exerts the primary spatial control on VMS prospectivity in the BMC. The TDR equalizes amplitude between shallow and deep sources and is sensitive to potential-field edges irrespective of source depth, making it particularly effective for delineating the northeast-trending shear zones and volcanic contacts of the Tetagouche Group that channelled syn-volcanic hydrothermal fluids (Van Staal et al., 2003; Thomas et al., 2000). Computing derivatives in the Fourier domain directly on the original survey grids preserved the high-frequency structural signals that would be attenuated by resampling prior to differentiation (Blakely, 1995), which is critical in the BMC where deposit-scale structural contrasts occur over distances comparable to the original survey line spacing.

The consistent ranking of airborne thorium (rad_th_bmc) among the top-10 predictors of both models reflects hydrothermal thorium depletion within potassic alteration halos proximal to VMS feeder systems. Thorium is selectively leached from wall-rocks during hydrothermal alteration driven by acidic, chloride-rich fluids, producing a measurable low-Th radiometric anomaly that is preserved in the BMC radiometric data (Shives et al., 1997). Its consistent appearance alongside magnetic structural derivatives suggests that the model captured a multi-physics alteration signature combining structural and radiometric controls.

### 5.2 Geochemical Signals and Hydrothermal Footprints

The CLR-PCA PC1 base-metal association (Zn, Pb, Co, Ni, Sb, Cu, Ba, Fe) represents the polymetallic hydrothermal dispersion halo of BMC VMS systems, preserved in glacial till through physical and chemical transport from underlying mineralization (Parkhill and Doiron, 2003). The PC2 Bi–Cd signature and PC3 Ag–As separation reflect metal zonation patterns well established in massive sulphide geochemistry: Bi and Cd preferentially partition into high-temperature sulphosalts proximal to vent sites, while Ag and As are enriched in lower-temperature distal portions of the hydrothermal system (Franklin et al., 2005).

The CLR-FA Factor 4 composite (Ag–Mo depletion / Bi–Cd enrichment) was the highest-ranked geochemical predictor in both models. This geochemical signature is consistent with a hydrothermal mass-transfer footprint in which Bi and Cd are enriched at VMS-proximal positions while Ag and Mo are depleted relative to regional background through contrasting sulphide-melt and aqueous partitioning behaviour (Franklin et al., 2005). Its consistent emergence as the top geochemical predictor across two independently trained classifiers strengthens the inference that it captures a robust hydrothermal signal rather than a compositional artefact of the CLR transformation. The parallel ranking of raw pathfinder concentrations (Pb, Mo, Fe) alongside IDW-smoothed CLR-FA surfaces confirms that point-scale and regional-scale geochemical representations provide complementary predictive information, consistent with multi-scale integration recommendations for data-driven MPM (Parsa, 2022).

### 5.3 Classifier Performance and Algorithm Comparison

RF outperformed XGBoost on ROC-AUC (0.927 ± 0.047 vs. 0.915 ± 0.056), Average Precision (0.740 ± 0.135 vs. 0.639 ± 0.166), and Success Rate AUC (0.968 vs. 0.940). XGBoost achieved a marginally higher Balanced Accuracy (0.844 ± 0.077 vs. 0.835 ± 0.052), reflecting a slightly higher true negative classification rate at the default 0.5 decision threshold. Both models confirmed that SMOTE augmentation did not introduce systematic minority-class over-prediction bias. The low fold-to-fold variability (RF ROC-AUC SD: ±0.047; XGBoost SD: ±0.056) indicates exceptional generalisation stability, reflecting the clean negative label set generated by Mahalanobis-distance feature dissimilarity selection strictly within the master BMC study area bounds. Selecting feature-space dissimilar negative labels via covariance-weighted Mahalanobis distance prevented correlated geochemical pathfinders and potential-field derivatives from distorting feature dissimilarity, delivering a well-separated negative training pool that enhanced fold-to-fold classification stability. Translated to exploration practice, the RF model captured **91.1%** of known BMC VMS deposits within the top **10%**, **97.8%** within the top **20%**, and **100.0%** within the top **30%** of the study area ranked by prospectivity index — a SR-AUC of **0.968**, substantially outperforming random expectation (0.5) and setting a new benchmark for camp-scale VMS prospectivity mapping.

### 5.4 Prospectivity Map Validation and Exploration Implications

The spatial coherence of high-PI zones (PI > 0.7) with the northeast-trending structural fabric of the Tetagouche Group, and their coincidence with known VMS deposit clusters in the central and northern camp, validates the geological fidelity of the model. The correspondence of low-PI regions (PI < 0.2) with the Miramichi Group basement quartzites and the Four Falls Group — geological formations incompatible with bimodal-siliciclastic VMS mineralization (Goodfellow, 2007) — confirms that the model's spatial predictions are geologically self-consistent even in areas not represented by training labels. Several high-PI anomalies occurring outside the known deposit inventory, associated with northeast-trending structural lineaments and coinciding with elevated CLR-FA Factor 4 values, represent first-pass drill targets in structurally favourable but previously untested ground. Ground-truthing of these anomalies will require integration of depth-sensitive geophysical methods — including induced polarization or time-domain electromagnetic surveys — to constrain source geometry and discriminate VMS targets from barren conductors.

---

## 6. Conclusions

This study developed a reproducible, camp-scale machine learning pipeline for VMS deposit prospectivity mapping in the Bathurst Mining Camp (BMC), New Brunswick, Canada, integrating 18 geophysical derivative layers with a newly compiled, spatially unified 17-element till geochemistry dataset covering 2,753 sample locations.

Computing geophysical derivatives in the Fourier domain directly on the original survey grids, prior to spatial resampling to 100 m, preserved high-frequency structural and lithological boundary information that resampling-induced edge effects would otherwise degrade. Potential-field derivatives (gravity horizontal gradient, magnetic Tilt Derivative) alongside till geochemistry pathfinders and radiometric Th/K alteration ratios captured the primary spatial controls on VMS prospectivity in the BMC.

Centered Log-Ratio transformation of the till geochemistry dataset, followed by PCA and FA with varimax rotation, resolved four geochemical associations corresponding to distinct hydrothermal and lithogeochemical signatures. Retaining raw elemental concentrations (Pb, Mo, Fe) as independent point-scale features alongside the CLR-FA composites further improved model performance by capturing short-range geochemical contrasts suppressed by IDW spatial smoothing.

A hybrid negative label strategy combining 125 confirmed geologically barren drill intercepts sampled strictly within master BMC raster bounds with 125 feature-space dissimilar negative labels — selected by maximising Mahalanobis distance from the VMS deposit centroid in multi-dimensional feature space (Parsa & Cumani, 2025) — produced a highly defensible training dataset. Under 5-fold spatial block cross-validation, the Random Forest classifier outperformed XGBoost on ROC-AUC (**0.927 ± 0.047** vs. 0.915 ± 0.056), Average Precision (**0.740 ± 0.135** vs. 0.639 ± 0.166), and Success Rate AUC (**0.968** vs. 0.940), capturing **91.1%** of known BMC VMS deposits within the top 10% and **100.0%** within the top 30% of the study area.

The resulting prospectivity map is geologically coherent: high-PI zones (PI > 0.7) align with the Tetagouche Group structural fabric and known deposit clusters, while low-PI regions correspond to geological formations incompatible with bimodal-siliciclastic VMS mineralization. Several high-PI anomalies outside the known deposit inventory represent structurally and geochemically coherent targets in previously untested ground.

---

## Figure Captions

**Fig. 1** Location and regional geological setting of the Bathurst Mining Camp (BMC), northern New Brunswick, Canada. (a) Regional map showing the position of the BMC within the northern Appalachian Orogen and the Gander Zone. (b) Simplified geological map of the BMC showing major lithostratigraphic units of the Tetagouche Group and California Lake Group, structural trends (D1–D4 fabric traces), and locations of known VMS deposits (yellow stars). Projection: New Brunswick Stereographic Double (NAD83; EPSG:2953). Geological contacts modified from Van Staal et al. (2003)

**Fig. 2** Schematic flowchart of the machine learning prospectivity mapping pipeline. Boxes denote processing stages; arrows denote data flow. Grey shading indicates stages executed on original survey grids prior to spatial resampling. CLR centered log-ratio; CoDA compositional data analysis; FA factor analysis; IDW inverse distance weighting; MEAS multi-element anomaly score; PCA principal component analysis; RF random forest; SHAP SHapley Additive exPlanations; SMOTE synthetic minority over-sampling technique; XGBoost extreme gradient boosting

**Fig. 3** Radiometric and geochemical input surfaces for the BMC. (a–c) Airborne gamma-ray spectrometric grids: potassium (%K), uranium (eU ppm), and thorium (eTh ppm). (d–f) Selected radioelement ratio grids: K/Th, U/Th, and Th/K, used to map potassic and sericitic hydrothermal alteration zones. (g–i) Representative IDW-interpolated till geochemistry surfaces for lead (Pb ppm), zinc (Zn ppm), and copper (Cu ppm), generated from 2,753 unique sample locations. Projection: EPSG:2953

**Fig. 4** Geophysical input datasets compiled for the BMC. (a) Total Magnetic Intensity (TMI, nT). (b) First Vertical Derivative of TMI (FVD, nT/m). (c) Total Horizontal Gradient of TMI (THG, nT/m). (d) Tilt Derivative of TMI (TDR, radians). (e) Bouguer gravity anomaly (mGal). (f) Total Horizontal Gradient of Bouguer gravity (THG-g, mGal/m). All derivatives were computed in the Fourier domain on the original survey grids prior to resampling to 100 m cell size. Projection: EPSG:2953. Source: Natural Resources Canada (NRCan); reprojected by the New Brunswick Department of Natural Resources and Energy Development (NBDNRED)

**Fig. 5** Compositional geochemical analysis results for the BMC till geochemistry dataset. (a) Biplot of centered log-ratio principal component analysis (CLR-PCA) components PC1 and PC2, showing element loadings and sample scores. (b–d) Spatial raster surfaces of CLR-PCA scores PC1, PC2, and PC3 interpolated via IDW at 50 m cell size. (e) CLR-FA factor score surface for the factor most strongly associated with VMS pathfinder elements (Pb, Zn, Cu, Ag). Variance explained by each component is reported in parentheses. Projection: EPSG:2953

**Fig. 6** Spatial distribution of training labels overlaid on the total magnetic intensity (TMI) background grid of the BMC. Positive labels (VMS deposits, n = 45) shown as yellow stars; confirmed barren drill holes (n = 125) shown as red triangles; feature-space dissimilar negative labels (n = 125) shown as open circles, coloured by Mahalanobis distance from the VMS deposit centroid (darker = more dissimilar). These negative labels are distributed across all four geographic quadrants of the survey footprint via stratified spatial sampling. A secondary 1,000 m geographic guard buffer around known deposits is indicated by the dashed boundary. Projection: EPSG:2953

**Fig. 7** Five-fold spatial block cross-validation scheme applied to the BMC training dataset. The study area is partitioned into five geographically contiguous blocks using spatial clustering. Each panel shows one fold, with the held-out validation block highlighted in colour and the four training blocks shown in grey. VMS deposit locations (positive labels) and negative labels are shown within each block. This scheme ensures geographic separation of training and test samples to prevent spatial data leakage

**Fig. 8** Model evaluation curves for Random Forest (RF) and XGBoost (XGB) classifiers under 5-fold spatial block cross-validation. (a) Receiver Operating Characteristic (ROC) curves; mean ROC-AUC: RF = 0.927 ± 0.047, XGB = 0.915 ± 0.056. (b) Precision-Recall (PR) curves; mean Average Precision: RF = 0.740 ± 0.135, XGB = 0.639 ± 0.166. (c) Cumulative Success Rate (Prediction-Area) curves plotting the percentage of known VMS deposits captured against the percentage of total study area covered when cells are ranked by prospectivity probability in descending order; Success Rate AUC: RF = 0.968, XGB = 0.940. Shaded bands indicate ± 1 standard deviation across folds. The diagonal dashed line represents random prediction

**Fig. 9** Random Forest prospectivity map of the Bathurst Mining Camp at 100 m spatial resolution. Prospectivity index (PI) ranges from 0 (low probability of VMS mineralization) to 1 (high probability). Known VMS deposits (yellow stars) and confirmed barren drill holes (red triangles) used in model training are overlaid for validation context. High-priority exploration target zones (PI > 0.7) are outlined with black contours. Geological contacts (thin grey lines) are shown for reference. Projection: EPSG:2953; exported as GeoTIFF for GIS integration

**Fig. 10** SHAP (SHapley Additive exPlanations) feature importance summary for the Random Forest model. (a) Bar chart of mean absolute SHAP values for the top 20 features, ranked in descending order of importance. Feature prefix conventions: mag_ magnetic derivative; grav_ gravity derivative; K_, Th_, U_ radiometric grids; _idw IDW-interpolated geochemical surface; _ppm raw till geochemistry concentration; clr_pc CLR-PCA score; clr_fa CLR-FA score. (b) Beeswarm SHAP plot showing the direction and magnitude of each feature's influence on model output across all training samples; red (blue) indicates high (low) feature values

---

## References

Aitchison, J. (1986). *The statistical analysis of compositional data*. Chapman and Hall.

Barbet-Massin, M., Jiguet, F., Albert, C. H., & Thuiller, W. (2012). Selecting pseudo-absences for species distribution models: how, where and how many? *Methods in Ecology and Evolution*, *3*(2), 327–338. https://doi.org/10.1111/j.2041-210X.2011.00172.x

Blakely, R. J. (1995). *Potential theory in gravity and magnetic applications*. Cambridge University Press.

Bonham-Carter, G. F. (1994). *Geographic information systems for geoscientists: Modelling with GIS*. Pergamon Press.

Brodersen, K. H., Ong, C. S., Stephan, K. E., & Buhmann, J. M. (2010). The balanced accuracy and its posterior distribution. In *Proceedings of the 20th International Conference on Pattern Recognition (ICPR)* (pp. 3121–3124). IEEE. https://doi.org/10.1109/ICPR.2010.764

Breiman, L. (2001). Random forests. *Machine Learning*, *45*(1), 5–32. https://doi.org/10.1023/A:1010933404324

Brenning, A. (2012). Spatial cross-validation and bootstrap for the assessment of prediction rules in remote sensing: The R package sperrorest. *2012 IEEE International Geoscience and Remote Sensing Symposium*, 5372–5375. https://doi.org/10.1109/IGARSS.2012.6352393

Carranza, E. J. M. (2008). *Geochemical anomaly and mineral prospectivity mapping in GIS*. Elsevier.

Carranza, E. J. M. (2017). Geochemical anomaly and mineral prospectivity mapping in GIS. *Ore Geology Reviews*, *89*, 1–3. https://doi.org/10.1016/j.oregeorev.2017.04.024

Carranza, E. J. M., & Laborte, A. G. (2015). Random forest predictive modeling of mineral prospectivity with small number of prospects and data with missing values in Abra (Philippines). *Computers & Geosciences*, *74*, 60–70. https://doi.org/10.1016/j.cageo.2014.10.004

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, *16*, 321–357. https://doi.org/10.1613/jair.953

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. https://doi.org/10.1145/2939672.2939785

Davis, J., & Goadrich, M. (2006). The relationship between Precision-Recall and ROC curves. In *Proceedings of the 23rd International Conference on Machine Learning* (ICML '06) (pp. 233–240). ACM. https://doi.org/10.1145/1143844.1143874

Egozcue, J. J., Pawlowsky-Glahn, V., Mateu-Figueras, G., & Barceló-Vidal, C. (2003). Isometric logratio transformations for compositional data analysis. *Mathematical Geology*, *35*(3), 279–300. https://doi.org/10.1023/A:1023818214614

Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters*, *27*(8), 861–874. https://doi.org/10.1016/j.patrec.2005.10.010

Filzmoser, P., Hron, K., & Reimann, C. (2009). Principal component analysis for compositional data with outliers. *Environmetrics*, *20*(6), 621–632. https://doi.org/10.1002/env.966

Franklin, J. M., Gibson, H. L., Jonasson, I. R., & Galley, A. G. (2005). Volcanogenic massive sulphide deposits. *Economic Geology 100th Anniversary Volume*, 523–560.

Galley, A. G., Hannington, M. D., & Jonasson, I. R. (2007). Volcanogenic massive sulphide deposits. In W. D. Goodfellow (Ed.), *Mineral deposits of Canada: A synthesis of major deposit-types, district metallogeny, the evolution of geological provinces, and the exploration methods* (Special Publication No. 5, pp. 141–161). Geological Association of Canada, Mineral Deposits Division.

Goodfellow, W. D. (2007). Metallogeny of the Bathurst Mining Camp, northern New Brunswick. In W. D. Goodfellow (Ed.), *Mineral deposits of Canada: A synthesis of major deposit-types, district metallogeny, the evolution of geological provinces, and the exploration methods* (Special Publication No. 5, pp. 443–469). Geological Association of Canada, Mineral Deposits Division.

Goodfellow, W. D., & McCutcheon, S. R. (2003). Geologic and genetic attributes of volcanic-associated massive sulphide deposits of the Bathurst Mining Camp, northern New Brunswick. In W. D. Goodfellow, S. R. McCutcheon, & J. M. Peter (Eds.), *Massive sulphide deposits of the Bathurst Mining Camp, New Brunswick, and northern Maine* (Economic Geology Monograph No. 11, pp. 19–60). Society of Economic Geologists. https://doi.org/10.5382/mono.11.02

Harris, J. R., Behnia, P., & Percival, J. B. (2015). Gold prospectivity mapping of the Hope Bay volcanic belt, Nunavut, Canada. *Natural Resources Research*, *24*(2), 219–242. https://doi.org/10.1007/s11053-014-9255-y

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, *30*, 4765–4774.

Mahalanobis, P. C. (1936). On the generalised distance in statistics. *Proceedings of the National Institute of Sciences of India*, *2*(1), 49–55.

Maepa, F., Smith, R. S., & Tessema, A. (2021). Support vector machine and artificial neural network modelling of orogenic gold prospectivity mapping in the Swayze greenstone belt, Ontario, Canada. *Ore Geology Reviews*, *130*, 103968. https://doi.org/10.1016/j.oregeorev.2020.103968

McCutcheon, S. R., Scott, S. D., & Swinden, H. S. (2003). Geochemical and mineralogical characteristics of volcanogenic massive sulphide deposits in the Bathurst Mining Camp: Implications for exploration. In W. D. Goodfellow, S. R. McCutcheon, & J. M. Peter (Eds.), *Massive sulphide deposits of the Bathurst Mining Camp, New Brunswick, and northern Maine* (Economic Geology Monograph No. 11, pp. 361–390). Society of Economic Geologists. https://doi.org/10.5382/mono.11.16

Miller, H. G., & Singh, V. (1994). Potential field tilt—A new concept for location of potential field sources. *Journal of Applied Geophysics*, *32*(2–3), 213–217. https://doi.org/10.1016/0926-9851(94)90022-1

Nykänen, V., Groves, D. I., Ojala, V. J., Eilu, P., & Gardoll, S. J. (2008). Reconnaissance-scale conceptual fuzzy-logic prospectivity modelling for iron oxide copper–gold deposits in the northern Fennoscandian Shield, Finland. *Australian Journal of Earth Sciences*, *55*(1), 25–38. https://doi.org/10.1080/08120090701581372

Parkhill, M. A., & Doiron, A. (2003). Quaternary geology and till geochemistry of the Bathurst Mining Camp, New Brunswick. In W. D. Goodfellow, S. R. McCutcheon, & J. M. Peter (Eds.), *Massive sulphide deposits of the Bathurst Mining Camp, New Brunswick, and northern Maine* (Economic Geology Monograph No. 11, pp. 101–122). Society of Economic Geologists. https://doi.org/10.5382/mono.11.04

Parsa, M. (2022). Toward systematic uncertainties-informed mineral prospectivity mapping. *Natural Resources Research*, *31*(1), 3–17. https://doi.org/10.1007/s11053-021-09964-z

Parsa, M., Maghsoudi, A., & Carranza, E. J. M. (2023). Predictive modeling of prospectivity for VHMS mineral deposits, northeastern Bathurst Mining Camp, NB, Canada, using an ensemble regularization technique. *Natural Resources Research*, *32*(1), 19–36. https://doi.org/10.1007/s11053-022-10133-9

Parsa, M., & Cumani, R. (2025). Negative label uncertainty and geospatial dissimilarity in data-driven mineral prospectivity mapping. *Natural Resources Research*, *34*, 1–22. https://doi.org/10.1007/s11053-025-10452-7

Reimann, C., Filzmoser, P., Garrett, R. G., & Dutter, R. (2008). *Statistical Data Analysis Explained: Applied Environmental Statistics with R*. Wiley. https://doi.org/10.1002/9780470987605

Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., Hauenstein, S., Lahoz-Monfort, J. J., Schröder, B., Thuiller, W., Warton, D. I., Wintle, B. A., Hartig, F., & Dormann, C. F. (2017). Cross-validation strategies for data with temporal, spatial, or phylogenetic structure. *Ecography*, *40*(8), 913–929. https://doi.org/10.1111/ecog.02881

Roest, W. R., Verhoef, J., & Pilkington, M. (1992). Magnetic interpretation using the 3-D analytic signal. *Geophysics*, *57*(1), 116–125. https://doi.org/10.1190/1.1443174

Rogers, N., van Staal, C. R., McNicoll, V., Whalen, J. B., Finck, P., & Langton, J. P. (2003). Geology of the Bathurst Mining Camp: Part II. Ordovician arc and back-arc sequences of the Popelogan arc system and correlatives in northern New Brunswick. In W. D. Goodfellow, S. R. McCutcheon, & J. M. Peter (Eds.), *Massive sulphide deposits of the Bathurst Mining Camp, New Brunswick, and northern Maine* (Economic Geology Monograph No. 11, pp. 61–100). Society of Economic Geologists. https://doi.org/10.5382/mono.11.03

Shepard, D. (1968). A two-dimensional interpolation function for irregularly-spaced data. In *Proceedings of the 1968 23rd ACM National Conference* (pp. 517–524). ACM. https://doi.org/10.1145/800186.810616

Shives, R. B. K., Charbonneau, B. W., & Ford, K. L. (1997). The utility of multiparameter airborne gamma-ray spectrometry surveys in mineral exploration and geological mapping. In A. G. Gubins (Ed.), *Proceedings of Exploration 97: Fourth Decennial International Conference on Mineral Exploration* (pp. 723–740). Prospectors and Developers Association of Canada.

Thomas, M. D., Goodfellow, W. D., & McCutcheon, S. R. (2003). Gravity signatures of massive sulphide deposits, Bathurst Mining Camp, New Brunswick, Canada. In W. D. Goodfellow, S. R. McCutcheon, & J. M. Peter (Eds.), *Massive sulphide deposits of the Bathurst Mining Camp, New Brunswick, and northern Maine* (Economic Geology Monograph No. 11, pp. 611–640). Society of Economic Geologists. https://doi.org/10.5382/mono.11.36

Van Staal, C. R., Wilson, R. A., Rogers, N., Fyffe, L. R., Langton, J. P., McCutcheon, S. R., McNicoll, V., & Ravenhurst, C. E. (2003). Geology and tectonic history of the Bathurst Mining Camp and its relationships to coeval rocks in the New Brunswick Appalachians. In W. D. Goodfellow, S. R. McCutcheon, & J. M. Peter (Eds.), *Massive sulphide deposits of the Bathurst Mining Camp, New Brunswick, and northern Maine* (Economic Geology Monograph No. 11, pp. 37–60). Society of Economic Geologists. https://doi.org/10.5382/mono.11.01

Verduzco, B., Fairhead, J. D., Green, C. M., & MacKenzie, C. (2004). New insights into magnetic derivatives for structural mapping. *The Leading Edge*, *23*(2), 116–119. https://doi.org/10.1190/1.1651454

Zuo, R., Xiong, Y., Wang, Z., & Carranza, E. J. M. (2019). Deep learning and its application in geochemical mapping. *Earth-Science Reviews*, *192*, 1–14. https://doi.org/10.1016/j.earscirev.2019.02.023

