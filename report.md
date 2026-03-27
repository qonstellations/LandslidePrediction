# Replication Report: Multi-Modal Geospatial Mixture-of-Experts for Landslide Susceptibility Modeling

## 1. Methodology

Our pipeline formulates landslide prediction as a binary spatial classification task, fusing heterogeneous multi-modal data into a unified, normalized spatial tensor $X \in \mathbb{R}^{C \times H \times W}$.

### A. Data Fusion & Representation Learning
We aggregate $C=8$ distinct geospatial channels:
1.  **Topological:** Copernicus DEM ($30\text{m}$)
2.  **Structural:** Sentinel-1 SAR (Synthetic Aperture Radar)
3.  **Spectral:** Sentinel-2 Optical RGB (B02, B03, B04)
4.  **Meteorological:** Short-term (7-day) and continuous long-term (yearly) rainfall sequences
5.  **Hydrological:** Aggregated Soil Moisture indices

To reconcile spatial resolution differentials, all modalities are spatially re-sampled to adhere to the standard Sentinel-2 geographic grid via nearest-neighbor interpolation. To bound gradient variance during optimization, feature-wise Min-Max normalization is applied independently per channel, mapping all values to $x_c \in [0, 1]$. The final unified tensor adopts dimensions $X \in \mathbb{R}^{8 \times 200 \times 581}$.

### B. Patch Extraction Pipeline
Given the spatial nature of the input tensor, we extract localized contexts by implementing a 2D sliding window of dimensions $P \times P$ (where $P=64$). This operation yields a sequence of input sub-grids $x_i \in \mathbb{R}^{8 \times 64 \times 64}$.
The ground truth generation function $M(x_i)$ applies a strict boolean heuristic: if the corresponding target mask region contains any positive landslide pixel, the entire patch is assigned a positive label ($y_i = 1$), otherwise negative ($y_i = 0$).

### C. Architectural Formulations

We benchmark a baseline Convolutional Neural Network (CNN) against an advanced Mixture-of-Experts (MoE) configuration.

**1. Baseline CNN Feature Extractor:**
The CNN acts as our primary encoder, evaluating mapping $f_{\theta}: \mathbb{R}^{C \times P \times P} \mapsto \mathbb{R}^{D}$, where $D=128$. It consists of dual convolutional blocks (`Conv2d -> ReLU -> MaxPool2d`), projecting the localized $8$-channel spatial features into a dense vector space prior to linear classification.

**2. Mixture-of-Experts (MoE) Subsystem:**
The MoE formulation bypasses the standard dense classification head, increasing representational capacity without scaling feature extraction costs. Let $v_i = f_{\theta}(x_i) \in \mathbb{R}^{128}$ act as the encoded feature representation.
- **Experts ($E_k$):** We configure $K=2$ fundamentally asymmetric Multi-Layer Perceptrons (MLPs). $E_1$ acts as a shallow discriminator, while $E_2$ is parameterized with an additional hidden layer for higher-order nonlinear mapping. Both project $v_i \mapsto \mathbb{R}^1$ (raw logits).
- **Gating Network (Router) $R_{\phi}$:** A linear projection generates routing logits $g_i = R_{\phi}(v_i) \in \mathbb{R}^K$. During training, Gaussian noise is additively injected to promote exploration and prevent early routing collapse. The normalized assignment probabilities $w_i \in \Delta^{K-1}$ are parameterized via Softmax calculation:
  $w_i = \text{Softmax}(g_i + \epsilon \cdot \mathcal{N}(0, I))$ where $\epsilon = 0.01$.
- **Final Aggregation:** The combined output logit $\hat{z}_i$ is the weighted expectation of the expert predictions: $\hat{z}_i = \sum_{k=1}^K w_{i,k} E_k(v_i)$.


## 2. Experimental Setup & Optimization Scheme

1.  **Optimization Function:** Architectures are optimized over 20 epochs using the Adam optimizer ($\alpha = 10^{-3}$).
2.  **Imbalance Mitigation:** Landslide events manifest as severe minority class anomalies. We counteract dataset skew by calculating and introducing a class-frequency weight $\mathcal{W}_{pos}$ directly into the formulation of the binary objective. We optimize the weighted Binary Cross-Entropy with Logits loss:
    $\mathcal{L}_{BCE} = -[y_i \cdot \mathcal{W}_{pos} \cdot \log(\sigma(\hat{z}_i)) + (1 - y_i) \cdot \log(1 - \sigma(\hat{z}_i))]$
3.  **Stride Parameterization:** We execute two experimental configurations altering the patch overlapping density: Stride 32 ($50\%$ overlap) serving as the baseline sampling density, and Stride 8 ($87.5\%$ overlap) functioning as dynamic translational data augmentation.
4.  **Zero-Shot Transfer:** Post-convergence on the Puthumala primary dataset, out-of-distribution (OOD) systemic validity is measured by directly scoring the Wayanad region without fine-tuning bounds.

## 3. Empirical Results

The MoE architecture systematically eclipses the CNN baseline bounds, empirically validating that dynamically localized routing is significantly more resilient for heterogeneous geospatial topologies.

| Parametrization | CNN Baseline Accuracy | MoE Architecture Accuracy |
| :--- | :--- | :--- |
| **Stride 32 (Low Data Density)** | $\sim 89.0\%$ | $\mathbf{\sim 94.0\%}$ |
| **Stride 8 (High Data Density)** | $\sim 97.7\%$ | $\mathbf{\sim 98.7\%}$ |

### Router Degeneracy vs. Distributed Load
By systematically observing the expected routing behavior $\mathbb{E}[w_i]$, we captured shifting behavioral paradigms intrinsic to MoE networks:
- **Low Data Support Regime (Stride 32):** The gating network exhibited rapid representation collapse. A singular expert monopolized predictive responsibilities ($\sim 100\%$ routing weight), indicating that sparse distributions fail to penalize degenerate decision boundaries between experts.
- **High Data Support Regime (Stride 8):** The introduction of dense translational redundancy forced representation disentanglement. The router stabilized into an asynchronous steady state, partitioning the input space effectively ($\sim 65\%$ to $E_1$, $\sim 35\%$ to $E_2$), successfully exploiting the structural variance across both sub-network branches.

## 4. Analytical Observations

- **Unsupervised Semantic Branching:** We do not constrain experts via explicit inductive biases or semantic clustering (e.g., arbitrarily forcing $E_1$ to model elevation and $E_2$ to model rainfall coefficients). Instead, mathematical specialization emerges latently through backpropagation; driven strictly by the router's loss gradients matching divergent multi-modal input topologies to the mathematically optimal expert surface conditionally.
- **Translational Extrapolation:** The significant performance delta observed when transitioning from Stride 32 $\rightarrow$ Stride 8 fundamentally confirms that dense windowing successfully operates as geometric data augmentation. The encoder $f_{\theta}$ learns substantially stronger spatial and translational invariance bounds, decisively suppressing overfitting against localized localized patch boundaries.

## 5. Limitations & Trajectories

1.  **Data Leakage Risk via Spatial Autocorrelation:** The overlapping generation logic ($\text{Stride} < P$) guarantees structural autocorrelation between adjacent inputs, intrinsically implicitly inflating validation metrics if testing and training sets intersect regionally. **Avenue:** Subsequent frameworks must be validated utilizing strict spatial blocking regimes (e.g., spatial k-fold cross-validation).
2.  **Harmonic Degradation of the Topological Tensor:** Resolving all matrices explicitly downward to the Sentinel-2 sampling rate actively destroys critical high-frequency, granular spatial derivatives latent in the original Copernicus DEM mapping. **Avenue:** Bypassing explicit early-stage fusion by migrating to a late-fusion hierarchical transformer or multi-scale feature pyramid network (FPN) geometry, accommodating native resolution parsing.
3.  **Static Temporal Approximations:** Methodically treating cumulative and sparse rainfall events explicitly as static, integrated 2-dimensional spatial grids actively compresses the inherent Markovian properties governing real-world saturated soil dynamics. **Avenue:** Transition to decoding meteorological sequences via native temporal manifolds (e.g., via Spatiotemporal 3D Convolutions or integrated ConvLSTMs).
