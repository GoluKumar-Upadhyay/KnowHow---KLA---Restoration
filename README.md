![Project](https://img.shields.io/badge/DISTRIBUTION--ADAPTIVE--RESTORATION-gray?style=flat-square) ![Version](https://img.shields.io/badge/v1.0-1e3a8a?style=flat-square)

# Distribution-Adaptive Restoration of Degraded Semiconductor Inspection Images

***A Statistical Investigation and Corrected Mixture-Density Restoration Network***

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Array%20Ops-013243?style=flat-square&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-Distribution%20Fitting-8CAAE6?style=flat-square&logo=scipy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=flat-square&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=flat-square&logo=plotly&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=flat-square&logo=jupyter&logoColor=white)
![NVIDIA CUDA](https://img.shields.io/badge/NVIDIA-CUDA%20GPU-76B900?style=flat-square&logo=nvidia&logoColor=white)
![TensorBoard](https://img.shields.io/badge/TensorBoard-Training%20Logs-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![LPIPS](https://img.shields.io/badge/LPIPS-Perceptual%20Metric-9C27B0?style=flat-square)

![Task](https://img.shields.io/badge/Task-Image%20Restoration-673AB7?style=flat-square)
![Models](https://img.shields.io/badge/Models%20Trained-4-4CAF50?style=flat-square)
![Degradations](https://img.shields.io/badge/Degradation%20Types-3-2196F3?style=flat-square)
![Params](https://img.shields.io/badge/Params-116K-FF9800?style=flat-square)
![GPU](https://img.shields.io/badge/Verified%20GPU-RTX%203050-76B900?style=flat-square&logo=nvidia&logoColor=white)
![Dataset](https://img.shields.io/badge/Training%20Pairs-3%2C200-00BCD4?style=flat-square)

---

Restoration of semiconductor inspection images degraded by speckle noise,
additive Gaussian noise, and spatial downsampling — built as an
evidence-first research project: every architectural and loss-function
decision below follows a corresponding statistical test on the real data,
not convention or assumption.

---

## 1. Problem Statement

**Task (KLA challenge):** restore degraded semiconductor inspection
images. Input is a grayscale image (128x128) affected by speckle noise,
additive Gaussian noise, and downsampling relative to a clean
ground-truth image (256x256). Output must be a clean, super-resolved
image matching the ground truth as closely as possible. The model must
(a) generalize to structures not seen during training, and (b) be fast at
inference — explicitly benchmarked by the challenge.

No defect labels are provided or required — this is a pure image
restoration task, not defect detection.

---

## 2. Approach

Rather than assuming a standard denoising loss (L2/L1) would work, we
measured the **actual noise distribution in the real data** before
choosing an architecture or loss function, tested an initial hypothesis
against the evidence, revised it when the evidence disagreed, and only
then built and trained a model — then diagnosed and corrected a real
implementation bug discovered during that training, using the same
evidence-first discipline.

---

## 3. Key Findings

### 3.1 Censored-observation hypothesis — tested and rejected
An initial hypothesis treated "values exceeding the true image range" as
sensor clipping (a Tobit censored-regression problem). A direct
diagnostic on the real degraded images (pixel-value histograms, per-image
maximum clustering, exact-maximum pixel pile-up) found **no clipping
signature**: per-image maxima vary continuously, with no pile-up at any
repeated ceiling. This hypothesis was abandoned rather than forced onto
the data.

### 3.2 Global residual distribution — heavy-tailed, not Gaussian
Four candidate distributions (Gaussian, Laplace, Student-t, Generalized
Gaussian) were fit by maximum likelihood to the true residual
(`NoisyLR - downsampled(GT)`) over 6,553,600 pixels from 400 matched
pairs. The Generalized Gaussian fit (**beta = 0.845**) achieved the best
AIC by a decisive margin (~54,700 better than the next-best fit);
Gaussian was the worst of the four candidates.

### 3.3 Local (per-patch) analysis — the pivotal discovery
Fitting beta independently on 51,200 valid 32x32 patches across the
**full** 3,200-pair training set gave a local mean beta of **1.451**
(std 0.428) — substantially different from the global pooled value of
0.845. A one-way ANOVA confirmed this spatial variation is highly
significant (**F = 18.06, p < 1e-300**), and local beta correlates
significantly with edge density (r = -0.388), local variance
(r = -0.550), and entropy (r = -0.430), all p < 1e-300.

**This finding drove the architecture:** the apparent global
heavy-tailedness is, in part, a statistical artifact of pooling
spatially heterogeneous local noise regimes — motivating a model that
estimates the local mixture directly, rather than applying one fixed
global assumption.

### 3.4 A real implementation bug: mixture-component collapse
An initial trained implementation of the proposed model was directly
inspected and found to have collapsed: two of three learned mixture
components converged to an identical, non-heavy-tailed value
(`beta = 3.0`, the clamp boundary), and one component was completely
unused (0% usage). This was traced, via direct gradient analysis, to a
**zero-gradient region in a hard `clamp()` operation** — once two
components' pre-clamp values exceeded the boundary simultaneously, no
loss term, including a diversity penalty specifically designed to
prevent this, could separate them again, since the diversity penalty's
own gradient was also blocked by the clamp.

**Fix:** replaced the hard clamp with a smooth sigmoid reparameterization
(never fully saturates, so gradient never reaches exactly zero) and
added an entropy-based load-balancing penalty (stronger than the
mean-squared-error version tested first, which proved insufficient in
two independent runs). Both models were retrained from scratch on real
GPU hardware; the corrected models converged to a stable, non-degenerate
configuration (beta approximately [1.76, 2.20, 2.50], usage approximately
[0.19, 0.40, 0.41]) sustained from roughly epoch 60 through epoch 200
with no further collapse.

---

## 4. Final Solution

**Architecture:** `DistributionMixtureRestorationNet` = Local
Distribution Mixture Head (LDMH, predicts a per-pixel soft mixture over
K=3 learned Generalized-Gaussian prototypes) → FiLM feature conditioning
→ a compact NAFNet-style backbone (activation-free blocks, PixelShuffle
x2 upsampling for the 128→256 super-resolution) → restored image.

**Loss:** mixture Generalized-Gaussian negative log-likelihood + a
Charbonnier reconstruction term + a beta-diversity penalty + an
entropy-based load-balancing penalty (the last two added as the
component-collapse correction, Section 3.4).

**Four models trained** under an identical full-scale protocol (AdamW,
cosine LR schedule, mixed precision, gradient clipping, early stopping,
seed=42, batch=8, up to 300 epochs, 2880/320 train/validation split),
isolating each contribution:

| # | Model | Isolates |
|---|---|---|
| 1 | `BaselineNet` | Backbone only, plain Charbonnier loss |
| 2 | `Baseline_GenCharbonnier` | Backbone only, fixed global heavy-tail loss (beta=0.845) |
| 3 | `LDMH` | Mixture loss, FiLM conditioning **off** |
| 4 | `DistributionMixtureRestorationNet` | Mixture loss + FiLM conditioning **on** — proposed final model |

**No public/external dataset was used** — all training and evaluation
uses only the KLA-provided paired train set (GT + NoisyLR) and the
unlabeled competition test set.

---

## 5. Final Results

### Full-scale training (all four corrected models)

| Model | Best Epoch | PSNR (dB) | SSIM |
|---|---|---|---|
| Baseline | 200 | 28.120 | 0.7503 |
| + Global Gen. Charbonnier | 200 | 28.124 | 0.7511 |
| LDMH (corrected) | 200 | 28.138 | 0.7519 |
| **Full (corrected)** | 195 | **28.206** | **0.7533** |

Ordering is monotonic across all four configurations, consistent with
the design hypothesis: each additional mechanism yields a further,
modest improvement.

### Per-degradation-type robustness (n=30 held-out images)

| Model | Speckle | Gaussian | Downsample | Combined |
|---|---|---|---|---|
| Baseline | 24.580 | 28.096 | 28.449 | 24.218 |
| + Global GC | 24.635 | **28.203** | **28.649** | **24.256** |
| LDMH | 24.579 | 28.077 | 28.528 | 24.218 |
| **Full (corrected)** | **24.661** | 27.989 | 28.307 | 24.241 |

The corrected Full model achieves the best PSNR specifically on the
**speckle-only** condition — the degradation type most directly connected
to the multiplicative, heavy-tailed statistics motivating this work. The
fixed global loss remains strongest overall on 3 of 4 conditions; the
Full model is weakest on Gaussian-only. Reported candidly as a
degradation-specific, not uniform, advantage.

### Measured efficiency (NVIDIA RTX 3050, 4GB laptop GPU)

| Model | Params | GFLOPs | GPU (batch=1) | GPU (batch=16, FP16) |
|---|---|---|---|---|
| Baseline | 86,945 | 3.898 | 59.33 img/s | 200.84 img/s (1.72x) |
| Full | 116,138 | 4.844 | 54.89 img/s | 186.15 img/s (1.72x) |

### Out-of-distribution evaluation (10 external microscopy images)
Mean PSNR 35.87 dB (std 2.64), mean SSIM 0.746 (std 0.025) — external
Carinthia-S images with synthetic degradation applied, evaluated as
small-sample supporting evidence, not proof of full domain generalization.

---

## 6. Repository Structure

Actual project layout (flat root — no extra wrapper folder):

```
Semi conductor image paper/
├── Data_Analysis_Images/       # saved figures from the analysis notebooks
├── Documents/
|   |_____INSTALLATION.md              # library and hardware setup (CPU / GPU / H100)
|   |_____MODEL_USAGE_GUIDE.md          # detailed usage guide for loading and running the model
|   |_____Research.docx                 # Detailed Explanation of of Research Testing and Praposed Model Process
├── models/
│   ├── backbone.py              # NAFNet-style restoration backbone (x2 SR)
│   ├── ldmh.py                   # Local Distribution Mixture Head (corrected: sigmoid
│   │                              #   reparameterization + entropy load-balancing)
│   └── restoration_net.py        # BaselineNet + DistributionMixtureRestorationNet
├── utils/
│   ├── data.py                    # paired dataset loading + geometric augmentation
│   ├── losses.py                  # Charbonnier, Generalized Charbonnier
│   ├── metrics.py                  # PSNR / SSIM (numpy + torch)
│   └── train.py                    # full training loop (AdamW, cosine LR, AMP, grad
│                                    #   clipping, early stopping, per-epoch component-
│                                    #   health logging, TensorBoard, CSV, checkpointing)
├── notebooks/
│   ├── 03_Local_Distribution_Analysis.ipynb   # beta-map, ANOVA, correlations
│   ├── 04_LDMH_Design.ipynb                    # LDMH unit tests (no training)
│   ├── 05_Model_Training.ipynb                 # trains all 4 models, full protocol
│   ├── 06_Ablation_Study.ipynb                 # ablation ladder + hard-pixel metric
│   ├── 07_Comparison_and_OOD.ipynb              # degradation-type robustness + OOD
│   ├── 08_Visualization.ipynb                   # beta-maps, mixture maps, feature maps
│   ├── 09_Inference_Benchmark.ipynb             # params, FLOPs, latency, throughput, FP16
│   ├── Proposed_Model_Testing.ipynb             # full before/after demo + batch predictions
│   └── Quick_Load_And_Predict.ipynb             # minimal load-model-and-predict example
├── results/                    # checkpoints, logs, CSVs, figures (created at runtime)
├── Test_NoisyLR/                # KLA-provided unlabeled competition test set
├── train/                       # KLA-provided paired training data (GT + NoisyLR)
├── inference.py                # standalone CLI inference script (the judged deliverable)
| 
|
├── README.md                    # this file
├── data.zip                      # raw provided dataset archive
│
├── 01_Data_Analysis-1.py                              # early/exploratory: superseded by
├── 01_Data_Analysis_2.ipynb                            #   notebooks/03_Local_Distribution_
├── 01_Data_Analysis_3.ipynb                            #   Analysis.ipynb; kept for history,
├── 02_Local_Distribution_Analysis.ipynb                #   not part of the current pipeline
├── validates_the_statistical_claim_original_data.ipynb # early/exploratory: superseded by
└── validates_the_statistical_claim_synthesized.ipynb   #   the Phase-5 loss-comparison results
                                                          #   reported in Section 3
```

**Note on the loose files at root:** `01_Data_Analysis*`,
`02_Local_Distribution_Analysis.ipynb`, and the two
`validates_the_statistical_claim_*` notebooks are earlier exploratory
versions from before the project was organized into `notebooks/03-09`.
They are kept for development history but are **not** part of the
current, judged pipeline — the equivalent, current versions are
`notebooks/03_Local_Distribution_Analysis.ipynb` and the Phase-5
synthetic loss comparison described in Section 3.3.

---

## 7. How to Run

### Setup
See **`INSTALLATION.md`** for required libraries and hardware-specific
PyTorch installation (CPU, consumer GPU, or H100).

### Training
Open and run `notebooks/05_Model_Training.ipynb` top to bottom. Trains
all four models under the identical full-scale protocol described above.
Checkpoints, CSV logs, TensorBoard logs, and curve plots are saved
automatically to `results/`.

### Inference (the judged deliverable)
```bash
python inference.py --input_dir /path/to/degraded_npy_folder \
                     --output_dir /path/to/restored_output_folder \
                     --batch_size 16 \
                     --half
```
Full argument reference, hardware-specific tuning, and troubleshooting:
see **`MODEL_USAGE_GUIDE.md`**.

### Quick interactive check
Open `notebooks/Quick_Load_And_Predict.ipynb` for the minimal
load-model → load-image → predict → visualize example, or
`notebooks/Proposed_Model_Testing.ipynb` for the fuller test suite
(batch processing, quantitative metrics, multiple examples).

### Efficiency benchmarking
`notebooks/09_Inference_Benchmark.ipynb` reports parameter count, GFLOPs,
CPU/GPU latency, batched throughput, GPU peak memory, and an FP32-vs-FP16
comparison.

---

## 8. Documentation Index

| Document | Contents |
|---|---|
| `README.md` (this file) | Project overview, findings, results, structure |
| `INSTALLATION.md` | Required libraries; CPU / consumer GPU / H100 setup |
| `MODEL_USAGE_GUIDE.md` | Detailed usage guide: CLI arguments, hardware-specific commands, verification steps, troubleshooting |
| Manuscript (separate deliverable, not in this folder) | Full IEEE-format writeup: complete statistical derivations, mathematical formulation of the LDMH and the component-collapse correction, full experimental protocol, and citations |

---

## 9. Known Limitations

- **Statistical significance across seeds is not established.** All
  results (Tables in Section 5) come from single training runs per
  configuration; PSNR/SSIM differences on the order of 0.01–0.1 dB have
  not been tested for significance across multiple seeds.
- **The reduced-scale ablation study** (200 pairs, 3 epochs) still favors
  the simpler fixed-loss baseline over the full proposed model; the
  full-scale results (Section 5) reverse this ordering, consistent with a
  training-budget explanation, but a full-scale re-run of this specific
  ablation has not been performed to confirm it directly.
- **No comparison against externally-released restoration architectures**
  (Restormer, SwinIR, NAFNet as officially released, MambaIR) has been
  completed — only the models built and trained in this repository have
  been evaluated end-to-end.
- **Component usage, while no longer collapsing, is not perfectly
  uniform** (approximately 19%/40%/41% rather than an ideal 33%/33%/33%).
- **ONNX export, INT8 quantization, and TensorRT conversion** are not
  implemented in this repository; IoT/edge deployment guidance in
  `MODEL_USAGE_GUIDE.md` is based on the model's small size, not a verified
  benchmark on such hardware.
- One of the competition's own example slides shows a **Blur** step in
  an illustrative degradation pipeline, not present in the literal stated
  degradation formula (speckle + downsampling + additive Gaussian). This
  was flagged during development but not further investigated.
- `SPECKLE_LOOKS` / `GAUSSIAN_SIGMA` used in the isolated
  per-degradation-type robustness test (Section 5) are reasonable
  starting values, not fitted to the exact real speckle/Gaussian split,
  which is not directly separable from the real mixed data without
  further assumptions.

---

## 10. Reproducibility

- Dataset: 3,200 KLA-provided matched pairs; 2,880/320 train/validation
  split, fixed seed 42.
- Full-scale training: AdamW, cosine-annealing schedule, automatic mixed
  precision, gradient-norm clipping, early stopping (patience 30, maximum
  300 epochs), batch size 8, geometric augmentation (flip/rotation) on
  the training split only.
- LDMH hyperparameters: K=3 components, beta range [0.3, 2.5], softmax
  temperature 1.5, diversity margin 0.3, loss weights
  (mixture NLL, diversity, load-balance) = (1.0, 0.1, 0.3).
- Checkpoints record model weights, optimizer state, epoch, best
  validation PSNR/SSIM, and full per-epoch training history, including
  per-epoch mixture-component beta values, usage, and minimum pairwise
  beta distance, for both LDMH-based models.
- All measured efficiency figures were obtained on a single NVIDIA RTX
  3050 laptop GPU (4GB); results on other hardware, including production
  deployment GPU classes, have not been independently measured.