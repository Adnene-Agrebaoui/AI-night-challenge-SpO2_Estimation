# Contactless SpO2 Estimation from Facial RGB Videos

A machine learning pipeline that estimates blood oxygen saturation (SpO2) from facial video, without any physical contact. The system uses remote photoplethysmography (rPPG) to detect subtle color changes in skin caused by pulsatile blood flow.

---

## What This Does

Most pulse oximeters require a clip on your finger. This system does the same job using only a standard RGB camera pointed at a person's face. It extracts physiological signals from video, then uses a trained model to predict SpO2 values.

Two models are trained and compared:

- **Approach 1** — Standard regression. The model learns purely from data with no domain knowledge applied.
- **Approach 2** — Constrained regression. The model is guided by physiological rules during training: valid SpO2 range, temporal smoothness, and higher penalty for clinically critical predictions.

---

## Project Structure

```
.
├── SpO2_Estimation_rPPG.ipynb   # Main notebook (full pipeline)
├── model_unconstrained.pt        # Saved weights — Approach 1
├── model_constrained.pt          # Saved weights — Approach 2
├── results_table.csv             # Quantitative comparison table
├── eda_plots.png                 # Exploratory data analysis
├── rppg_algorithms.png           # rPPG signal comparison (Green / CHROM / POS)
├── results_dashboard.png         # Full results visualization
└── error_cdf.png                 # Cumulative error distribution
```

---

## Requirements

```
Python >= 3.8
torch
torchvision
opencv-python-headless
mediapipe
scikit-learn
pandas
numpy
scipy
matplotlib
seaborn
tqdm
```

Install all at once:

```bash
pip install torch torchvision opencv-python-headless mediapipe scikit-learn pandas numpy scipy matplotlib seaborn tqdm
```

---

## How to Run

1. Open `SpO2_Estimation_rPPG.ipynb` in Jupyter or VS Code.
2. Update the data paths at the top of Section 1:
   ```python
   DATA_PATH = 'your_dataset.csv'
   VIDEO_DIR = 'your_videos/'
   ```
3. Run all cells in order.

If no dataset is provided, the notebook automatically generates a synthetic dataset of 18,173 samples for demonstration.

---

## Dataset Format

The CSV should contain one row per video segment with the following columns:

| Column | Description |
|---|---|
| feature1 ... feature6 | Extracted rPPG signal features (AC/DC ratios, R-ratio, SNR, etc.) |
| HR | Ground truth heart rate (bpm) |
| SpO2 | Ground truth blood oxygen saturation (%) |
| segment_id | Unique identifier for each time segment |
| video_path | Path to the source video file |

The train/test split is done by video, not by row, to prevent data leakage across segments from the same recording.

---

## Pipeline Overview

```
Video file
   |
   v
Face detection (MediaPipe Face Mesh / Haar cascade fallback)
   |
   v
ROI extraction (forehead region)
   |
   v
RGB trace accumulation (per frame spatial average)
   |
   v
rPPG signal extraction (Green / CHROM / POS algorithms)
   |
   v
Feature extraction (AC/DC ratios, R-ratio, SNR, PSD peak)
   |
   v
SpO2 regression model (Approach 1 or 2)
   |
   v
SpO2 prediction (%)
```

---

## rPPG Algorithms

Three algorithms are implemented for extracting the pulse signal from facial color changes:

**Green Channel** — Simplest method. Uses the green channel alone, which has the highest contrast for hemoglobin absorption.

**CHROM** — Chrominance-based method (de Haan & Jeanne, 2013). Separates the pulsatile signal from specular reflection by operating in the chrominance plane.

**POS** — Plane Orthogonal to Skin (Wang et al., 2017). Projects the RGB signal onto a plane orthogonal to the skin-tone direction, making it more robust to illumination changes.

POS is used as the primary algorithm in the pipeline.

---

## Model Details

Both models share the same MLP architecture:

- Input layer (6 features)
- Hidden layers: 256 -> 128 -> 64 -> 32, with BatchNorm, GELU activation, and Dropout
- Output: single SpO2 value

**Approach 1** uses standard MSE loss.

**Approach 2** adds three components to the loss:
- Range penalty: penalizes predictions outside [70, 100]
- Smoothness penalty: penalizes large jumps between consecutive predictions
- Clinical weighting: applies 2x loss weight for samples where SpO2 < 95%

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| MAE | Mean absolute error in SpO2 percentage |
| RMSE | Root mean squared error |
| R2 | Coefficient of determination |
| Stability | Mean absolute change between consecutive predictions |
| Out-of-range % | Fraction of predictions outside the valid [70, 100] range |

---

## Key Results

The constrained model (Approach 2) produces zero out-of-range predictions and shows better temporal stability. The unconstrained model may achieve lower error on balanced data but can produce physiologically impossible values, which is unacceptable in a clinical context.

Per-category analysis shows the constrained model is especially more accurate in the hypoxic range (SpO2 < 95%), which is the most clinically important segment.

---

## Limitations

- Performance depends heavily on lighting conditions and camera quality.
- Models trained on homogeneous populations may underperform on different skin tones due to differences in melanin absorption.
- This is not a replacement for medical-grade pulse oximetry.
- Accuracy in the hypoxic range (SpO2 < 90%) may be limited by data scarcity.

---

## Reproducibility

All random seeds are fixed at the start of the notebook (seed = 42). Results should be fully reproducible across runs on the same hardware.

---

## References

- de Haan, G. & Jeanne, V. (2013). Robust pulse rate from chrominance-based rPPG. *IEEE Transactions on Biomedical Engineering.*
- Wang, W. et al. (2017). Algorithmic principles of remote PPG. *IEEE Transactions on Biomedical Engineering.*
- Verkruysse, W. et al. (2008). Remote plethysmographic imaging using ambient light. *Optics Express.*
