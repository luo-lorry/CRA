# Conformal Risk Adaptation (CRA)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive implementation of Adaptive Alpha Conformal Prediction methods for improved uncertainty quantification in medical image segmentation and other applications.

## 🎯 Overview

This repository contains the implementation of a novel conformal risk control method for segmentation tasks:

- **Conformal Risk Adaptation (CRA)**: Learns confidence level for adaptive prediction sets

It significantly outperforms traditional conformal risk control (CRC) approach by adapting to the difficulty of individual samples.

## 🏆 Key Features

- **8 Conformal Prediction Methods**: Complete comparison framework including CRC, CRA, AA-CRC, AA-CRA, and 4 ablation methods
- **Statistical Analysis**: Comprehensive two-sample t-tests with effect size calculations
- **Rich Visualizations**: Box plots, trend analysis, and distribution comparisons
- **Modular Design**: Easy to extend with new methods and datasets
- **GPU Support**: Efficient implementation with CUDA acceleration
- **Reproducible**: Fixed random seeds and detailed experimental logs

## 🚀 Quick Start

### Basic Usage (Example: Running CRA for a single alpha)
```python
import numpy as np
import os
from src.methods.cra_method import run_cra_experiment

# Define your data folder (e.g., where phat_sorted.npy, phat_calibrated.npy, label_sorted.npy are located)
data_folder = "data/POLYPS" 

# Load your data (assuming these files exist in data_folder)
# phat_raw: raw predicted probabilities (flattened, sorted)
# phat_calibrated: probability-calibrated predicted probabilities (flattened, sorted)
# labels: ground truth masks (flattened, sorted corresponding to phat)
phat_raw = np.load(os.path.join(data_folder, "phat_sorted.npy"))
phat_calibrated = np.load(os.path.join(data_folder, "phat_calibrated.npy"))
labels = np.load(os.path.join(data_folder, "label_sorted.npy"))

# Filter out samples with no positive labels (important for coverage metrics)
has_positive_labels = labels.sum(axis=1) > 0
phat_raw = phat_raw[has_positive_labels]
phat_calibrated = phat_calibrated[has_positive_labels]
labels = labels[has_positive_labels]

n_total_samples = phat_raw.shape

# Define data split for this example (e.g., 70% calibration, 30% test)
# For CRA, all data not used for test is used for calibration.
test_ratio = 0.3
n_test = int(n_total_samples * test_ratio)
n_calib = n_total_samples - n_test

perm = np.random.permutation(n_total_samples)
calib_indices = perm[:n_calib]
test_indices = perm[n_calib:]

# Run CRA experiment
alpha_level = 0.1 # 90% target coverage
n_strat_groups = 5 # Number of stratification groups for CRA

results_cra = run_cra_experiment(
    phat_calibrated=phat_calibrated, 
    labels=labels, 
    calib_indices=calib_indices,
    test_indices=test_indices,
    alpha=alpha_level,
    n_groups=n_strat_groups
)

if results_cra:
    print(f"CRA Results for alpha={alpha_level}:")
    print(f"  Average Coverage: {np.mean(results_cra['coverage']):.3f}")
    print(f"  Average Gap: {np.mean(results_cra['gap']):.3f}")
    print(f"  Average Precision: {np.mean(results_cra['precision']):.3f}")
    print(f"  Average Size: {np.mean(results_cra['size']):.3f}")
else:
    print(f"CRA experiment failed for alpha={alpha_level}.")
```
### Complete Experimental Comparison
```bash
python scripts/run_all_experiments.py --input_folder data/POLYPS --alpha_values 0.05 0.1 0.2 --n_repeats 50 --train_ratio 0.3 --calib_ratio 0.2
```

## 📊 Methods Comparison

| Method | Non-conformity Score | Calibration | Stratification | Prediction Set |
|--------|---------------------|-------------|----------------|----------------|
| AA-CRC | Learned thresholds | Raw phat | None | Threshold-based |
| AA-CRA | Learned α' values | Raw phat | None | CRA-style |
| CRC | Direct phat | Raw phat | None | Threshold-based |
| CRC w strat | Direct phat | Raw phat | phat-sum groups | Group thresholds |
| CRA w/o calib+strat | CRA scores | Raw phat | None | Threshold-based |
| CRA w/o calib | CRA scores | Raw phat | phat-sum groups | Group thresholds |
| CRA w/o strat | CRA scores | Calibrated phat | None | Threshold-based |
| **CRA** | CRA scores | Calibrated phat | phat-sum groups | Group thresholds |

## 🔬 Experimental Framework

### Evaluation Metrics
- **Coverage**: Fraction of true positives included in prediction sets
- **Gap**: |Actual Coverage - Target Coverage|
- **Precision**: True positives / All predictions
- **Size**: Relative size of prediction sets

### Statistical Testing
- Two-sample t-tests with Bonferroni correction

## 📋 Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy 1.21+
- SciPy 1.7+
- Matplotlib 3.5+
- Pandas 1.3+
- tqdm 4.62+
- scikit-learn 1.0+
- torchvision (for ResNet weights)

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Thanks to the conformal prediction community for foundational work
  - [CRC](https://github.com/aangelopoulos/conformal-risk)
  - [AA-CRC](https://github.com/vincentblot28/AA-CRC)
- Image segmentation datasets providers
