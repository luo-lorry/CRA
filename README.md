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

### Basic Usage
```python
from src.experiments.aa_cra import run_aa_cra_experiment
from src.utils.data_utils import load_medical_data

# Load your data
phat, labels, images = load_medical_data("path/to/dataset")

# Run AA-CRA experiment
results = run_aa_cra_experiment(
    phat=phat, 
    labels=labels, 
    images=images,
    alpha=0.1  # 90% target coverage
)

print(f"Average Coverage: {np.mean(results['coverage']):.3f}")
print(f"Average Gap: {np.mean(results['gap']):.3f}")
```
### Complete Experimental Comparison
```bash
python scripts/run_experiments.py --input_folder data/POLYPS --alpha_values 0.05 0.1 0.2
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

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Thanks to the conformal prediction community for foundational work
  - [CRC](https://github.com/aangelopoulos/conformal-risk)
  - [AA-CRC](https://github.com/vincentblot28/AA-CRC)
- Image segmentation datasets providers
