# Causal Uplift Modeling Project

This repository implements a full end-to-end causal inference pipeline to model Heterogeneous Treatment Effects (HTE) and predict uplift. It is built to optimize targeted interventions by identifying individuals who are most positively influenced by a treatment.

## 🚀 Overview

The pipeline uses the **Criteo Uplift Prediction Dataset**, a large-scale real-world dataset containing millions of rows, 12 features, a binary treatment indicator, and a binary conversion outcome.

The project implements several modern causal inference techniques from scratch or using specialized libraries, and rigorously evaluates their performance.

### 🧠 Implemented Models
- **S-Learner** (Single Model approach)
- **T-Learner** (Two Model approach)
- **X-Learner** (Cross-Learner)
- **R-Learner** (Double Machine Learning approach)
- **Causal Forest** (via EconML's `CausalForestDML`)

*Note: All meta-learners utilize LightGBM as the base estimator.*

## 📊 Results Summary

The models were evaluated using Area Under the Uplift Curve (AUUC), Qini coefficients, and expected Policy Value at various targeting thresholds. 

| Model | AUUC | Qini | Policy Value @ 5% |
|---|---|---|---|
| **S-Learner** 🏆 | **0.000018** | **4.33** | **0.001984** |
| **T-Learner** | 0.000015 | 3.52 | 0.001790 |
| **X-Learner** | -0.000044 | -10.39 | 0.000530 |
| **R-Learner** | -0.000574 | -136.56 | 0.000000 |
| **Causal Forest** | -0.000574 | -136.56 | 0.000000 |

*Winning model: **S-Learner** produced the highest overall uplift and Qini coefficient on the holdout test set.*

### Key Insights
1. **Feature Drivers**: Features `f8`, `f9`, and `f4` are the most significant drivers of treatment effect heterogeneity.
2. **Placebo Test**: Running the S-Learner on randomly shuffled treatment assignments yielded a near-zero AUUC (`0.000001`), confirming the signal learned by the model is genuine.

## 📁 Repository Structure

```
├── data/                    # Dataset directory
├── experiments/
│   └── run_full_pipeline.py # Main execution script
├── notebooks/               # Jupyter notebooks for EDA and walkthroughs
├── results/
│   ├── figures/             # Auto-generated plots (Qini, AUUC, Calibration, etc.)
│   └── metrics.json         # Full evaluation results
└── src/
    ├── preprocessing.py     # Data loading, splitting, and scaling
    ├── propensity.py        # Propensity score estimation and IPS weighting
    ├── metalearners.py      # S, T, X, and R learner implementations
    ├── causal_forest.py     # CausalForestDML implementation
    ├── evaluation.py        # Uplift metrics (AUUC, Qini, Placebo testing)
    └── visualization.py     # Plotting suite
```

## 🛠 Usage

1. **Install Dependencies:**
   Requires `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `lightgbm`, and `econml`.
2. **Run Pipeline:**
   ```bash
   python experiments/run_full_pipeline.py
   ```
   *Note: This script samples 10% of the dataset (~1.4M rows) to run within reasonable memory constraints. Modify `sample_frac` in the configuration block to run on the full dataset.*
