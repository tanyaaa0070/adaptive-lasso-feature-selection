# 🎯 Adaptive LASSO Feature Selection

A comprehensive machine learning project that implements and compares **Adaptive LASSO (Dynamic Soft Thresholding)** with standard regression models for feature selection on the [House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) dataset.

---

## 📌 Overview

Feature selection is crucial for building interpretable and efficient ML models. This project demonstrates how **Adaptive LASSO** improves upon standard LASSO by assigning data-driven, adaptive weights to each feature's penalty — preserving important predictors while aggressively removing irrelevant ones.

### Models Compared

| Model | Description | Key Feature |
|---|---|---|
| **Linear Regression** | No regularization | Baseline benchmark |
| **Ridge Regression** | L2 penalty (Euclidean norm) | Shrinks coefficients uniformly |
| **Standard LASSO** | L1 penalty (Manhattan norm) | Sparse feature selection |
| **Adaptive LASSO** | Weighted L1 penalty | Data-driven, dynamic thresholding |

---

## 📂 Project Structure

```
feature_selection_using_LASSO/
├── adaptive_lasso_feature_selection.ipynb   # Main notebook with analysis & visualizations
├── download_data.py                         # Script to download House Prices dataset
├── run_notebook.py                          # Execute notebook headlessly
├── train.csv                                # Dataset (1,460 samples × 81 features)
├── requirements.txt                         # Python dependencies
├── Figures/                                 # Output visualizations
│   ├── mse_comparison.png                   # MSE bar chart
│   ├── coefficient_shrinkage.png            # LASSO vs Adaptive LASSO comparison
│   ├── feature_selection_comparison.png     # Feature counts across models
│   ├── top_features_adaptive_lasso.png      # Top 20 selected features
│   └── adaptive_weights_distribution.png    # Adaptive weight histograms
└── README.md                                # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/adaptive-lasso-feature-selection.git
   cd adaptive-lasso-feature-selection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download dataset (optional - included in repo):**
   ```bash
   python download_data.py
   ```

### Running the Project

#### Option 1: Run Jupyter Notebook
```bash
jupyter notebook adaptive_lasso_feature_selection.ipynb
```

#### Option 2: Execute Notebook Headlessly
```bash
python run_notebook.py
```

---

## 🔬 Key Features

✅ **Adaptive LASSO Implementation** — Custom adaptive weight calculation  
✅ **Model Comparison** — 4 regression models side-by-side performance analysis  
✅ **Feature Importance Analysis** — Identifies and ranks feature relevance  
✅ **Visualization Suite** — Comprehensive plots for model insights  
✅ **Reproducible Results** — Fixed random seed for consistency  

---

## 📊 Results & Insights

| Model | MSE | RMSE | Features Selected | Features Eliminated |
|---|---|---|---|---|
| Linear Regression | 2,641,205,374 | 51,393 | 244 (100%) | 0 |
| **Ridge Regression** | **1,294,235,032** | **35,975** | 243 (99.6%) | 1 |
| Standard LASSO | 1,830,218,058 | 42,781 | 197 (80.7%) | 47 |
| Adaptive LASSO | 2,459,492,127 | 49,593 | 195 (79.9%) | 49 |

### Key Findings

- **Ridge Regression** achieved the lowest MSE, benefiting from shrinkage without forced sparsity
- **Standard LASSO** eliminated 47 features (19.3%) while improving over Linear Regression
- **Adaptive LASSO** eliminated 49 features (20.1%) — the most aggressive feature selection
- Top features identified: `RoofMatl`, `GrLivArea`, `OverallQual`, `YearBuilt`, `TotalBsmtSF`

---

## 📚 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | ≥1.5 | Data manipulation |
| numpy | ≥1.23 | Numerical computing |
| scikit-learn | ≥1.2 | ML algorithms & evaluation |
| matplotlib | ≥3.6 | Plotting & visualization |
| seaborn | ≥0.12 | Statistical visualization |
| nbconvert | ≥7.2 | Notebook conversion |

---

## 💡 Technical Details

### Adaptive LASSO Methodology

1. **Initialization**: Calculate OLS coefficients
2. **Weight Calculation**: $w_j = ||\hat{\beta}_{OLS,j}||^{-\gamma}$ (where $\gamma = 1$)
3. **Optimization**: Minimize $\frac{1}{2n}||y - X\beta||_2^2 + \lambda \sum_{j=1}^p w_j |\beta_j|$
4. **Selection**: Features with non-zero coefficients are selected

### Performance Metric

Mean Squared Error (MSE) for model comparison across train/test splits.

---

## 📖 Mathematical Background

### Standard LASSO

$$\hat{\beta}_{\text{LASSO}} = \arg\min_{\beta} \left\{ \frac{1}{2n} \|y - X\beta\|_2^2 + \lambda \sum_{j=1}^{p} |\beta_j| \right\}$$

### Adaptive LASSO (Dynamic Soft Thresholding)

$$\hat{\beta}_{\text{AdaLASSO}} = \arg\min_{\beta} \left\{ \frac{1}{2n} \|y - X\beta\|_2^2 + \lambda \sum_{j=1}^{p} w_j |\beta_j| \right\}$$

Where adaptive weights are: $w_j = \frac{1}{|\hat{\beta}_j^{\text{init}}| + \epsilon}$

- **Large initial coefficients** → small weight → less penalization (important features preserved)
- **Small initial coefficients** → large weight → more penalization (irrelevant features removed)

---

## 📈 Visualizations

The notebook generates 5 publication-quality plots:

1. **MSE Comparison** — Bar chart comparing all 4 models
2. **Coefficient Shrinkage** — Side-by-side LASSO vs Adaptive LASSO coefficient profiles
3. **Feature Selection** — Non-zero vs zeroed coefficients per model
4. **Top Features** — Horizontal bar chart of top 20 Adaptive LASSO features
5. **Adaptive Weights** — Distribution and log-scale histogram of adaptive weights

---

## 📚 References

- Zou, H. (2006). *The Adaptive LASSO and Its Oracle Properties*. Journal of the American Statistical Association, 101(476), 1418–1429.
- Tibshirani, R. (1996). *Regression Shrinkage and Selection via the LASSO*. Journal of the Royal Statistical Society, Series B, 58(1), 267–288.

---

## 📝 License

This project is for educational and research purposes.
