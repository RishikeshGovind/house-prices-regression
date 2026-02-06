# House Prices — Advanced Regression with Model Stacking

This repository contains an end-to-end regression pipeline for predicting house prices using the Ames Housing dataset.

The project covers data preprocessing, feature engineering, multiple gradient boosting models, and a stacked ensemble to improve predictive performance.

---

## Dataset

- **Source:** Kaggle — [House Prices: Advanced Regression Techniques](https://www.kaggle.com/code/rg18aa/house-prices-advanced-regression-by-model-stacking)
- **Target variable:** `SalePrice`
- **Evaluation metric:** RMSE (on log-transformed target)

---

## Workflow Overview

1. Data loading and inspection  
2. Exploratory data analysis (EDA)  
3. Missing value handling  
4. Feature engineering  
5. Model training with cross-validation  
6. Model stacking  
7. Final prediction and submission generation  

---

## Feature Engineering

Key engineered features include:

- Total square footage features
- House age and remodel age
- Quality × size interaction features
- Log-transformed skewed numerical variables
- Ordinal encoding for quality-based categorical features

---

## Models

### Base Models
- **CatBoost (2 configurations)**
- **LightGBM (2 configurations)**

Each base model is trained using K-Fold cross-validation, producing out-of-fold predictions.

### Meta Model
- **ElasticNet Regression**

The meta model is trained on out-of-fold predictions from the base models to produce final stacked predictions.

---

## Validation

- **Cross-validation:** K-Fold
- **Metric:** RMSE on `log1p(SalePrice)`
- Stacking is performed using only out-of-fold predictions to avoid leakage.

---

## Usage

1. Open `house_prices_stacking.ipynb`
2. Run all cells to reproduce training and predictions
3. Generated submission file:
submission.csv


---

## Dependencies

- Python 3
- NumPy
- pandas
- scikit-learn
- LightGBM
- CatBoost
- matplotlib
- seaborn

---

## Notes

- The notebook is designed to be reproducible.
- Hyperparameters are fixed for consistency across runs.
- No test data leakage is used during training or stacking.

---
