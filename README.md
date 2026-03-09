# Zara Menswear Sales Prediction

## Live App
The deployed Lovable app is available at: **[https://clear-insight-app.lovable.app]**

Predicting sales volume for 252 Zara menswear products using five regression models, with full SHAP explainability and an interactive Streamlit dashboard.

Built as part of MSIS 522 (Analytics and Machine Learning) at the Foster School of Business, University of Washington.

---

## Overview

This project follows an end-to-end data science workflow on a dataset of Zara men's clothing products collected in February 2024. Each product is described by its retail price, store placement, promotion status, seasonality, and clothing category. The goal is to predict sales volume using a range of regression approaches, compare their performance, and explain the best model's predictions using SHAP values.

**Best result:** XGBoost with R² = 0.474 and RMSE = 568.3 on a held-out test set.

---


## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/your-username/zara-menswear-sales-prediction.git
cd zara-menswear-sales-prediction
```

### 2. Install dependencies

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

### 3. Add the dataset

Place `zara_menswear.csv` inside the `data/` folder before running anything.

### 4. Run the analysis pipeline

```bash
python zara_analysis.py
```

This trains all five models, saves them to `models/`, and writes all plots to `plots/`. It must be run before launching the dashboard.

### 5. Launch the Streamlit dashboard

```bash
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`. The app requires the saved models and plots from the previous step.

---

## Models

| Model | Tuning | Notes |
|---|---|---|
| Linear Regression | None | Baseline |
| Decision Tree | 5-fold GridSearchCV | max_depth, min_samples_leaf |
| Random Forest | 5-fold GridSearchCV | n_estimators, max_depth |
| XGBoost | 5-fold GridSearchCV | n_estimators, max_depth, learning_rate |
| Neural Network (MLP) | None | Keras, 2 hidden layers, Adam optimizer |

---

## Dashboard Tabs

| Tab | Contents |
|---|---|
| Executive Summary | Dataset description, business context, key findings |
| Descriptive Analytics | Target distribution, feature plots, correlation heatmap |
| Model Performance | Comparison table, predicted vs. actual plots, hyperparameters |
| Explainability and Interactive Prediction | SHAP plots, live prediction with custom inputs and waterfall chart |

---

## Reproducibility

- `random_state=42` is used for all train/test splits, model initialization, and cross-validation.
- All dependencies are pinned in `requirements.txt`.
- Models are saved with `joblib` and loaded directly by the dashboard without retraining.
