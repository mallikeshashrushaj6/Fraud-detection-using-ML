[README.md](https://github.com/user-attachments/files/29491490/README.md)
# Fraud Detection Using Machine Learning

A machine learning pipeline that detects fraudulent credit card transactions using classification models trained on highly imbalanced data.

## Overview

This project identifies fraudulent transactions from a dataset where fraud makes up only 0.17% of all records. It compares three models, handles class imbalance with SMOTE, and uses SHAP for model explainability.

## Dataset

- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Records:** 284,807 transactions (473 fraudulent after duplicate removal)
- **Features:** 28 PCA-transformed features (V1–V28), Time, Amount
- **Note:** `creditcard.csv` is not included in this repo due to file size. Download it from Kaggle and place it in the project root before running.

## Models Used

| Model | Purpose |
|---|---|
| Logistic Regression | Fast, interpretable baseline |
| Random Forest | Ensemble method, handles non-linearity |
| XGBoost | Best performer on imbalanced tabular data |

## Techniques

- **SMOTE** — synthetic oversampling of the minority class, applied only to training data to avoid data leakage
- **StandardScaler** — normalizes `Amount` and `Time`
- **Threshold tuning** — adjusts the fraud-decision cutoff (default 0.3) based on precision/recall tradeoff
- **SHAP** — explains which features drive each model's fraud predictions

## Evaluation Metrics

Primary metric is **PR-AUC** (Precision-Recall AUC), not accuracy, since accuracy is misleading on a 99.8%/0.2% imbalanced dataset. Also reports ROC-AUC, F1, Precision, and Recall.

## How to Run

```bash
pip install -r requirements.txt
python fraud_detection.py
```

Place `creditcard.csv` in the same folder before running.

## Output Files

- `eda_analysis.png` — exploratory data analysis charts
- `correlation_heatmap.png` — feature correlation with fraud
- `model_evaluation.png` — confusion matrices and PR curves for all 3 models
- `shap_importance.png` / `shap_beeswarm.png` — feature importance via SHAP
- `risk_scores.csv` — per-transaction fraud probability and risk level (LOW/MEDIUM/HIGH)

## Author

Shashrusha Mallike
