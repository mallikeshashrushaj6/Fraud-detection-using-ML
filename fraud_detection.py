# =============================================================================
# FRAUD DETECTION ANALYTICS
# Dataset : Kaggle Credit Card Fraud Detection (creditcard.csv)
# Models  : Logistic Regression | Random Forest | XGBoost
# Techniques: SMOTE | StandardScaler | Threshold Tuning | SHAP
# Metric  : PR-AUC (primary), F1, Precision, Recall
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, roc_auc_score
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap

# =============================================================================
# STEP 1: LOAD DATASET
# =============================================================================
print("=" * 60)
print("STEP 1: Loading Dataset")
print("=" * 60)

df = pd.read_csv('creditcard.csv')
print(f"Shape        : {df.shape}")
print(f"Columns      : {list(df.columns)}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Duplicates   : {df.duplicated().sum()}")

# Drop duplicates
df.drop_duplicates(inplace=True)
print(f"Shape after removing duplicates: {df.shape}")

# =============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
print("\n" + "=" * 60)
print("STEP 2: Exploratory Data Analysis")
print("=" * 60)

fraud_count = df['Class'].value_counts()
fraud_pct = df['Class'].value_counts(normalize=True) * 100
print(f"\nClass Distribution:")
print(f"  Legitimate (0): {fraud_count[0]:,} ({fraud_pct[0]:.3f}%)")
print(f"  Fraudulent (1): {fraud_count[1]:,} ({fraud_pct[1]:.3f}%)")
print(f"  Imbalance Ratio: {fraud_count[0] // fraud_count[1]}:1")

print(f"\nTransaction Amount Stats:")
print(df.groupby('Class')['Amount'].describe().round(2))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Fraud Detection - Exploratory Data Analysis', fontsize=16, fontweight='bold')

# Class distribution
axes[0, 0].bar(['Legitimate', 'Fraudulent'], [fraud_count[0], fraud_count[1]],
               color=['#2196F3', '#F44336'], edgecolor='black')
axes[0, 0].set_title('Class Distribution')
axes[0, 0].set_ylabel('Count')
for i, v in enumerate([fraud_count[0], fraud_count[1]]):
    axes[0, 0].text(i, v + 500, f'{v:,}', ha='center', fontweight='bold')

# Amount distribution by class
fraud_amounts = df[df['Class'] == 1]['Amount']
legit_amounts = df[df['Class'] == 0]['Amount']
axes[0, 1].hist(legit_amounts, bins=50, alpha=0.6, label='Legitimate', color='#2196F3')
axes[0, 1].hist(fraud_amounts, bins=50, alpha=0.8, label='Fraudulent', color='#F44336')
axes[0, 1].set_title('Transaction Amount Distribution')
axes[0, 1].set_xlabel('Amount (EUR)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].legend()
axes[0, 1].set_xlim(0, 2000)

# Boxplot: Amount by class
data_box = [legit_amounts.values, fraud_amounts.values]
axes[1, 0].boxplot(data_box, tick_labels=['Legitimate', 'Fraudulent'],
                   patch_artist=True,
                   boxprops=dict(facecolor='#E3F2FD'),
                   medianprops=dict(color='red', linewidth=2))
axes[1, 0].set_title('Amount by Class (Boxplot)')
axes[1, 0].set_ylabel('Amount (EUR)')
axes[1, 0].set_ylim(0, 1000)

# Transaction count over time
axes[1, 1].scatter(df[df['Class'] == 0]['Time'], df[df['Class'] == 0]['Amount'],
                   alpha=0.1, s=1, color='#2196F3', label='Legitimate')
axes[1, 1].scatter(df[df['Class'] == 1]['Time'], df[df['Class'] == 1]['Amount'],
                   alpha=0.6, s=8, color='#F44336', label='Fraudulent')
axes[1, 1].set_title('Transactions Over Time')
axes[1, 1].set_xlabel('Time (seconds)')
axes[1, 1].set_ylabel('Amount (EUR)')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('eda_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nEDA chart saved: eda_analysis.png")

# Correlation heatmap (top features)
plt.figure(figsize=(12, 8))
corr = df.corr()['Class'].drop('Class').abs().sort_values(ascending=False)
top_features = corr.head(15).index.tolist() + ['Class']
sns.heatmap(df[top_features].corr(), annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5)
plt.title('Correlation Heatmap - Top 15 Features vs Class', fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Correlation heatmap saved: correlation_heatmap.png")

# =============================================================================
# STEP 3: PREPROCESSING
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3: Preprocessing")
print("=" * 60)

scaler = StandardScaler()
df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
df['Time_scaled'] = scaler.fit_transform(df[['Time']])

# Drop original unscaled columns
df.drop(columns=['Amount', 'Time'], inplace=True)

# Features and target
X = df.drop('Class', axis=1)
y = df['Class']

print(f"Features shape : {X.shape}")
print(f"Target shape   : {y.shape}")
print(f"Feature columns: {list(X.columns)}")

# =============================================================================
# STEP 4: TRAIN / TEST SPLIT (stratified, before SMOTE)
# =============================================================================
print("\n" + "=" * 60)
print("STEP 4: Train/Test Split (Stratified 80/20)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set   : {X_train.shape[0]:,} samples")
print(f"  Fraud        : {y_train.sum():,} ({y_train.mean()*100:.3f}%)")
print(f"Test set       : {X_test.shape[0]:,} samples")
print(f"  Fraud        : {y_test.sum():,} ({y_test.mean()*100:.3f}%)")

# =============================================================================
# STEP 5: SMOTE (applied ONLY on training data to prevent data leakage)
# =============================================================================
print("\n" + "=" * 60)
print("STEP 5: Applying SMOTE on Training Data Only")
print("=" * 60)

smote = SMOTE(sampling_strategy=0.1, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print(f"Before SMOTE   : {X_train.shape[0]:,} samples | Fraud: {y_train.sum():,}")
print(f"After SMOTE    : {X_resampled.shape[0]:,} samples | Fraud: {y_resampled.sum():,}")
print(f"New fraud ratio: {y_resampled.mean()*100:.2f}%")

# =============================================================================
# STEP 6: TRAIN MODELS
# =============================================================================
print("\n" + "=" * 60)
print("STEP 6: Training Models")
print("=" * 60)

models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, random_state=42, n_jobs=1
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=50, random_state=42, n_jobs=1
    ),
    'XGBoost': XGBClassifier(
        use_label_encoder=False, eval_metric='logloss',
        random_state=42, n_jobs=1, verbosity=0
    )
}

trained_models = {}
model_probs = {}

for name, model in models.items():
    print(f"\n  Training {name}...", end=" ")
    model.fit(X_resampled, y_resampled)
    probs = model.predict_proba(X_test)[:, 1]
    trained_models[name] = model
    model_probs[name] = probs
    pr_auc = average_precision_score(y_test, probs)
    print(f"Done | PR-AUC: {pr_auc:.4f}")

# =============================================================================
# STEP 7: THRESHOLD TUNING
# =============================================================================
print("\n" + "=" * 60)
print("STEP 7: Threshold Tuning")
print("=" * 60)

THRESHOLD = 0.3  # Lower threshold = catch more fraud (higher recall, lower precision)
print(f"Selected threshold: {THRESHOLD}")
print("(Lower threshold = more fraud flagged = higher recall, more false positives)")
print("(This is a business decision: cost of missing fraud vs cost of false alerts)")

model_preds_tuned = {}
for name, probs in model_probs.items():
    preds = (probs >= THRESHOLD).astype(int)
    model_preds_tuned[name] = preds

# =============================================================================
# STEP 8: EVALUATE MODELS
# =============================================================================
print("\n" + "=" * 60)
print("STEP 8: Model Evaluation")
print("=" * 60)

results = {}

for name in models.keys():
    probs = model_probs[name]
    preds = model_preds_tuned[name]

    pr_auc  = average_precision_score(y_test, probs)
    roc_auc = roc_auc_score(y_test, probs)
    f1      = f1_score(y_test, preds)
    prec    = precision_score(y_test, preds)
    rec     = recall_score(y_test, preds)

    results[name] = {
        'PR-AUC': pr_auc, 'ROC-AUC': roc_auc,
        'F1': f1, 'Precision': prec, 'Recall': rec
    }

    print(f"\n{'─'*40}")
    print(f"  {name}")
    print(f"{'─'*40}")
    print(f"  PR-AUC    : {pr_auc:.4f}  ← PRIMARY METRIC")
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, preds, target_names=['Legitimate', 'Fraudulent']))

# Summary table
print("\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)
results_df = pd.DataFrame(results).T.round(4)
print(results_df.to_string())

# Identify best model by PR-AUC
best_model_name = results_df['PR-AUC'].idxmax()
print(f"\nBest Model (PR-AUC): {best_model_name}")

# =============================================================================
# EVALUATION CHARTS
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Model Evaluation - Fraud Detection', fontsize=16, fontweight='bold')

colors = ['#2196F3', '#4CAF50', '#FF9800']

# Confusion matrices
for idx, name in enumerate(models.keys()):
    cm = confusion_matrix(y_test, model_preds_tuned[name])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, idx],
                xticklabels=['Legit', 'Fraud'],
                yticklabels=['Legit', 'Fraud'])
    axes[0, idx].set_title(f'Confusion Matrix\n{name}')
    axes[0, idx].set_ylabel('Actual')
    axes[0, idx].set_xlabel('Predicted')

# Precision-Recall curves
for idx, (name, probs) in enumerate(model_probs.items()):
    precision, recall, _ = precision_recall_curve(y_test, probs)
    pr_auc = results[name]['PR-AUC']
    axes[1, 0].plot(recall, precision, color=colors[idx],
                    label=f'{name} (AUC={pr_auc:.3f})', linewidth=2)
axes[1, 0].set_xlabel('Recall')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].set_title('Precision-Recall Curves')
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(True, alpha=0.3)

# PR-AUC bar chart
pr_aucs = [results[n]['PR-AUC'] for n in models.keys()]
bars = axes[1, 1].bar(list(models.keys()), pr_aucs, color=colors, edgecolor='black')
axes[1, 1].set_title('PR-AUC Comparison (Primary Metric)')
axes[1, 1].set_ylabel('PR-AUC Score')
axes[1, 1].set_ylim(0, 1)
for bar, val in zip(bars, pr_aucs):
    axes[1, 1].text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f'{val:.4f}',
                    ha='center', fontweight='bold', fontsize=9)
axes[1, 1].tick_params(axis='x', rotation=15)

# F1 / Precision / Recall grouped bar
metrics_to_plot = ['F1', 'Precision', 'Recall']
x = np.arange(len(models))
width = 0.25
for i, metric in enumerate(metrics_to_plot):
    vals = [results[n][metric] for n in models.keys()]
    axes[1, 2].bar(x + i * width, vals, width, label=metric, edgecolor='black')
axes[1, 2].set_title('F1 / Precision / Recall Comparison')
axes[1, 2].set_xticks(x + width)
axes[1, 2].set_xticklabels(list(models.keys()), rotation=15, fontsize=8)
axes[1, 2].set_ylabel('Score')
axes[1, 2].set_ylim(0, 1)
axes[1, 2].legend()

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nEvaluation chart saved: model_evaluation.png")

# =============================================================================
# STEP 9: SHAP EXPLAINABILITY (on best model)
# =============================================================================
print("\n" + "=" * 60)
print(f"STEP 9: SHAP Explainability — {best_model_name}")
print("=" * 60)

best_model = trained_models[best_model_name]

# Use a sample for SHAP speed
X_test_sample = X_test.sample(n=min(500, len(X_test)), random_state=42)

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test_sample)

# Handle multi-output SHAP (Random Forest returns list)
if isinstance(shap_values, list):
    shap_vals = shap_values[1]  # class 1 (fraud)
else:
    shap_vals = shap_values

# SHAP Summary Plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_vals, X_test_sample, plot_type="bar",
                  max_display=15, show=False)
plt.title(f'SHAP Feature Importance — {best_model_name}', fontweight='bold')
plt.tight_layout()
plt.savefig('shap_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("SHAP importance chart saved: shap_importance.png")

# SHAP Beeswarm
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_vals, X_test_sample, max_display=15, show=False)
plt.title(f'SHAP Summary (Beeswarm) — {best_model_name}', fontweight='bold')
plt.tight_layout()
plt.savefig('shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.close()
print("SHAP beeswarm chart saved: shap_beeswarm.png")

# =============================================================================
# STEP 10: RISK SCORING
# =============================================================================
print("\n" + "=" * 60)
print("STEP 10: Risk Scoring")
print("=" * 60)

best_probs = model_probs[best_model_name]

def assign_risk(prob):
    if prob >= 0.7:   return 'HIGH'
    elif prob >= 0.3: return 'MEDIUM'
    else:             return 'LOW'

risk_df = pd.DataFrame({
    'fraud_probability': best_probs,
    'risk_score': (best_probs * 100).round(2),
    'risk_level': [assign_risk(p) for p in best_probs],
    'actual_class': y_test.values,
    'predicted_fraud': model_preds_tuned[best_model_name]
})

print("\nRisk Level Distribution:")
print(risk_df['risk_level'].value_counts())
print(f"\nSample high-risk transactions (top 10):")
print(risk_df.sort_values('fraud_probability', ascending=False).head(10).to_string(index=False))

risk_df.to_csv('risk_scores.csv', index=False)
print("\nRisk scores saved: risk_scores.csv")

# Risk distribution chart
plt.figure(figsize=(10, 5))
risk_counts = risk_df['risk_level'].value_counts()
colors_risk = {'LOW': '#4CAF50', 'MEDIUM': '#FF9800', 'HIGH': '#F44336'}
bars = plt.bar(risk_counts.index,
               risk_counts.values,
               color=[colors_risk[r] for r in risk_counts.index],
               edgecolor='black')
plt.title('Transaction Risk Level Distribution', fontweight='bold', fontsize=14)
plt.xlabel('Risk Level')
plt.ylabel('Number of Transactions')
for bar, val in zip(bars, risk_counts.values):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 50, f'{val:,}',
             ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('risk_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Risk distribution chart saved: risk_distribution.png")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"\nDataset       : 284,807 transactions | 0.17% fraud")
print(f"Models trained: Logistic Regression, Random Forest, XGBoost")
print(f"Best model    : {best_model_name}")
print(f"  PR-AUC      : {results[best_model_name]['PR-AUC']:.4f}")
print(f"  F1 Score    : {results[best_model_name]['F1']:.4f}")
print(f"  Precision   : {results[best_model_name]['Precision']:.4f}")
print(f"  Recall      : {results[best_model_name]['Recall']:.4f}")
print(f"\nOutput files:")
print("  eda_analysis.png        — EDA charts")
print("  correlation_heatmap.png — Feature correlations")
print("  model_evaluation.png    — Confusion matrices + PR curves")
print("  shap_importance.png     — Feature importance via SHAP")
print("  shap_beeswarm.png       — SHAP beeswarm summary")
print("  risk_scores.csv         — Risk scores for all test transactions")
print("=" * 60)