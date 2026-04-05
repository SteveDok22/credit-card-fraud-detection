"""
Pre-compute all data needed by the dashboard into small files.
Run this ONCE locally before deploying.
"""
import pandas as pd
import numpy as np
import joblib
import json
import os

print("Generating dashboard-ready data...")

# --- For page_fraud_study and page_hypotheses ---
# Save a sample of raw data (much smaller than full 150MB)
df = pd.read_csv("data/creditcard.csv")

# Keep ALL fraud + sample of legit (enough for all visualisations)
fraud = df[df['Class'] == 1]
legit = df[df['Class'] == 0].sample(10000, random_state=42)
sample = pd.concat([legit, fraud]).sample(frac=1, random_state=42)

os.makedirs("outputs/dashboard", exist_ok=True)

# Save full class counts for accurate stats
dataset_stats = {
    'total_transactions': int(len(df)),
    'fraud_count': int(df['Class'].sum()),
    'legit_count': int((df['Class'] == 0).sum()),
    'fraud_pct': float(df['Class'].mean()),
    'amount_mean_legit': float(df[df['Class'] == 0]['Amount'].mean()),
    'amount_median_legit': float(df[df['Class'] == 0]['Amount'].median()),
    'amount_mean_fraud': float(df[df['Class'] == 1]['Amount'].mean()),
    'amount_median_fraud': float(df[df['Class'] == 1]['Amount'].median()),
}
with open("outputs/dashboard/dataset_stats.json", 'w') as f:
    json.dump(dataset_stats, f, indent=2)

sample.to_csv("outputs/dashboard/data_sample.csv", index=False)
print(f"Data sample saved: {len(sample)} rows")

# --- For page_ml_performance (train confusion matrix) ---
X_train = pd.read_csv("outputs/v1/X_train_resampled.csv")
y_train = pd.read_csv("outputs/v1/y_train_resampled.csv").squeeze()
model = joblib.load("outputs/v2/fraud_model_optimized.pkl")

from sklearn.metrics import confusion_matrix
y_train_pred = model.predict(X_train)
cm_train = confusion_matrix(y_train, y_train_pred).tolist()

with open("outputs/dashboard/train_confusion_matrix.json", 'w') as f:
    json.dump(cm_train, f)
print("Train confusion matrix saved.")