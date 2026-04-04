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
