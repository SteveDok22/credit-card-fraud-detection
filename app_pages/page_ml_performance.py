"""Page 7: ML Pipeline Performance — evaluation metrics and comparison."""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score
)


def page_ml_performance():
    """Display the ML pipeline performance page."""

    st.title("📊 ML Pipeline Performance")

    # Load data
    X_train = pd.read_csv("outputs/v1/X_train_resampled.csv")
    y_train = pd.read_csv("outputs/v1/y_train_resampled.csv").squeeze()
    y_test = pd.read_csv("outputs/v1/y_test.csv").squeeze()
    model = joblib.load("outputs/v2/fraud_model_optimized.pkl")
    y_test_proba = joblib.load("outputs/v2/test_probabilities.pkl")

    with open("outputs/v2/optimal_threshold.json") as f:
        threshold = json.load(f)['optimal_threshold']
    with open("outputs/v2/tuning_results.json") as f:
        tuning = json.load(f)

    y_test_pred = (y_test_proba >= threshold).astype(int)
    test_report = classification_report(
        y_test, y_test_pred, output_dict=True,
        target_names=['Legitimate', 'Fraud']
    )

    # Success statement (LO4: 4.2)
    fraud_f1 = test_report['Fraud']['f1-score']
    if fraud_f1 >= 0.80:
        st.success(
            f"✅ **The ML pipeline meets the business requirements.** "
            f"The XGBoost model achieves F1 = {fraud_f1:.4f} on the "
            f"fraud class, exceeding the target of F1 ≥ 0.80."
        )
    else:
        st.warning(
            f"⚠️ The model achieves F1 = {fraud_f1:.4f}. "
            f"See details below."
        )

    st.write("---")