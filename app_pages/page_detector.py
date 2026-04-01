"""Page 4: Fraud Detector — answers BR2 with predictions and SHAP."""
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import joblib
import json
import time


def page_detector():
    """Display the fraud detection tool page."""

    st.title("🎯 Fraud Detection Tool")
    st.info(
        "**Business Requirement 2:** Predict whether a transaction is "
        "fraudulent or legitimate with explainable results."
    )
    st.write("---")

    # Load model and explainer
    model = joblib.load("outputs/v2/fraud_model_optimized.pkl")
    explainer = joblib.load("outputs/v2/shap_explainer.pkl")
    feature_names = pd.read_csv(
        "outputs/v1/X_test_engineered.csv", nrows=0
    ).columns.tolist()

    with open("outputs/v2/optimal_threshold.json") as f:
        threshold = json.load(f)['optimal_threshold']

    input_mode = st.radio(
        "Input Method",
        ["Manual Entry", "Upload CSV", "Live Simulation"]
    )

    if input_mode == "Manual Entry":
        _manual_entry(model, explainer, feature_names, threshold)
    elif input_mode == "Upload CSV":
        _csv_upload(model, feature_names, threshold)
    elif input_mode == "Live Simulation":
        _live_simulation(model, feature_names, threshold)