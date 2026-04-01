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

def _manual_entry(model, explainer, feature_names, threshold):
    """Handle manual transaction entry."""

    st.subheader("Enter Transaction Details")
    st.caption(
        "Adjust the key features below. Unspecified V-features "
        "default to 0 (mean of PCA distribution)."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        amount = st.number_input("Amount (€)", 0.0, 50000.0, 100.0)
        time_val = st.number_input("Time (seconds)", 0, 172800, 50000)
    with col2:
        v14 = st.slider("V14", -20.0, 20.0, 0.0, 0.1)
        v12 = st.slider("V12", -20.0, 20.0, 0.0, 0.1)
    with col3:
        v10 = st.slider("V10", -20.0, 20.0, 0.0, 0.1)
        v17 = st.slider("V17", -20.0, 20.0, 0.0, 0.1)
    
    if st.button("🔍 Analyse Transaction", type="primary"):
        # Build input with all features
        input_data = np.zeros(len(feature_names))
        input_df = pd.DataFrame([input_data], columns=feature_names)

        # Set the user-specified values
        input_df['Time'] = time_val
        input_df['Amount'] = amount
        input_df['V14'] = v14
        input_df['V12'] = v12
        input_df['V10'] = v10
        input_df['V17'] = v17

        # Engineer features to match training
        input_df['Hour'] = (time_val / 3600) % 24
        input_df['Is_Night'] = int(
            input_df['Hour'].values[0] >= 22
            or input_df['Hour'].values[0] <= 5
        )
        input_df['Amount_log'] = np.log1p(amount)
        input_df['V14_x_V12'] = v14 * v12
        input_df['V14_x_V10'] = v14 * v10
        v_cols = [f'V{i}' for i in range(1, 29)]
        input_df['V_mean'] = input_df[v_cols].mean(axis=1)
        input_df['V_std'] = input_df[v_cols].std(axis=1)
        input_df['V_skew'] = input_df[v_cols].skew(axis=1)

        proba = model.predict_proba(input_df)[0][1]

        # Display result
        st.write("---")
        col_r, col_g = st.columns([1, 1])
