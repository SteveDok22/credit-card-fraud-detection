"""Page 4: Fraud Detector — answers BR2 with predictions and SHAP."""
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import joblib
import json
import time
from src.data_management import (
    load_model, load_feature_names, load_simulation_sample
)
import shap

def page_detector():
    """Display the fraud detection tool page."""

    st.title("🎯 Fraud Detection Tool")
    st.info(
        "**Business Requirement 2:** Predict whether a transaction is "
        "fraudulent or legitimate with explainable results."
    )
    st.write("---")

    # Load model and explainer
    model = load_model()
    explainer = shap.TreeExplainer(model)
    feature_names = load_feature_names()

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

        with col_r:
            if proba >= threshold:
                st.error(f"⚠️ **FRAUD DETECTED**")
            else:
                st.success(f"✅ **LEGITIMATE TRANSACTION**")
            st.metric("Fraud Probability", f"{proba:.1%}")
            st.metric("Threshold", f"{threshold:.2f}")

        with col_g:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                title={'text': "Fraud Risk Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': (
                        "#EF553B" if proba >= threshold else "#00CC96"
                    )},
                    'steps': [
                        {'range': [0, 30], 'color': '#d4edda'},
                        {'range': [30, 70], 'color': '#fff3cd'},
                        {'range': [70, 100], 'color': '#f8d7da'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'value': threshold * 100
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        # SHAP explanation
        st.subheader("Why This Prediction?")
        shap_values = explainer.shap_values(input_df)

        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP Value': shap_values[0],
            'Abs SHAP': np.abs(shap_values[0])
        }).sort_values('Abs SHAP', ascending=False).head(10)

        fig = go.Figure(go.Bar(
            x=shap_df['SHAP Value'],
            y=shap_df['Feature'],
            orientation='h',
            marker_color=[
                '#EF553B' if v > 0 else '#636EFA'
                for v in shap_df['SHAP Value']
            ]
        ))
        fig.update_layout(
            title="Top 10 Feature Contributions (SHAP)",
            xaxis_title="SHAP Value (→ fraud | ← legitimate)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Red bars push the prediction toward FRAUD, "
            "blue bars push toward LEGITIMATE."
        )


def _csv_upload(model, feature_names, threshold):
    """Handle CSV batch upload."""

    st.subheader("Upload Transactions CSV")
    uploaded = st.file_uploader("Choose a CSV file", type=['csv'])

    if uploaded:
        batch_df = pd.read_csv(uploaded)
        st.write(f"Loaded {len(batch_df)} transactions")

        # Check columns match
        missing = set(feature_names) - set(batch_df.columns)
        if missing:
            st.error(
                f"Missing columns: {missing}. "
                f"CSV must contain all {len(feature_names)} features."
            )
            return

        probas = model.predict_proba(batch_df[feature_names])[:, 1]
        batch_df['Fraud_Probability'] = probas
        batch_df['Risk'] = pd.cut(
            probas, bins=[0, 0.3, 0.7, 1.0],
            labels=['🟢 Low', '🟡 Medium', '🔴 High']
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Transactions", len(batch_df))
        col2.metric("Flagged as Fraud", f"{(probas >= threshold).sum()}")
        col3.metric("Flag Rate", f"{(probas >= threshold).mean():.1%}")

        st.dataframe(
            batch_df[['Amount', 'Fraud_Probability', 'Risk']]
            .sort_values('Fraud_Probability', ascending=False),
            use_container_width=True
        )


def _live_simulation(model, feature_names, threshold):
    """Handle live transaction simulation."""

    st.subheader("▶ Live Transaction Stream")
    st.caption("Simulating real-time transaction monitoring")

    if st.button("Start Simulation", type="primary"):
        X_test = load_simulation_sample()
        progress = st.progress(0)
        results_container = st.empty()
        results = []

        for i in range(20):
            sample = X_test.sample(1)
            proba = model.predict_proba(sample)[0][1]
            status = "🔴 FRAUD" if proba >= threshold else "🟢 LEGIT"

            results.append({
                'TX #': i + 1,
                'Amount': f"€{abs(sample['Amount'].values[0]):.2f}",
                'Probability': f"{proba:.1%}",
                'Status': status
            })

            with results_container.container():
                st.dataframe(
                    pd.DataFrame(results), use_container_width=True
                )

            progress.progress((i + 1) / 20)
            time.sleep(0.3)

        fraud_count = sum(1 for r in results if 'FRAUD' in r['Status'])
        st.metric("Fraud Detected", f"{fraud_count} / 20")