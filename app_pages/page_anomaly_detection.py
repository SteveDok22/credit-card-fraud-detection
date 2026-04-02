"""Page 6: Anomaly Detection — answers BR3 with autoencoder results."""
import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
import joblib
import json
from sklearn.metrics import classification_report, f1_score


def page_anomaly_detection():
    """Display the anomaly detection page."""

    st.title("🤖 Unsupervised Anomaly Detection")
    st.info(
        "**Business Requirement 3:** Identify novel fraud patterns "
        "without relying on historical labels."
    )
    st.write("---")

    st.header("How It Works")
    st.write(
        "The autoencoder learns to reconstruct **normal** (legitimate) "
        "transactions. When it encounters a fraudulent transaction, it "
        "produces a **higher reconstruction error** because the fraud "
        "pattern differs from what it learned as 'normal'."
    )
    st.code(
        "Input (38) → [32] → [16] → [8 bottleneck] → [16] → [32] → Output (38)\n"
        "      Encoder                                    Decoder",
        language=None
    )

    st.write("---")

    # Load data
    y_test = pd.read_csv("outputs/v1/y_test.csv").squeeze()
    reconstruction_errors = joblib.load(
        "outputs/v3/reconstruction_errors.pkl"
    )
    with open("outputs/v3/ae_threshold.json") as f:
        ae_data = json.load(f)

    legit_errors = reconstruction_errors[y_test == 0]
    fraud_errors = reconstruction_errors[y_test == 1]

    # Error distribution plot
    st.header("Reconstruction Error Analysis")

    col1, col2 = st.columns(2)
    col1.metric(
        "Legit Mean Error", f"{legit_errors.mean():.6f}"
    )
    col2.metric(
        "Fraud Mean Error", f"{fraud_errors.mean():.6f}"
    )

    error_df = pd.DataFrame({
        'Error': reconstruction_errors,
        'Class': ['Fraud' if c == 1 else 'Legitimate' for c in y_test]
    })

    fig = px.histogram(
        error_df, x='Error', color='Class',
        barmode='overlay', opacity=0.7,
        color_discrete_map={
            'Legitimate': '#636EFA', 'Fraud': '#EF553B'
        },
        title='Reconstruction Error Distribution by Class',
        nbins=100
    )
    st.plotly_chart(fig, use_container_width=True)
    st.write(
        "**Interpretation:** Fraudulent transactions show higher "
        "reconstruction errors because the autoencoder was only trained "
        "on legitimate patterns. The overlap between distributions shows "
        "why this approach works best as a complementary layer."
    )

    st.write("---")

    # Interactive threshold
    st.header("Anomaly Threshold Selection")
    ae_threshold = st.slider(
        "Reconstruction Error Threshold",
        min_value=float(np.percentile(legit_errors, 80)),
        max_value=float(np.percentile(legit_errors, 99.9)),
        value=float(ae_data['threshold']),
        format="%.6f"
    )

    y_pred_ae = (reconstruction_errors > ae_threshold).astype(int)
    report = classification_report(
        y_test, y_pred_ae, output_dict=True,
        target_names=['Legitimate', 'Fraud'], zero_division=0
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Recall (Fraud)", f"{report['Fraud']['recall']:.3f}")
    col2.metric("Precision (Fraud)", f"{report['Fraud']['precision']:.3f}")
    col3.metric("F1 (Fraud)", f"{report['Fraud']['f1-score']:.3f}")

    st.write("---")

    # Comparison with XGBoost
    st.header("Supervised vs Unsupervised")