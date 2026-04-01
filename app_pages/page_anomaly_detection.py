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