"""Page 5: Threshold & Cost Analysis — interactive threshold tuning."""
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import joblib
import json
from sklearn.metrics import confusion_matrix, f1_score


def page_threshold_analysis():
    """Display the threshold and cost analysis page."""

    st.title("⚖️ Threshold & Cost Analysis")
    st.write(
        "Explore how the decision threshold affects fraud detection "
        "performance and business costs."
    )
    st.write("---")

    # Load data
    y_test = pd.read_csv("outputs/v1/y_test.csv").squeeze()
    y_proba = joblib.load("outputs/v2/test_probabilities.pkl")

    with open("outputs/v2/optimal_threshold.json") as f:
        default_threshold = json.load(f)['optimal_threshold']