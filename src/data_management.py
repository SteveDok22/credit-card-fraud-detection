"""
Functions for loading data and ML artifacts across the dashboard.
"""
import pandas as pd
import joblib
import json
import streamlit as st


@st.cache_data
def load_sample_data():
    """Load the sampled dataset for visualisations."""
    return pd.read_csv("outputs/dashboard/data_sample.csv")


@st.cache_data
def load_dataset_stats():
    """Load pre-computed dataset statistics."""
    with open("outputs/dashboard/dataset_stats.json") as f:
        return json.load(f)


@st.cache_data
def load_test_labels():
    """Load test set labels."""
    return pd.read_csv("outputs/v1/y_test.csv").squeeze()


@st.cache_data
def load_feature_names():
    """Load feature names list."""
    with open("outputs/dashboard/feature_names.json") as f:
        return json.load(f)


@st.cache_data
def load_simulation_sample():
    """Load small test sample for live simulation."""
    return pd.read_csv("outputs/dashboard/simulation_sample.csv")


@st.cache_data
def load_train_confusion_matrix():
    """Load pre-computed train set confusion matrix."""
    with open("outputs/dashboard/train_confusion_matrix.json") as f:
        return json.load(f)


@st.cache_resource
def load_model():
    """Load the trained XGBoost model."""
    return joblib.load("outputs/v2/fraud_model_optimized.pkl")


@st.cache_resource
def load_shap_explainer():
    """Load the SHAP explainer."""
    return joblib.load("outputs/v2/shap_explainer.pkl")