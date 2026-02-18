"""
Functions for loading data and ML artifacts across the dashboard.
"""
import pandas as pd
import joblib
import streamlit as st


@st.cache_data
def load_raw_data():
    """Load the raw credit card fraud dataset."""
    df = pd.read_csv("data/creditcard.csv")
    return df


@st.cache_data
def load_train_test_data(version="v1"):
    """Load the train/test split from a specific output version."""
    path = f"outputs/{version}"
    X_train = pd.read_csv(f"{path}/X_train.csv")
    X_test = pd.read_csv(f"{path}/X_test.csv")
    y_train = pd.read_csv(f"{path}/y_train.csv").squeeze()
    y_test = pd.read_csv(f"{path}/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


@st.cache_resource
def load_model(version="v2"):
    """Load a trained ML model/pipeline."""
    return joblib.load(f"outputs/{version}/fraud_pipeline_v2.pkl")


@st.cache_resource
def load_shap_explainer(version="v2"):
    """Load the SHAP explainer for the trained model."""
    return joblib.load(f"outputs/{version}/shap_explainer.pkl")