"""
Main entry point for the Credit Card Fraud Detection Dashboard.
Run with: streamlit run app.py
"""
import streamlit as st

st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

from app_pages.page_summary import page_summary
from app_pages.page_fraud_study import page_fraud_study
from app_pages.page_hypotheses import page_hypotheses
from app_pages.page_detector import page_detector
from app_pages.page_threshold_analysis import page_threshold_analysis
from app_pages.page_anomaly_detection import page_anomaly_detection
from app_pages.page_ml_performance import page_ml_performance


pages = {
    "📋 Project Summary": page_summary,
    "🔎 Fraud Pattern Study": page_fraud_study,
    "🧪 Project Hypotheses": page_hypotheses,
    "🎯 Fraud Detector": page_detector,
    "⚖️ Threshold & Cost Analysis": page_threshold_analysis,
    "🤖 Anomaly Detection": page_anomaly_detection,
    "📊 ML Pipeline Performance": page_ml_performance,
}


def main():
    """Run the dashboard application."""

    st.sidebar.title("Navigation")
    st.sidebar.write("---")
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    st.sidebar.write("---")
    st.sidebar.info(
        "**Credit Card Fraud Detection**\n\n"
        "ML-powered system for identifying fraudulent transactions "
        "using supervised and unsupervised approaches.\n\n"
        "Built for **SecurePay Solutions**"
    )

    pages[selection]()

if __name__ == "__main__":
    main()