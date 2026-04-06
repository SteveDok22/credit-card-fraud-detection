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

from app_pages.page_summary import page_summary  # noqa: E402
from app_pages.page_fraud_study import page_fraud_study  # noqa: E402
from app_pages.page_hypotheses import page_hypotheses  # noqa: E402
from app_pages.page_detector import page_detector  # noqa: E402
from app_pages.page_threshold_analysis import page_threshold_analysis  # noqa: E402
from app_pages.page_anomaly_detection import page_anomaly_detection  # noqa: E402
from app_pages.page_ml_performance import page_ml_performance  # noqa: E402


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
