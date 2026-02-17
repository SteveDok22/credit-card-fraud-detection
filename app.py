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


# -- Page imports will go here as pages are built --
# from app_pages.page_summary import page_summary
# from app_pages.page_fraud_study import page_fraud_study
# from app_pages.page_hypotheses import page_hypotheses
# from app_pages.page_detector import page_detector
# from app_pages.page_threshold_analysis import page_threshold_analysis
# from app_pages.page_anomaly_detection import page_anomaly_detection
# from app_pages.page_ml_performance import page_ml_performance


def main():
    """Run the dashboard application."""

    st.sidebar.title("Navigation")
    st.sidebar.write("---")

    # Placeholder until pages are built
    st.title("Credit Card Fraud Detection System")
    st.info(
        "🚧 **Dashboard under construction.**\n\n"
        "Pages will be added as the ML pipeline is completed."
    )

    st.write("---")
    st.header("Project Status")
    st.write("- ✅ Data Collection (Notebook 01)")
    st.write("- 🔄 Data Visualisation (Notebook 02)")
    st.write("- ⬜ Data Cleaning (Notebook 03)")
    st.write("- ⬜ Feature Engineering (Notebook 04)")
    st.write("- ⬜ XGBoost Modelling (Notebook 05)")
    st.write("- ⬜ Autoencoder (Notebook 06)")
    st.write("- ⬜ Model Comparison (Notebook 07)")


if __name__ == "__main__":
    main()