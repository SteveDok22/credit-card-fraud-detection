"""Page 1: Project Summary — overview, business requirements, terminology."""
import streamlit as st


def page_summary():
    """Display the project summary page."""

    st.title("Credit Card Fraud Detection System")
    st.write("---")

    st.header("Project Overview")
    st.info(
        "**Dataset:** 284,807 credit card transactions collected over 2 days "
        "in September 2013 by European cardholders.\n\n"
        "**Fraud cases:** 492 (0.17% of all transactions)\n\n"
        "**Features:** Time, Amount, and 28 PCA-transformed components "
        "(V1-V28) anonymised for confidentiality.\n\n"
        "**Source:** [Kaggle — Credit Card Fraud Detection]"
        "(https://www.kaggle.com/mlg-ulb/creditcardfraud)"
    )

    st.write("---")
    st.header("Business Requirements")

    st.success(
        "**BR1 — Fraud Pattern Study**\n\n"
        "The client is interested in understanding which transaction "
        "patterns correlate with fraudulent activity, so their analysts "
        "can identify high-risk behaviours."
    )
    st.success(
        "**BR2 — Fraud Prediction**\n\n"
        "The client is interested in predicting whether a given "
        "transaction is fraudulent or legitimate, with explainable "
        "results showing why a transaction was flagged."
    )
    st.success(
        "**BR3 — Anomaly Detection**\n\n"
        "The client wants an unsupervised anomaly detection system "
        "that can identify novel fraud patterns without relying on "
        "historical labels, as a complementary approach."
    )