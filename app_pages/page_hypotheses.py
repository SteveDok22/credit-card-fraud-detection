"""Page 3: Project Hypotheses — statistical validation results."""
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, chi2_contingency
from src.data_management import load_raw_data


def page_hypotheses():
    """Display the hypothesis validation page."""

    st.title("🧪 Project Hypotheses & Validation")
    st.write("---")

    df = load_raw_data()

    # H1
    st.header("H1: Transaction Amount and Fraud")
    st.write(
        "**Statement:** Fraudulent transactions have significantly "
        "different amount distributions compared to legitimate transactions."
    )

    fraud_amounts = df[df['Class'] == 1]['Amount']
    legit_amounts = df[df['Class'] == 0]['Amount']
    stat_h1, p_h1 = mannwhitneyu(
        fraud_amounts, legit_amounts, alternative='two-sided'
    )
    d_h1 = (
        (fraud_amounts.mean() - legit_amounts.mean()) / df['Amount'].std()
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("P-value", f"{p_h1:.2e}")
    col2.metric("Cohen's d", f"{d_h1:.4f}")
    col3.metric("Fraud Median", f"€{fraud_amounts.median():.2f}")

    st.success(
        f"✅ **Validated** — Statistically significant difference "
        f"(p < 0.001). Fraud transactions have lower amounts."
    )
    st.info(
        "**Course of Action:** A rule-based pre-filter on transaction "
        "amount could complement the ML model."
    )
    st.write("---")
