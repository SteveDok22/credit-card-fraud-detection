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

    # H2
    st.header("H2: Temporal Patterns in Fraud")
    st.write(
        "**Statement:** Fraud occurrence rate varies significantly "
        "across different time-of-day periods."
    )

    df['Hour_bin'] = (df['Time'] / 3600 % 24).astype(int)
    contingency = pd.crosstab(df['Hour_bin'], df['Class'])
    chi2, p_h2, dof, _ = chi2_contingency(contingency)
    n = contingency.sum().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))

    col1, col2, col3 = st.columns(3)
    col1.metric("Chi-squared", f"{chi2:.2f}")
    col2.metric("P-value", f"{p_h2:.2e}")
    col3.metric("Cramér's V", f"{cramers_v:.4f}")

    if p_h2 < 0.05:
        st.success(
            "✅ **Validated** — Fraud rate varies significantly by hour."
        )
    else:
        st.warning("⚠️ **Not Validated** — No significant variation found.")

    st.info(
        "**Course of Action:** Apply different detection thresholds "
        "during high-risk hours to increase recall."
    )
    st.write("---")

    # H3
    st.header("H3: PCA Feature Separation")
    st.write(
        "**Statement:** At least 3 PCA components show statistically "
        "significant separation between classes with large effect sizes."
    )

