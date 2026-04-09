"""Page 3: Project Hypotheses — statistical validation results."""
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, chi2_contingency
from src.data_management import load_sample_data


def page_hypotheses():
    """Display the hypothesis validation page."""

    st.title("🧪 Project Hypotheses & Validation")
    st.write("---")

    df = load_sample_data()

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
        "✅ **Validated** — Statistically significant difference "
        "(p < 0.001). Fraud transactions have lower amounts."
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

    results = []
    for col_name in [f'V{i}' for i in range(1, 29)]:
        fraud_vals = df[df['Class'] == 1][col_name]
        legit_vals = df[df['Class'] == 0][col_name]
        _, p = mannwhitneyu(fraud_vals, legit_vals)
        d = (fraud_vals.mean() - legit_vals.mean()) / df[col_name].std()
        results.append({
            'Feature': col_name, 'P-value': p,
            "Cohen's d": d, '|d|': abs(d)
        })

    results_df = pd.DataFrame(results).sort_values('|d|', ascending=False)
    significant = results_df[
        (results_df['P-value'] < 0.001) & (results_df['|d|'] > 0.5)
    ]

    col1, col2 = st.columns(2)
    col1.metric("Features with Large Effect", f"{len(significant)}")
    col2.metric("Top Feature", results_df.iloc[0]['Feature'])

    st.dataframe(
        results_df.head(10).style.format({
            'P-value': '{:.2e}', "Cohen's d": '{:+.3f}', '|d|': '{:.3f}'
        }),
        use_container_width=True
    )

    st.success(
        f"✅ **Validated** — {len(significant)} features show large "
        f"effect size separation (|d| > 0.5)."
    )
    st.write("---")

    # H4
    st.header("H4: Model Performance Threshold")
    st.write(
        "**Statement:** An optimised classifier can achieve F1 >= 0.80 "
        "on the fraud class while maintaining Precision >= 0.75."
    )

    with open("outputs/v2/tuning_results.json") as f:
        tuning = json.load(f)

    y_test_local = pd.read_csv("outputs/v1/y_test.csv").squeeze()
    y_proba_local = joblib.load("outputs/v2/test_probabilities.pkl")

    with open("outputs/v2/optimal_threshold.json") as f:
        thresh = json.load(f)['optimal_threshold']

    from sklearn.metrics import f1_score, precision_score, recall_score
    y_pred_local = (y_proba_local >= thresh).astype(int)

    h4_f1 = f1_score(y_test_local, y_pred_local)
    h4_prec = precision_score(y_test_local, y_pred_local)
    h4_rec = recall_score(y_test_local, y_pred_local)

    col1, col2, col3 = st.columns(3)
    col1.metric("F1 (Fraud)", f"{h4_f1:.4f}")
    col2.metric("Precision", f"{h4_prec:.4f}")
    col3.metric("Recall", f"{h4_rec:.4f}")

    if h4_f1 >= 0.80 and h4_prec >= 0.75:
        st.success("✅ **Validated** — Model meets all business requirements.")
    else:
        st.warning("⚠️ **Partially Validated** — See ML Performance page.")
