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

    # Interactive threshold slider
    threshold = st.slider(
        "Decision Threshold",
        min_value=0.05, max_value=0.95,
        value=float(default_threshold), step=0.01,
        help="Lower = catch more fraud (higher recall) but more false alarms"
    )

    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precision", f"{precision:.3f}")
    col2.metric("Recall", f"{recall:.3f}")
    col3.metric("F1-Score", f"{f1:.3f}")
    col4.metric("Threshold", f"{threshold:.2f}")

    st.write("---")

    # Confusion Matrix + Cost Calculator
    col_cm, col_costs = st.columns(2)

    with col_cm:
        st.subheader("Confusion Matrix")
        fig = go.Figure(data=go.Heatmap(
            z=[[tn, fp], [fn, tp]],
            x=['Pred Legit', 'Pred Fraud'],
            y=['Actual Legit', 'Actual Fraud'],
            colorscale='RdBu',
            text=[
                [f"TN: {tn:,}", f"FP: {fp:,}"],
                [f"FN: {fn:,}", f"TP: {tp:,}"]
            ],
            texttemplate="%{text}",
            textfont={"size": 16},
            showscale=False
        ))
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col_costs:
        st.subheader("💰 Business Cost Calculator")
        cost_fn = st.number_input(
            "Cost per missed fraud ($)", value=5000, step=500,
            help="Average loss when fraud is not caught"
        )
        cost_fp = st.number_input(
            "Cost per false alarm ($)", value=50, step=10,
            help="Cost of investigating a legitimate transaction"
        )

        total_cost = (fn * cost_fn) + (fp * cost_fp)
        fraud_losses = fn * cost_fn
        investigation_costs = fp * cost_fp

        st.metric("Total Business Cost", f"${total_cost:,.0f}")
        st.write(
            f"- Missed fraud losses: **${fraud_losses:,.0f}** ({fn} cases)"
        )
        st.write(
            f"- Investigation costs: **${investigation_costs:,.0f}** "
            f"({fp} cases)"
        )

    st.write("---")

    # Threshold optimisation curve
    st.subheader("Optimal Threshold Analysis")

    thresholds = np.arange(0.05, 0.95, 0.01)
    costs = []
    f1s = []
    for t in thresholds:
        y_t = (y_proba >= t).astype(int)
        tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_test, y_t).ravel()
        costs.append(fn_t * cost_fn + fp_t * cost_fp)
        f1s.append(f1_score(y_test, y_t, zero_division=0))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=thresholds, y=costs, name='Business Cost ($)',
        line=dict(color='#636EFA')
    ))
    fig.add_trace(go.Scatter(
        x=thresholds, y=np.array(f1s) * max(costs), name='F1 (scaled)',
        line=dict(color='#EF553B', dash='dot')
    ))
    fig.add_vline(
        x=threshold, line_dash='dash', line_color='green',
        annotation_text=f'Current: {threshold:.2f}'
    )
    fig.update_layout(
        title='Threshold vs Business Cost',
        xaxis_title='Decision Threshold',
        yaxis_title='Cost ($)',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    optimal_cost_idx = np.argmin(costs)
    st.info(
        f"💡 **Cost-optimal threshold:** {thresholds[optimal_cost_idx]:.2f} "
        f"(Total cost: ${costs[optimal_cost_idx]:,.0f})"
    )
