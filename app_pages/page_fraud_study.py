"""Page 2: Fraud Pattern Study — answers BR1 with visualisations."""
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from src.data_management import load_raw_data


def page_fraud_study():
    """Display the fraud pattern study page."""

    st.title("🔎 Fraud Pattern Study")
    st.info(
        "**Business Requirement 1:** Understand which transaction "
        "patterns correlate with fraudulent activity."
    )
    st.write("---")

    df = load_raw_data()

    # Plot 1: Class Distribution
    if st.checkbox("Show Class Distribution"):
        fig = px.bar(
            x=['Legitimate', 'Fraud'],
            y=[len(df[df['Class'] == 0]), len(df[df['Class'] == 1])],
            color=['Legitimate', 'Fraud'],
            color_discrete_map={
                'Legitimate': '#636EFA', 'Fraud': '#EF553B'
            },
            labels={'x': 'Transaction Class', 'y': 'Count'},
            title='Transaction Class Distribution'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.write(
            "**Interpretation:** The dataset is extremely imbalanced — "
            "284,315 legitimate transactions (99.83%) vs 492 fraud cases "
            "(0.17%). This 577:1 ratio necessitates specialised techniques "
            "like SMOTE oversampling during the modelling phase."
        )
        
    # Plot 2: Amount Distribution
    if st.checkbox("Show Amount Distribution"):
        fig = px.histogram(
            df, x='Amount', color='Class',
            marginal='box', barmode='overlay',
            color_discrete_map={0: '#636EFA', 1: '#EF553B'},
            opacity=0.7,
            title='Transaction Amount Distribution by Class',
            labels={'Class': 'Transaction Class'}
        )
        fig.update_layout(
            xaxis_title='Amount (€)', yaxis_title='Count'
        )
