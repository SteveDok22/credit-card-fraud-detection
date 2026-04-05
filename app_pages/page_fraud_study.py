"""Page 2: Fraud Pattern Study — answers BR1 with visualisations."""
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
ffrom src.data_management import load_sample_data, load_dataset_stats


def page_fraud_study():
    """Display the fraud pattern study page."""

    st.title("🔎 Fraud Pattern Study")
    st.info(
        "**Business Requirement 1:** Understand which transaction "
        "patterns correlate with fraudulent activity."
    )
    st.write("---")

    df = load_sample_data()
    stats = load_dataset_stats()

    # Plot 1: Class Distribution
    if st.checkbox("Show Class Distribution"):
        fig = px.bar(
            x=['Legitimate', 'Fraud'],
            y=[stats['legit_count'], stats['fraud_count']],
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
        st.plotly_chart(fig, use_container_width=True)
        st.write(
            "**Interpretation:** Fraudulent transactions cluster at lower "
            "amounts compared to legitimate ones. This suggests fraudsters "
            "may deliberately keep amounts low to avoid triggering existing "
            "rule-based detection systems."
        )

    # Plot 3: Correlation Heatmap
    if st.checkbox("Show Feature Correlation Heatmap"):
        correlations = df.corr()['Class'].drop('Class').abs()
        top_features = correlations.sort_values(
            ascending=False
        ).head(12).index.tolist() + ['Class']

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            df[top_features].corr(), annot=True,
            cmap='RdBu_r', center=0, fmt='.2f', ax=ax
        )
        ax.set_title('Correlation Heatmap — Top 12 Features')
        st.pyplot(fig)
        st.write(
            "**Interpretation:** V14, V12, and V10 show the strongest "
            "correlations with the fraud class. These features will be "
            "the primary drivers in the ML model."
        )

    # Plot 4: Violin Plots
    if st.checkbox("Show Top Discriminating Features"):
        feature = st.selectbox(
            "Select Feature",
            ['V14', 'V12', 'V10', 'V17', 'V11', 'V4']
        )
        sample = pd.concat([
            df[df['Class'] == 0].sample(5000, random_state=42),
            df[df['Class'] == 1]
        ])
        fig = px.violin(
            sample, y=feature, color='Class', box=True,
            color_discrete_map={0: '#636EFA', 1: '#EF553B'},
            title=f'{feature} Distribution by Class'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.write(
            f"**Interpretation:** {feature} shows distinct distribution "
            f"patterns between fraud and legitimate transactions, "
            f"confirming its value as a discriminating feature."
        )

    # Plot 5: Fraud Rate Over Time
    if st.checkbox("Show Fraud Rate by Hour"):
        df['Hour'] = (df['Time'] / 3600) % 24
        hourly = df.groupby(
            df['Hour'].astype(int)
        )['Class'].mean().reset_index()
        hourly.columns = ['Hour', 'Fraud_Rate']

        fig = px.line(
            hourly, x='Hour', y='Fraud_Rate', markers=True,
            title='Fraud Rate by Hour of Day'
        )
        fig.add_hline(
            y=df['Class'].mean(), line_dash='dash',
            annotation_text='Average fraud rate'
        )
        fig.update_layout(xaxis=dict(dtick=1))
        st.plotly_chart(fig, use_container_width=True)
        st.write(
            "**Interpretation:** Fraud rate varies throughout the day, "
            "with certain hours showing elevated risk. This temporal "
            "pattern is captured by the engineered Hour feature."
        )

    # Plot 6: 2D Scatter
    if st.checkbox("Show Feature Space (V14 vs V12)"):
        sample = pd.concat([
            df[df['Class'] == 0].sample(3000, random_state=42),
            df[df['Class'] == 1]
        ])
        fig = px.scatter(
            sample, x='V14', y='V12', color='Class',
            color_discrete_map={0: '#636EFA', 1: '#EF553B'},
            opacity=0.5,
            title='2D Feature Space: V14 vs V12'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.write(
            "**Interpretation:** Fraud transactions form visible clusters "
            "in specific regions of the V14-V12 space, suggesting that "
            "a non-linear classifier can learn effective decision boundaries."
        )

    st.write("---")
    st.success(
        "**BR1 Conclusion:** Fraudulent transactions exhibit distinct "
        "patterns in amount distribution, temporal occurrence, and PCA "
        "component values. Key discriminating features include V14, V12, "
        "V10, and transaction Amount."
    )