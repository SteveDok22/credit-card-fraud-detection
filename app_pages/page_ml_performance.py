"""Page 7: ML Pipeline Performance — evaluation metrics and comparison."""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import joblib
import json
from src.data_management import load_model, load_train_confusion_matrix
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score
)


def page_ml_performance():
    """Display the ML pipeline performance page."""

    st.title("📊 ML Pipeline Performance")

    # Load data
    cm_train_data = load_train_confusion_matrix()
    y_test = pd.read_csv("outputs/v1/y_test.csv").squeeze()
    model = joblib.load("outputs/v2/fraud_model_optimized.pkl")
    y_test_proba = joblib.load("outputs/v2/test_probabilities.pkl")

    with open("outputs/v2/optimal_threshold.json") as f:
        threshold = json.load(f)['optimal_threshold']
    with open("outputs/v2/tuning_results.json") as f:
        tuning = json.load(f)

    y_test_pred = (y_test_proba >= threshold).astype(int)
    test_report = classification_report(
        y_test, y_test_pred, output_dict=True,
        target_names=['Legitimate', 'Fraud']
    )

    # Success statement (LO4: 4.2)
    fraud_f1 = test_report['Fraud']['f1-score']
    if fraud_f1 >= 0.80:
        st.success(
            f"✅ **The ML pipeline meets the business requirements.** "
            f"The XGBoost model achieves F1 = {fraud_f1:.4f} on the "
            f"fraud class, exceeding the target of F1 ≥ 0.80."
        )
    else:
        st.warning(
            f"⚠️ The model achieves F1 = {fraud_f1:.4f}. "
            f"See details below."
        )

    st.write("---")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Confusion Matrix", "ROC & PR Curves",
        "Feature Importance", "Hyperparameters"
    ])

    with tab1:
        st.header("Confusion Matrices")
        col1, col2 = st.columns(2)

        # Train set
        with col1:
            st.subheader("Train Set")
            cm_train = np.array(cm_train_data)
            fig = go.Figure(data=go.Heatmap(
                z=cm_train, colorscale='Blues',
                text=[[f"{v:,}" for v in row] for row in cm_train],
                texttemplate="%{text}",
                textfont={"size": 14}, showscale=False
            ))
            fig.update_layout(
                xaxis_title='Predicted', yaxis_title='Actual',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

        # Test set
        with col2:
            st.subheader("Test Set")
            cm_test = confusion_matrix(y_test, y_test_pred)
            fig = go.Figure(data=go.Heatmap(
                z=cm_test, colorscale='Reds',
                text=[[f"{v:,}" for v in row] for row in cm_test],
                texttemplate="%{text}",
                textfont={"size": 14}, showscale=False
            ))
            fig.update_layout(
                xaxis_title='Predicted', yaxis_title='Actual',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

        # Classification report
        st.subheader("Test Set Classification Report")
        report_df = pd.DataFrame(test_report).transpose()
        st.dataframe(
            report_df.style.format('{:.4f}'),
            use_container_width=True
        )

    with tab2:
        st.header("ROC and Precision-Recall Curves")
        col1, col2 = st.columns(2)

        with col1:
            fpr, tpr, _ = roc_curve(y_test, y_test_proba)
            roc_auc = auc(fpr, tpr)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                name=f'XGBoost (AUC = {roc_auc:.3f})',
                line=dict(color='#636EFA', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], name='Random',
                line=dict(dash='dash', color='grey')
            ))
            fig.update_layout(
                title='ROC Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            prec_vals, rec_vals, _ = precision_recall_curve(
                y_test, y_test_proba
            )
            avg_prec = average_precision_score(y_test, y_test_proba)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rec_vals, y=prec_vals,
                name=f'PR Curve (AP = {avg_prec:.3f})',
                line=dict(color='#EF553B', width=2)
            ))
            fig.update_layout(
                title='Precision-Recall Curve',
                xaxis_title='Recall',
                yaxis_title='Precision',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("Feature Importance")

        # SHAP plot
        st.subheader("SHAP Global Feature Importance")
        st.image("outputs/v2/shap_summary.png")

        # XGBoost native importance
        st.subheader("XGBoost Feature Importance")
        fi = pd.read_csv("outputs/v2/feature_importance.csv")
        fig = px.bar(
            fi.head(15), x='importance', y='feature',
            orientation='h',
            title='Top 15 Features',
            color='importance', color_continuous_scale='Blues'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.header("Hyperparameter Tuning")
        st.write(
            f"**Method:** RandomizedSearchCV "
            f"({tuning['n_iterations']} iterations, "
            f"{tuning['cv_folds']}-fold stratified CV)"
        )
        st.write(f"**Best CV F1:** {tuning['best_score']:.4f}")

        st.subheader("Best Parameters")
        st.json(tuning['best_params'])

        st.subheader("Tuning Rationale")
        rationale = pd.DataFrame({
            'Parameter': [
                'n_estimators', 'max_depth', 'learning_rate',
                'min_child_weight', 'subsample', 'colsample_bytree',
                'gamma', 'scale_pos_weight'
            ],
            'Values Tested': [
                '100, 300, 500', '3, 5, 7', '0.01, 0.05, 0.1',
                '1, 3, 5', '0.7, 0.8, 0.9', '0.7, 0.8, 0.9',
                '0, 0.1, 0.3', '1, 10, 50'
            ],
            'Rationale': [
                'Balance ensemble strength vs computation',
                'Shallow trees prevent overfitting on rare fraud',
                'Lower rates produce more robust models',
                'Conservative splits for rare class patterns',
                'Row subsampling reduces variance',
                'Prevents over-reliance on dominant features',
                'Regularisation through minimum loss reduction',
                'Additional class imbalance handling'
            ]
        })
        st.dataframe(rationale, use_container_width=True)