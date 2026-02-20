# Credit Card Fraud Detection System

A Machine Learning powered web application for detecting fraudulent credit card transactions. Built with Python, Streamlit, and XGBoost.

**Live App:** [Deployed Link](https://your-app.onrender.com)

---

## Table of Contents



---

## Project Overview

### Purpose

Credit Card Fraud Detection System is a Machine Learning powered web application that helps financial institutions detect fraudulent credit card transactions. The system combines **supervised classification** (XGBoost) with **unsupervised anomaly detection** (Autoencoder) to provide a comprehensive fraud detection solution with explainable predictions.

### Target Audience

- **Risk Analysts:** Professionals investigating flagged transactions who need explainable fraud predictions
- **Risk Managers:** Decision-makers who need to understand fraud patterns and optimise detection thresholds
- **Data Science Teams:** Teams evaluating ML approaches for fraud detection pipelines
- **FinTech Companies:** Organisations seeking to improve their fraud detection capabilities

### Value Proposition

- Detect fraudulent transactions with high accuracy using dual ML pipelines
- Understand **why** a transaction was flagged through SHAP explainability
- Optimise detection thresholds based on business cost trade-offs
- Identify novel fraud patterns that supervised models might miss



---

## Business Requirements

A fictional FinTech payment processing company, **SecurePay Solutions**, has been experiencing increasing losses due to fraudulent transactions. The Head of Risk Management has requested a data-driven solution to improve their fraud detection capabilities.

**BR1:** The client is interested in understanding which transaction patterns correlate with fraudulent activity, so their analysts can identify high-risk behaviours.

**BR2:** The client is interested in predicting whether a given transaction is fraudulent or legitimate, with explainable results showing why a transaction was flagged.

**BR3:** The client wants an unsupervised anomaly detection system that can identify novel fraud patterns without relying on historical labels, as a complementary approach to the supervised model.

---

# Hypotheses and Validation

### H1: Transaction Amount and Fraud
- **Statement:** Fraudulent transactions have significantly different amount distributions compared to legitimate transactions.
- **Validation:** Mann-Whitney U test + Cohen's d effect size
- **Result:** *To be completed after analysis*

### H2: Temporal Patterns in Fraud
- **Statement:** Fraud occurrence rate varies significantly across different time-of-day periods.
- **Validation:** Chi-squared test on hourly fraud rates
- **Result:** *To be completed after analysis*

### H3: PCA Feature Separation
- **Statement:** At least 3 PCA components show statistically significant separation between fraud and legitimate classes with large effect sizes.
- **Validation:** Mann-Whitney U test per feature + ranking by Cohen's d
- **Result:** *To be completed after analysis*

### H4: Model Performance Threshold
- **Statement:** An optimised ensemble classifier can achieve F1 >= 0.80 on the fraud class while maintaining Precision >= 0.75, meeting the business requirement for automated fraud screening.
- **Validation:** Evaluation metrics on holdout test set + comparison with unsupervised baseline
- **Result:** *To be completed after model training*

---


## ML Business Case

### ML Business Case 1: Supervised Fraud Classification (BR2)

| Element | Detail |