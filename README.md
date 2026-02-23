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

---

## Dataset Content

The dataset contains credit card transactions made by European cardholders in September 2013, collected over a period of two days.

| Attribute | Detail |
|-----------|--------|
| **Total Transactions** | 284,807 |
| **Fraud Cases** | 492 (0.17%) |
| **Legitimate Cases** | 284,315 (99.83%) |
| **Features** | 31 columns |
| **Imbalance Ratio** | ~577:1 |

### Feature Description

| Feature | Type | Description |
|---------|------|-------------|
| Time | Float | Seconds elapsed from first transaction |
| V1 — V28 | Float | PCA-transformed components (anonymised for confidentiality) |
| Amount | Float | Transaction amount in Euros |
| Class | Integer | Target — 0 (legitimate) or 1 (fraud) |

**Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

The dataset was collected and analysed during a research collaboration of Worldline and the Machine Learning Group of ULB (Université Libre de Bruxelles).

**Citation:** Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, and Gianluca Bontempi. *Calibrating Probability with Undersampling for Unbalanced Classification.* In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015.

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

## Rationale to Map Business Requirements to Data Visualisations and ML Tasks

### BR1: Fraud Pattern Analysis

**User Stories:**

### BR2: Supervised Fraud Prediction

**User Stories:**

### BR3: Unsupervised Anomaly Detection

**User Stories:**

---

## ML Business Case

### ML Business Case 1: Supervised Fraud Classification (BR2)

| Element | Detail |


### ML Business Case 2: Unsupervised Anomaly Detection (BR3)

| Element | Detail |

---

## Dashboard Design

### Page 1: Project Summary
- Overview of the dataset and business context
- Three business requirements displayed
- ML terminology glossary in expandable section
- Quick links to key pages

### Page 2: Fraud Pattern Study (BR1)
- Interactive checkbox-controlled visualisations
- 5+ plot types: bar chart, histogram, heatmap, violin, line plot, scatter
- Textual interpretation below each visualisation
- BR1 conclusion summary
- **Answers BR1**

### Page 3: Project Hypotheses
- Four hypotheses with statistical test results
- Side-by-side visualisations and metric displays
- Validated/Not Validated status indicators
- Potential courses of action for each hypothesis

### Page 4: Fraud Detector (BR2)
- Three input modes: Manual Entry, CSV Upload, Live Simulation
- Real-time fraud probability gauge
- SHAP waterfall explanation for each prediction
- Batch processing results with risk colour coding
- **Answers BR2**

### Page 5: Threshold & Cost Analysis
- Interactive threshold slider (0.05 — 0.95)
- Real-time confusion matrix updates
- Business cost calculator (missed fraud cost vs investigation cost)
- Cost-optimal threshold recommendation with visualisation

### Page 6: Anomaly Detection (BR3)
-

### Page 7: ML Pipeline Performance
-

---