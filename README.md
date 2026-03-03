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

### CRISP-DM Process

This project follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology:

| Phase | Description | Deliverable |
|-------|-------------|-------------|
| Business Understanding | Define fraud detection requirements | Business Requirements (BR1-BR3) |
| Data Understanding | Explore and visualise transaction data | Notebooks 01-02, Dashboard Page 2 |
| Data Preparation | Clean, engineer features, handle imbalance | Notebooks 03-04 |
| Modelling | Train XGBoost + Autoencoder pipelines | Notebooks 05-06 |
| Evaluation | Validate against business success metrics | Notebook 07, Dashboard Page 7 |
| Deployment | Streamlit dashboard on Render | Live application |

### ML Pipeline Flow

<div align="center">
<img src="docs\images\ml-pipeline-flow.png" alt="ML Pipeline Flow" width="900">
</div>

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

| ID | As a... | I want to... | So that I can... |
|----|---------|--------------|------------------|
| 1.1 | Risk Analyst | See the distribution of transaction amounts for fraud vs legitimate | Understand typical fraud behaviour |
| 1.2 | Risk Analyst | See which features correlate most strongly with fraud | Identify key risk indicators |
| 1.3 | Risk Analyst | See temporal patterns in fraud occurrence | Allocate monitoring resources effectively |
| 1.4 | Risk Analyst | Compare PCA feature distributions between classes | Understand which signals separate fraud |

**Data Visualisation Tasks:**
- Class distribution bar chart
- Amount distribution histograms with box plot marginals
- Correlation heatmap of top features vs fraud class
- Violin plots of top discriminating PCA components
- Fraud rate line plot by hour of day
- 2D scatter plot of top separating features

### BR2: Supervised Fraud Prediction

**User Stories:**

| ID | As a... | I want to... | So that I can... |
|----|---------|--------------|------------------|
| 2.1 | Fraud Analyst | Input transaction details and get a fraud probability | Make quick decisions on flagged transactions |
| 2.2 | Fraud Analyst | See why a transaction was flagged (SHAP explanation) | Explain decisions to customers and stakeholders |
| 2.3 | Team Lead | Upload a batch of transactions and see which are flagged | Prioritise the review queue |
| 2.4 | Team Lead | Adjust the decision threshold based on cost trade-offs | Balance missed fraud vs false alarms |

**ML Task:** Binary classification using XGBoost with SMOTE oversampling and SHAP explainability.

### BR3: Unsupervised Anomaly Detection

**User Stories:**

| ID | As a... | I want to... | So that I can... |
|----|---------|--------------|------------------|
| 3.1 | Risk Manager | Detect unusual transactions without relying on historical fraud labels | Catch novel fraud types not seen before |
| 3.2 | Risk Manager | Compare supervised and unsupervised approaches | Understand the value each brings to detection |

**ML Task:** Autoencoder-based anomaly detection trained on legitimate transactions only.

---

## ML Business Case

### ML Business Case 1: Supervised Fraud Classification (BR2)

| Element | Detail |
|---------|--------|
| **Aim** | Build a binary classifier to predict fraudulent vs legitimate transactions with explainable predictions |
| **Learning Method** | Supervised learning — binary classification using gradient boosting (XGBoost) with SMOTE oversampling for class imbalance (fraud = ~0.17% of transactions) |
| **Ideal Outcome** | Flag fraudulent transactions for review with high recall while keeping false positives manageable. Each prediction includes SHAP-based feature contribution explanation |
| **Model Output** | Fraud probability (0-1) per transaction + SHAP waterfall showing top contributing features. Configurable decision threshold optimised for business cost trade-off |
| **Success Metrics** | F1 >= 0.80 on fraud class (primary) · Precision >= 0.75 · Recall >= 0.75 · AUC-ROC >= 0.95 |
| **Failure Condition** | F1 < 0.60 or Recall < 0.50 |
| **Training Data** | 284,807 transactions, 30 features + 8 engineered features. 80/20 stratified split |

### ML Business Case 2: Unsupervised Anomaly Detection (BR3)

| Element | Detail |
|---------|--------|
| **Aim** | Build an autoencoder-based anomaly detection system that identifies unusual transaction patterns without relying on fraud labels |
| **Learning Method** | Unsupervised learning — autoencoder neural network trained on legitimate transactions only. High reconstruction error indicates anomalous (potentially fraudulent) transactions |
| **Ideal Outcome** | Detect novel fraud patterns that the supervised model might miss because they were not present in the training labels |
| **Model Output** | Reconstruction error score per transaction. Threshold set at chosen percentile of training reconstruction errors |
| **Success Metrics** | Recall >= 0.60 on known fraud cases (without having seen labels) · Precision >= 0.10 (acceptable for anomaly detection where flagged items go to manual review) |
| **Failure Condition** | Recall < 0.30 (misses most fraud entirely) |
| **Training Data** | Only legitimate transactions from training set (~227,451 transactions). Evaluated against full test set |

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
- Autoencoder architecture explanation
- Reconstruction error distribution by class
- Interactive anomaly threshold selection
- Supervised vs unsupervised comparison
- **Answers BR3**

### Page 7: ML Pipeline Performance
- Algorithm comparison table (Random Forest vs Gradient Boosting vs XGBoost)
- Confusion matrices for train and test sets
- ROC Curve and Precision-Recall Curve
- SHAP global feature importance plot
- Hyperparameter tuning details with rationale
- Clear model success/failure statement

---

## Features

### Existing Features

#### F1: Interactive Fraud Pattern Study

<div align="center">
<img src="docs/screenshots/fraud-study.png" alt="Fraud Study Screenshot" width="700">
</div>

---

#### F2: Fraud Detector with SHAP Explainability

<div align="center">
<img src="docs/screenshots/fraud-detector.png" alt="Fraud Detector Screenshot" width="700">
</div>

---

#### F3: Threshold & Cost Analysis

<div align="center">
<img src="docs/screenshots/threshold-analysis.png" alt="Threshold Analysis Screenshot" width="700">
</div>

---

#### F4: Anomaly Detection System

<div align="center">
<img src="docs/screenshots/anomaly-detection.png" alt="Anomaly Detection Screenshot" width="700">
</div>

---

#### F5: ML Pipeline Performance Dashboard

<div align="center">
<img src="docs/screenshots/ml-performance.png" alt="ML Performance Screenshot" width="700">
</div>

---

#### F6: Statistical Hypothesis Validation

<div align="center">
<img src="docs/screenshots/hypotheses.png" alt="Hypotheses Screenshot" width="700">
</div>

---

#### F7: Responsive Navigation

- Sidebar navigation with 7 dashboard pages
- Clear page titles and business requirement labels
- Consistent layout across all pages

---

## Technologies Used

### Languages

- Python 3.11

### Frameworks & Libraries

#### Machine Learning & Data Science

| Library | Version | Purpose |
|---------|---------|---------|
| pandas | 2.1.0 | Data manipulation and analysis |
| numpy | 1.24.0 | Numerical computing |
| scikit-learn | 1.3.0 | ML preprocessing, evaluation, pipelines |
| xgboost | 2.0.0 | Gradient boosting classifier |
| imbalanced-learn | 0.11.0 | SMOTE oversampling for class imbalance |
| shap | 0.42.0 | Model explainability (SHAP values) |
| tensorflow-cpu | 2.13.0 | Autoencoder neural network |
| scipy | 1.11.0 | Statistical hypothesis testing |

#### Data Visualisation

| Library | Version | Purpose |
|---------|---------|---------|


#### Web Application

| Library | Version | Purpose |
|---------|---------|---------|
| streamlit | 1.28.0 | Dashboard web application |
| joblib | 1.3.0 | Model serialisation and loading |

#### Data Collection

| Library | Version | Purpose |
|---------|---------|---------|
| kaggle | 1.6.0 | Dataset download from Kaggle API |

### Tools & Services

| Tool | Purpose |
|------|---------|
| Git | Version control |
| GitHub | Repository hosting |
| GitHub Projects | Agile project management |
| Render | Cloud deployment |
| Jupyter Notebook | Data analysis and modelling |
| VS Code | Code editor |

---

## Agile Methodology

### GitHub Projects Board

This project was developed using Agile methodology with GitHub Projects as the management tool.

**Board Link:** [Fraud Detection Project Board](https://github.com/users/SteveDok22/projects/XX)

### Sprint Structure

<div align="center">
<img src="docs\images\sprint-structure.png" alt="Sprint Structure" width="900">
</div>

### User Stories

All user stories were created as GitHub Issues with:
- Acceptance Criteria
- Tasks checklist
- Labels (Must Have, Should Have, Could Have)
- Linked to Business Requirements

---

## Testing

### Manual Testing

| Feature | Action | Expected | Result |
|---------|--------|----------|--------|

### User Story Testing

| User Story | Acceptance Criteria Met | Evidence |
|------------|----------------------|---------|

### Validator Testing

#### Python (PEP8 / Flake8)


<div align="center">

| File | Lines | Issues | Status |
|------|-------|--------|--------|


</div>

#### Jupyter Notebooks

All notebooks follow PEP8 standards and include Objectives/Inputs/Outputs headers.

---

### Bugs

### Resolved Issues

*Bugs will be documented as they are encountered during development.*


### Known Issues

| Issue | Description | Impact | Workaround |
|-------|-------------|--------|------------|
| — | — | — | — |

---

## Deployment

### Render Deployment

The application is deployed on Render.

**Live URL:** https://-app.onrender.com

#### Deployment Steps

1. **Create Render Account**
   - Sign up at [Render](https://render.com)

2. **Create New Web Service**
   - Click "New" → "Web Service"
   - Connect to GitHub repository

3. **Configure Build Settings**

   | Setting | Value |
   |---------|-------|
   | **Build Command** | `pip install -r requirements.txt` |
   | **Start Command** | `streamlit run app.py --server.port $PORT --server.headless true` |
   | **Python Version** | 3.11.5 |

4. **Deploy**
   - Click "Create Web Service"
   - Wait for build to complete

---

### Local Development

#### Prerequisites
- Python 3.11+
- Git
- Kaggle account (for dataset download)

#### Setup Steps

1. **Clone Repository**
```bash
git clone https://github.com/SteveDok22/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate    # Mac/Linux
venv\Scripts\activate       # Windows
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Download Dataset**
   - Place `kaggle.json` in project root
   - Run Notebook 01 to download the dataset

5. **Run Application**
```bash
streamlit run app.py
```

---

### Forking the Repository

1. Go to [GitHub Repository](https://github.com/SteveDok22/credit-card-fraud-detection)
2. Click "Fork" button (top right)
3. Clone forked repository

### Cloning the Repository

```bash
git clone https://github.com/YOUR-USERNAME/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

---

## Credits

### Dataset
- [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) — Machine Learning Group of ULB (Université Libre de Bruxelles) in collaboration with Worldline

### Documentation & Tutorials

| Resource | Usage |
|----------|-------|


### Code References
- Code Institute — CRISP-DM methodology and project structure guidance
- [Kaggle Notebooks](https://www.kaggle.com/mlg-ulb/creditcardfraud/code) — General approach inspiration (all code written independently)

### Acknowledgements
- **Code Institute** — For the learning materials and assessment framework

---
