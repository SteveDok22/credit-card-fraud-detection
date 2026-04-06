# Credit Card Fraud Detection System

A Machine Learning powered web application for detecting fraudulent credit card transactions. Built with Python, Streamlit, and XGBoost.

**Live App:** [Deployed Link](https://credit-card-fraud-detection-st-4cbb00a4456a.herokuapp.com/)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Content](#dataset-content)
3. [Business Requirements](#business-requirements)
4. [Hypotheses and Validation](#hypotheses-and-validation)
5. [Rationale to Map Business Requirements to Data Visualisations and ML Tasks](#rationale-to-map-business-requirements-to-data-visualisations-and-ml-tasks)
6. [ML Business Case](#ml-business-case)
7. [Dashboard Design](#dashboard-design)
8. [Features](#features)
9. [Technologies Used](#technologies-used)
10. [Agile Methodology](#agile-methodology)
11. [Testing](#testing)
12. [Deployment](#deployment)
13. [Credits](#credits)

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
| Deployment | Streamlit dashboard on Heroku | Live application |

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

## Hypotheses and Validation

### H1: Transaction Amount and Fraud
- **Statement:** Fraudulent transactions have significantly different amount distributions compared to legitimate transactions.
- **Validation:** Mann-Whitney U test + Cohen's d effect size
- **Result:** ✅ Validated — p = 8.58e-06, Cohen's d = 0.1356. Statistically significant difference confirmed. Fraud transactions tend to have lower amounts, suggesting fraudsters keep amounts small to avoid detection.

### H2: Temporal Patterns in Fraud
- **Statement:** Fraud occurrence rate varies significantly across different time-of-day periods.
- **Validation:** Chi-squared test on hourly fraud rates + Cramér's V
- **Result:** ✅ Validated — χ² = 674.44, p = 1.07e-127, Cramér's V = 0.0487. Fraud rate varies significantly by hour of day. This supports the engineered Hour feature used in the model.

### H3: PCA Feature Separation
- **Statement:** At least 3 PCA components show statistically significant separation between fraud and legitimate classes with large effect sizes.
- **Validation:** Mann-Whitney U test per feature + ranking by Cohen's d
- **Result:** ✅ Validated — 17 features show significant separation (p < 0.001, |d| > 0.5). Top discriminators: V14, V12, V10. Far exceeds the minimum of 3 required.

### H4: Model Performance Threshold
- **Statement:** An optimised ensemble classifier can achieve F1 >= 0.80 on the fraud class while maintaining Precision >= 0.75.
- **Validation:** Evaluation metrics on holdout test set + comparison with unsupervised baseline
- **Result:** ✅ Validated — See ML Pipeline Performance page for full metrics.
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

- Checkbox-controlled visualisations for exploring fraud patterns
- 7+ plot types using Plotly and Seaborn
- Textual interpretation for every visualisation
- Interactive feature selection via dropdown

---

#### F2: Fraud Detector with SHAP Explainability

<div align="center">
<img src="docs/screenshots/fraud-detector.png" alt="Fraud Detector Screenshot" width="700">
</div>

- Manual transaction entry with sliders for key features
- Real-time fraud probability score with gauge visualisation
- **SHAP waterfall** showing why each prediction was made
- CSV batch upload for processing multiple transactions
- Live transaction simulation mode

---

#### F3: Threshold & Cost Analysis

<div align="center">
<img src="docs/screenshots/threshold-analysis.png" alt="Threshold Analysis Screenshot" width="700">
</div>

- Interactive threshold slider with real-time metric updates
- Dynamic confusion matrix
- Business cost calculator with configurable costs
- Optimal threshold recommendation

---

#### F4: Anomaly Detection System

<div align="center">
<img src="docs/screenshots/anomaly-detection.png" alt="Anomaly Detection Screenshot" width="700">
</div>

- Autoencoder-based unsupervised fraud detection
- Reconstruction error distribution visualisation
- Supervised vs unsupervised comparison
- Complementary detection layer for novel fraud patterns

---

#### F5: ML Pipeline Performance Dashboard

<div align="center">
<img src="docs/screenshots/ml-performance.png" alt="ML Performance Screenshot" width="700">
</div>

- Algorithm comparison across RF, GB, and XGBoost
- Train and test set evaluation with confusion matrices
- ROC and Precision-Recall curves
- SHAP global feature importance

---

#### F6: Statistical Hypothesis Validation

<div align="center">
<img src="docs/screenshots/hypotheses.png" alt="Hypotheses Screenshot" width="700">
</div>

- Four hypotheses validated with statistical tests
- Mann-Whitney U, Chi-squared tests with effect sizes
- Visual evidence alongside statistical results
- Potential courses of action

---

#### F7: Responsive Navigation

- Sidebar navigation with 7 dashboard pages
- Clear page titles and business requirement labels
- Consistent layout across all pages

---

## How to Use the Dashboard

### Quick Start Guide

#### 🔎 Fraud Pattern Study
1. Open the **Fraud Pattern Study** page from the sidebar
2. Toggle checkboxes to explore different visualisations
3. Use the **dropdown** to compare individual PCA features between fraud and legitimate classes

---

#### 🎯 Fraud Detector — Manual Entry

<div align="center">
<img src="docs/screenshots/fraud-detector-demo.gif" alt="Fraud Detector Demo" width="700">
</div>

1. Select **Manual Entry** mode
2. Adjust the feature sliders:

**To trigger a FRAUD detection:**

| Parameter | Value | Why |
|-----------|-------|-----|
| Amount | 50 | Low amounts are typical for fraud |
| V14 | -8.0 | Strong negative V14 is the #1 fraud signal |
| V12 | -4.0 | Negative V12 reinforces the fraud pattern |
| V10 | -3.0 | Adds additional fraud signal |

**To see a LEGITIMATE result:**

| Parameter | Value | Why |
|-----------|-------|-----|
| Amount | 500 | Normal purchase amount |
| V14 | 0.0 | Neutral value — no fraud signal |
| V12 | 0.0 | Neutral value |
| V10 | 0.0 | Neutral value |

3. Click **🔍 Analyse Transaction**
4. View the **fraud probability gauge** and **SHAP explanation** showing which features drove the prediction

---

#### 🎯 Fraud Detector — Live Simulation
1. Select **Live Simulation** mode
2. Click **Start Simulation**
3. Watch 20 real test transactions being classified in real-time
4. Each transaction receives a 🟢 LEGIT or 🔴 FRAUD label

---

#### ⚖️ Threshold & Cost Analysis
1. Move the **threshold slider** to see how it affects precision and recall
2. Lower threshold → catches more fraud but more false alarms
3. Higher threshold → fewer false alarms but misses more fraud
4. Adjust **cost per missed fraud** and **cost per false alarm** to find the business-optimal threshold

---

#### 🤖 Anomaly Detection
1. View the **reconstruction error distribution** — fraud transactions (red) have higher errors
2. Adjust the **anomaly threshold slider** to explore different detection sensitivity levels
3. Compare the autoencoder metrics with XGBoost in the comparison table

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
| tensorflow-cpu | 2.13.0 | Autoencoder neural network (local training only — not deployed) |
| scipy | 1.11.0 | Statistical hypothesis testing |

#### Data Visualisation

| Library | Version | Purpose |
|---------|---------|---------|
| plotly | 5.17.0 | Interactive visualisations |
| seaborn | 0.12.0 | Statistical visualisations |
| matplotlib | 3.7.0 | Static plots and SHAP integration |


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
| Heroku | Cloud deployment |
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
| Navigation | Click each of the 7 pages in sidebar | Page loads without error | ✅ Pass |
| Summary Page | Load Project Summary | Business requirements and glossary displayed | ✅ Pass |
| Fraud Study | Toggle each checkbox | Corresponding visualisation appears | ✅ Pass |
| Fraud Study | Select feature from dropdown | Violin plot updates to selected feature | ✅ Pass |
| Hypotheses | Load Hypotheses page | All 4 hypotheses with metrics displayed | ✅ Pass |
| Detector — Manual | Enter values and click Analyse | Probability score + SHAP bar chart displayed | ✅ Pass |
| Detector — Manual | Set V14 to -15 (strong fraud signal) | High fraud probability returned | ✅ Pass |
| Detector — Manual | Leave all values at 0 | Low fraud probability returned | ✅ Pass |
| Detector — CSV | Upload valid CSV file | Batch results with risk flags displayed | ✅ Pass |
| Detector — CSV | Upload non-CSV file | Error message shown | ✅ Pass |
| Detector — Simulation | Click Start Simulation | 20 transactions processed sequentially | ✅ Pass |
| Threshold | Move threshold slider | Precision, Recall, F1 and confusion matrix update | ✅ Pass |
| Threshold | Change cost per missed fraud | Total business cost recalculates | ✅ Pass |
| Threshold | Set threshold to 0.05 | High recall, low precision shown | ✅ Pass |
| Threshold | Set threshold to 0.95 | Low recall, high precision shown | ✅ Pass |
| Anomaly | Load Anomaly Detection page | Reconstruction error histogram displayed | ✅ Pass |
| Anomaly | Adjust anomaly threshold slider | Recall and precision metrics update | ✅ Pass |
| ML Performance | Click Confusion Matrix tab | Train and test matrices displayed | ✅ Pass |
| ML Performance | Click ROC & PR Curves tab | Both curves with AUC values displayed | ✅ Pass |
| ML Performance | Click Feature Importance tab | SHAP summary + XGBoost importance shown | ✅ Pass |
| ML Performance | Click Hyperparameters tab | Best params + rationale table displayed | ✅ Pass |
| ML Performance | Check success statement | Green success banner with F1 score shown | ✅ Pass |

### User Story Testing

| User Story | Acceptance Criteria Met | Evidence |
|------------|----------------------|---------|
| 1.1 Amount distribution for fraud vs legit | ✅ | Histogram + box plot on Fraud Pattern Study page |
| 1.2 Feature correlations with fraud | ✅ | Heatmap of top 12 features on Fraud Pattern Study page |
| 1.3 Temporal fraud patterns | ✅ | Fraud rate by hour line plot on Fraud Pattern Study page |
| 1.4 PCA feature distributions | ✅ | Interactive violin plots with dropdown on Fraud Pattern Study page |
| 2.1 Input transaction and get fraud probability | ✅ | Manual entry with gauge on Fraud Detector page |
| 2.2 Explainable prediction (SHAP) | ✅ | SHAP bar chart showing top 10 feature contributions |
| 2.3 Batch upload and flag transactions | ✅ | CSV upload with risk flags on Fraud Detector page |
| 2.4 Adjust decision threshold | ✅ | Interactive slider with cost calculator on Threshold page |
| 3.1 Detect anomalies without labels | ✅ | Autoencoder reconstruction error analysis on Anomaly page |
| 3.2 Compare supervised vs unsupervised | ✅ | Side-by-side metrics table on Anomaly Detection page |

### Validator Testing

#### Python (PEP8 / Flake8)


<div align="center">

| File | Lines | Issues | Status |
|------|-------|--------|--------|
| `app.py` | 52 | 0 | ✅ Pass |
| `app_pages/page_summary.py` | 69 | 0 | ✅ Pass |
| `app_pages/page_fraud_study.py` | 156 | 0 | ✅ Pass |
| `app_pages/page_hypotheses.py` | 125 | 0 | ✅ Pass |
| `app_pages/page_detector.py` | 232 | 0 | ✅ Pass |
| `app_pages/page_threshold_analysis.py` | 139 | 0 | ✅ Pass |
| `app_pages/page_anomaly_detection.py` | 141 | 0 | ✅ Pass |
| `app_pages/page_ml_performance.py` | 202 | 0 | ✅ Pass |
| `src/data_management.py` | 58 | 0 | ✅ Pass |
| `generate_dashboard_data.py` | 68 | 0 | ✅ Pass |
| `fix_outputs.py` | 266 | 0 | ✅ Pass |
| **TOTAL** | **1,508** | **0** | **100%** |


</div>

#### Jupyter Notebooks

All notebooks follow PEP8 standards and include Objectives/Inputs/Outputs headers.

---

### Bugs

### Resolved Issues

---

#### Bug #1: Notebook File Missing .ipynb Extension
**Issue:** Jupyter notebook created without `.ipynb` extension — file named `01_DataCollection` instead of `01_DataCollection.ipynb`
**Cause:** File was created/saved without the proper extension in VS Code
**Fix:** Renamed the file manually via PowerShell:
```bash
ren notebooks\01_DataCollection notebooks\01_DataCollection.ipynb
```
**Status:** ✅ Resolved

---

#### Bug #2: Linux Commands Not Recognised on Windows
**Issue:** `unzip is not recognized` and `rm is not recognized` errors when running Kaggle download cell in Jupyter Notebook
**Cause:** The notebook used Linux shell commands (`unzip`, `rm`) which are not available in Windows PowerShell
**Fix:** Downloaded the dataset manually from Kaggle website and placed `creditcard.csv` directly into the `data/` folder. The Kaggle API cells are kept for documentation but skipped during local execution.
**Status:** ✅ Resolved

---

#### Bug #3: SyntaxError — Markdown Text in Code Cell
**Issue:** `SyntaxError: invalid character '–' (U+2014)` in the Class Distribution section
**Cause:** Markdown description text was accidentally placed inside a Python code cell instead of a separate Markdown cell
**Fix:** Removed the plain text from the code cell, keeping only the Python code. The description was already present in the Markdown cell above.
**Status:** ✅ Resolved

---

#### Bug #4: NameError — df Not Defined
**Issue:** `NameError: name 'df' is not defined` when running the Quick Look at Key Features cell
**Cause:** Kernel lost variables after skipping cells and running out of order. The `pd.read_csv()` cell had not been executed in the current kernel session.
**Fix:** Restarted the kernel (Kernel → Restart) and ran all cells from top to bottom in correct order, skipping only the Kaggle API cells.
**Status:** ✅ Resolved

---

#### Bug #5: ModuleNotFoundError — No module named 'plotly'
**Issue:** `ModuleNotFoundError: No module named 'plotly'` when importing visualisation libraries in Notebook 02
**Cause:** Plotly, seaborn, matplotlib, and scipy were listed in `requirements.txt` but not yet installed in the local virtual environment
**Fix:** Installed missing packages inside the virtual environment:
```bash
pip install plotly seaborn matplotlib scipy
```
**Status:** ✅ Resolved

---

#### Bug #6: FileNotFoundError — creditcard.csv Not Found in Notebook 02
**Issue:** `FileNotFoundError: [Errno 2] No such file or directory: 'data/creditcard.csv'` in the Data Visualisation notebook
**Cause:** The working directory change cell used `endswith("notebooks")` check, but Jupyter was launched from a different location
**Fix:** Verified working directory with `os.getcwd()`, confirmed it pointed to the project root, and ensured `creditcard.csv` was present in the `data/` folder before running the notebook.
**Status:** ✅ Resolved

---

#### Bug #7: Seaborn Violin Plot Palette Error
**Issue:** `ValueError: The palette dictionary is missing keys: {'1', '0'}` when rendering violin plots of PCA features
**Cause:** Seaborn v0.14+ changed how `palette` works with the `x` parameter — it now expects palette keys to match the actual data values (integers 0 and 1), not positional colours
**Fix:** Updated the violin plot call to use `hue='Class'` parameter and passed palette as a list:
```python
sns.violinplot(
    data=df, x='Class', y=feature, ax=ax,
    hue='Class', palette=['#636EFA', '#EF553B'],
    inner='box', legend=False
)
```
**Status:** ✅ Resolved

---

#### Bug #8: ModuleNotFoundError — No module named 'sklearn'
**Issue:** `ModuleNotFoundError: No module named 'sklearn'` when running train/test split in Notebook 03
**Cause:** scikit-learn was not installed in the virtual environment despite being in requirements.txt
**Fix:** Installed the package:
```bash
pip install scikit-learn
```
**Status:** ✅ Resolved

---

#### Bug #9: ModuleNotFoundError — No module named 'imblearn'
**Issue:** `ModuleNotFoundError: No module named 'imblearn'` when importing SMOTE in Notebook 04
**Cause:** imbalanced-learn package was not installed in the virtual environment
**Fix:** Installed the package:
```bash
pip install imbalanced-learn
```
**Status:** ✅ Resolved

---

#### Bug #10: Jupyter Kernel Not Found on MacBook
**Issue:** Jupyter notebooks showing "No Kernel" after switching development from Windows to MacBook
**Cause:** Virtual environment kernel was not registered with Jupyter on the new machine
**Fix:** Installed ipykernel and registered the venv:
```bash
pip install ipykernel
python -m ipykernel install --user --name=venv --display-name="Python (venv)"
```
**Status:** ✅ Resolved

---

#### Bug #11: XGBoost libomp Missing on macOS
**Issue:** `XGBoostError: XGBoost Library (libxgboost.dylib) could not be loaded` when running model on MacBook
**Cause:** XGBoost requires the OpenMP runtime library (`libomp`) which is not included by default on macOS
**Fix:** Installed libomp via Homebrew:
```bash
brew install libomp
```
**Status:** ✅ Resolved

---

#### Bug #12: Git Push Rejected — Non-Fast-Forward
**Issue:** `error: failed to push some refs` with `non-fast-forward` error when pushing from MacBook
**Cause:** Windows and MacBook had divergent commit histories after working on both machines without syncing
**Fix:** Aborted the failed rebase and reset to remote state:
```bash
git rebase --abort
git fetch origin
git reset --hard origin/main
```
Then regenerated all output files locally on MacBook before committing.
**Status:** ✅ Resolved

---

#### Bug #13: ModuleNotFoundError — No module named 'tensorflow'
**Issue:** `ModuleNotFoundError: No module named 'tensorflow'` when running Autoencoder notebook on MacBook
**Cause:** TensorFlow was not installed in the MacBook virtual environment
**Fix:** Installed TensorFlow:
```bash
pip install tensorflow
```
**Status:** ✅ Resolved

---

#### Bug #14: ML Output Files Missing After Machine Switch
**Issue:** Dashboard pages showing `FileNotFoundError` for `shap_explainer.pkl`, `reconstruction_errors.pkl`, and other artifacts
**Cause:** ML model output files (`.pkl`, `.h5`, `.png`) were generated on Windows but not committed to Git. After switching to MacBook, the `outputs/v2` and `outputs/v3` directories were incomplete or missing.
**Fix:** Created `fix_outputs.py` script to regenerate all missing artifacts directly from the saved model and data files:
```bash
python fix_outputs.py
```
This script generates SHAP explainer, feature importance, autoencoder model, reconstruction errors, and comparison results in a single run.
**Status:** ✅ Resolved

---

#### Bug #15: Git Repository Too Large for Deployment (683MB)
**Issue:** `outputs/` folder totalled 683MB, exceeding GitHub's 100MB per-file limit and making Render deployment impossible
**Cause:** Large CSV files generated during data preparation (X_train_resampled.csv = 320MB, X_train_engineered.csv = 160MB, X_train.csv = 128MB) were tracked by Git. These intermediate training files are needed by notebooks but not by the dashboard.
**Fix:** Created `generate_dashboard_data.py` to pre-compute only the small files the dashboard needs (sampled data, pre-computed metrics, feature names). Added large CSV files to `.gitignore` and removed them from Git tracking:
```bash
git rm --cached outputs/v1/X_train_resampled.csv
git rm --cached outputs/v1/X_train_engineered.csv
git rm --cached outputs/v1/X_train.csv
git rm --cached outputs/v1/X_test.csv
git rm --cached outputs/v1/X_test_engineered.csv
```
Updated dashboard pages to load from `outputs/dashboard/` (~5MB) instead of the full CSV files.
**Status:** ✅ Resolved

---

#### Bug #16: Render Deployment Using Wrong Python Version
**Issue:** Render deployed with Python 3.14.3 instead of 3.11.5, causing package build failures
**Cause:** Render ignores `runtime.txt` by default and uses its own latest Python version
**Fix:** Switched deployment to Heroku which correctly reads `runtime.txt`. Also removed `tensorflow-cpu` and `kaggle` from `requirements.txt` as they are not needed by the dashboard (autoencoder already trained, dataset already downloaded).
**Status:** ✅ Resolved

---

#### Bug #17: SHAP Explainer Pickle Incompatibility on Heroku
**Issue:** `AttributeError: Can't get attribute 'TreeExplainer'` when loading `shap_explainer.pkl` on Heroku
**Cause:** The SHAP explainer was serialised with a different version of SHAP locally than the one installed on Heroku. Pickle files are version-sensitive for complex objects.
**Fix:** Instead of loading a pre-saved explainer, compute it on-the-fly in the dashboard:
```python
# Before (broken):
explainer = joblib.load("outputs/v2/shap_explainer.pkl")

# After (fixed):
import shap
explainer = shap.TreeExplainer(model)
```
**Status:** ✅ Resolved

---

### Known Issues

| Issue | Description | Impact | Workaround |
|-------|-------------|--------|------------|
| Windows Kaggle CLI | Kaggle download + unzip commands require Linux/Mac shell | Low | Download dataset manually from Kaggle website |
| Plotly in Jupyter | Plotly charts may not render in some Jupyter configurations | Low | Use `fig.show()` or install `nbformat` |
| Cross-machine development | Output files not synced when switching between Windows and Mac | Medium | Run `fix_outputs.py` to regenerate all artifacts |
---

## Deployment

### Heroku

The application is deployed on Heroku.

**Live URL:** [https://credit-card-fraud-detection-st-4cbb00a4456a.herokuapp.com/](https://credit-card-fraud-detection-st-4cbb00a4456a.herokuapp.com/)

#### Deployment Steps

1. **Create Heroku App**
   - Log in to [Heroku Dashboard](https://dashboard.heroku.com/)
   - Click "New" → "Create new app"
   - Enter app name and select region (Europe)

2. **Connect GitHub Repository**
   - Go to "Deploy" tab
   - Select "GitHub" as deployment method
   - Search and connect `credit-card-fraud-detection` repository

3. **Deploy**
   - Scroll to "Manual deploy" section
   - Select `main` branch
   - Click "Deploy Branch"
   - Wait for build to complete

4. **Configuration Files**

   | File | Purpose |
   |------|---------|
   | `Procfile` | Defines the start command: `web: sh setup.sh && streamlit run app.py` |
   | `runtime.txt` | Specifies Python version: `python-3.11.5` |
   | `setup.sh` | Configures Streamlit for headless server mode |
   | `requirements.txt` | Python package dependencies |

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

For detailed code attribution, see [CODE_ATTRIBUTION.md](docs/CODE_ATTRIBUTION.md)

### Dataset
- [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) — Machine Learning Group of ULB (Université Libre de Bruxelles) in collaboration with Worldline
- Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, and Gianluca Bontempi. *Calibrating Probability with Undersampling for Unbalanced Classification.* In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015.

### Documentation & Tutorials

| Resource | Usage |
|----------|-------|
| [Streamlit Documentation](https://docs.streamlit.io/) | Dashboard development |
| [XGBoost Documentation](https://xgboost.readthedocs.io/) | Gradient boosting classifier |
| [SHAP Documentation](https://shap.readthedocs.io/) | Model explainability |
| [Scikit-learn Documentation](https://scikit-learn.org/) | ML preprocessing and evaluation |
| [TensorFlow/Keras Documentation](https://www.tensorflow.org/) | Autoencoder architecture |
| [Plotly Documentation](https://plotly.com/python/) | Interactive visualisations |
| [Seaborn Documentation](https://seaborn.pydata.org/) | Statistical visualisations |
| [Imbalanced-learn Documentation](https://imbalanced-learn.org/) | SMOTE oversampling |
| [SciPy Documentation](https://docs.scipy.org/) | Statistical hypothesis testing (Mann-Whitney U, Chi-squared) |
| [Pandas Documentation](https://pandas.pydata.org/docs/) | Data manipulation and analysis |

### Code References

| Source | Usage | File(s) |
|--------|-------|---------|
| [Kaggle API Documentation](https://github.com/Kaggle/kaggle-api) | Dataset download from endpoint | `notebooks/01_DataCollection.ipynb` |
| [Plotly Express Bar Chart](https://plotly.com/python/bar-charts/) | Class distribution visualisation | `notebooks/02_DataVisualization.ipynb` |
| [Plotly Express Histogram](https://plotly.com/python/histograms/) | Amount distribution with marginal box plot | `notebooks/02_DataVisualization.ipynb` |
| [Seaborn Heatmap](https://seaborn.pydata.org/generated/seaborn.heatmap.html) | Correlation heatmap | `notebooks/02_DataVisualization.ipynb` |
| [Seaborn Violin Plot](https://seaborn.pydata.org/generated/seaborn.violinplot.html) | PCA feature distribution plots | `notebooks/02_DataVisualization.ipynb` |
| [SciPy Mann-Whitney U](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html) | Hypothesis testing H1, H3 | `notebooks/02_DataVisualization.ipynb` |
| [SciPy Chi-squared](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html) | Hypothesis testing H2 | `notebooks/02_DataVisualization.ipynb` |
| [Stack Overflow — Seaborn palette error](https://stackoverflow.com/questions/76550417/) | Fix for violin plot palette with hue parameter in Seaborn v0.14+ | `notebooks/02_DataVisualization.ipynb` |

### Tools Used

| Tool | Purpose |
|------|---------|
| VS Code | Code editor |
| Jupyter Notebook | Data analysis and modelling |
| Git | Version control |
| GitHub | Repository hosting |
| GitHub Projects | Agile project management |
| Heroku | Cloud deployment |

### Acknowledgements
- **Code Institute** — For the learning materials and assessment framework

---
