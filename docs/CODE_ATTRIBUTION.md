# Code Attribution & References

This section provides comprehensive documentation of all external code, libraries, patterns, and resources used in this project. All code has been properly attributed, understood, adapted, and integrated.

---

## Core Libraries

### Streamlit (v1.28.0) — Dashboard Framework
- **Source:** [Streamlit Documentation](https://docs.streamlit.io/)
- **License:** Apache License 2.0
- **Usage:** Entire dashboard application

#### Code Adaptations:
```python
# Page config pattern from Streamlit documentation
# Used in app.py lines 7-12
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)
```
- **Reference:** [Streamlit set_page_config](https://docs.streamlit.io/develop/api-reference/configuration/st.set_page_config)
```python
# Sidebar radio navigation from Streamlit documentation
# Used in app.py lines 30-35
selection = st.sidebar.radio("Go to", list(pages.keys()))
pages[selection]()
```
- **Reference:** [Streamlit Sidebar](https://docs.streamlit.io/develop/api-reference/layout/st.sidebar)
```python
# Cache decorator for data loading from Streamlit documentation
# Used in src/data_management.py lines 8-11
@st.cache_data
def load_raw_data():
    df = pd.read_csv("data/creditcard.csv")
    return df
```
- **Reference:** [Streamlit Caching](https://docs.streamlit.io/develop/concepts/architecture/caching)
```python
# Cache resource for ML models from Streamlit documentation
# Used in src/data_management.py lines 20-23
@st.cache_resource
def load_model(version="v2"):
    return joblib.load(f"outputs/{version}/fraud_pipeline_v2.pkl")
```
- **Reference:** [Streamlit cache_resource](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource)

---

### XGBoost (v2.0.0) — Gradient Boosting Classifier
- **Source:** [XGBoost Documentation](https://xgboost.readthedocs.io/)
- **License:** Apache License 2.0
- **Usage:** Primary supervised fraud classification model

#### Code Adaptations:
```python
# XGBClassifier setup from XGBoost documentation
# Used in notebooks/05_Modelling_XGBoost.ipynb
from xgboost import XGBClassifier

xgb = XGBClassifier(
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)
```
- **Reference:** [XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier)
```python
# Feature importance extraction from XGBoost documentation
# Used in notebooks/05_Modelling_XGBoost.ipynb
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)
```
- **Reference:** [XGBoost Feature Importance](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier.feature_importances_)

---

### Scikit-learn (v1.3.0) — ML Preprocessing & Evaluation
- **Source:** [Scikit-learn Documentation](https://scikit-learn.org/)
- **License:** BSD 3-Clause License
- **Usage:** Train/test split, scaling, evaluation metrics, hyperparameter tuning

#### Code Adaptations:
```python
# Stratified train/test split from Scikit-learn documentation
# Used in notebooks/03_DataCleaning.ipynb
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```
- **Reference:** [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
```python
# RobustScaler for outlier-resistant scaling from Scikit-learn docs
# Used in notebooks/04_FeatureEngineering.ipynb
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
```
- **Reference:** [RobustScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html)
```python
# RandomizedSearchCV for hyperparameter tuning from Scikit-learn docs
# Used in notebooks/05_Modelling_XGBoost.ipynb
from sklearn.model_selection import RandomizedSearchCV

search = RandomizedSearchCV(
    xgb, param_distributions,
    n_iter=50, scoring='f1',
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    random_state=42, verbose=1, n_jobs=-1
)
```
- **Reference:** [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
```python
# Classification report and confusion matrix from Scikit-learn docs
# Used in notebooks/05_Modelling_XGBoost.ipynb and dashboard pages
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred,
                            target_names=['Legitimate', 'Fraud']))
cm = confusion_matrix(y_test, y_pred)
```
- **Reference:** [classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)
```python
# Mutual Information for feature selection from Scikit-learn docs
# Used in notebooks/04_FeatureEngineering.ipynb
from sklearn.feature_selection import mutual_info_classif

mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
```
- **Reference:** [mutual_info_classif](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html)

---

### Imbalanced-learn (v0.11.0) — SMOTE Oversampling
- **Source:** [Imbalanced-learn Documentation](https://imbalanced-learn.org/)
- **License:** MIT License
- **Usage:** Handling class imbalance via synthetic oversampling

#### Code Adaptations:
```python
# SMOTE oversampling from Imbalanced-learn documentation
# Used in notebooks/04_FeatureEngineering.ipynb
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```
- **Reference:** [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
```python
# ADASYN comparison from Imbalanced-learn documentation
# Used in notebooks/04_FeatureEngineering.ipynb
from imblearn.over_sampling import ADASYN

adasyn = ADASYN(random_state=42)
X_adasyn, y_adasyn = adasyn.fit_resample(X_train, y_train)
```
- **Reference:** [ADASYN](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.ADASYN.html)

---

### SHAP (v0.42.0) — Model Explainability
- **Source:** [SHAP Documentation](https://shap.readthedocs.io/)
- **License:** MIT License
- **Usage:** Explaining individual fraud predictions

#### Code Adaptations:
```python
# TreeExplainer for XGBoost from SHAP documentation
# Used in notebooks/05_Modelling_XGBoost.ipynb
import shap

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_shap_sample)
```
- **Reference:** [SHAP TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html)
```python
# SHAP summary plot from SHAP documentation
# Used in notebooks/05_Modelling_XGBoost.ipynb
shap.summary_plot(shap_values, X_shap_sample, show=False)
```
- **Reference:** [SHAP Summary Plot](https://shap.readthedocs.io/en/latest/generated/shap.plots.beeswarm.html)
```python
# SHAP waterfall plot for single prediction from SHAP documentation
# Used in notebooks/05_Modelling_XGBoost.ipynb and app_pages/page_detector.py
shap.waterfall_plot(
    shap.Explanation(
        values=single_shap[0],
        base_values=explainer.expected_value,
        data=single_transaction.values[0],
        feature_names=X_test.columns.tolist()
    )
)
```
- **Reference:** [SHAP Waterfall Plot](https://shap.readthedocs.io/en/latest/generated/shap.plots.waterfall.html)

---

### TensorFlow/Keras (v2.13.0) — Autoencoder Neural Network
- **Source:** [TensorFlow Documentation](https://www.tensorflow.org/)
- **License:** Apache License 2.0
- **Usage:** Unsupervised anomaly detection autoencoder

#### Code Adaptations:
```python
# Functional API autoencoder from Keras documentation
# Used in notebooks/06_Modelling_Autoencoder.ipynb
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization

input_layer = Input(shape=(input_dim,))
encoded = Dense(32, activation='relu')(input_layer)
encoded = BatchNormalization()(encoded)
encoded = Dropout(0.2)(encoded)
encoded = Dense(16, activation='relu')(encoded)
encoded = BatchNormalization()(encoded)
encoded = Dense(8, activation='relu')(encoded)

decoded = Dense(16, activation='relu')(encoded)
decoded = BatchNormalization()(decoded)
decoded = Dropout(0.2)(decoded)
decoded = Dense(32, activation='relu')(decoded)
decoded = BatchNormalization()(decoded)
decoded = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
```
- **Reference:** [Keras Functional API](https://www.tensorflow.org/guide/keras/functional_api)
```python
# EarlyStopping callback from Keras documentation
# Used in notebooks/06_Modelling_Autoencoder.ipynb
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)
```
- **Reference:** [Keras EarlyStopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping)

---

### SciPy (v1.11.0) — Statistical Testing
- **Source:** [SciPy Documentation](https://docs.scipy.org/)
- **License:** BSD 3-Clause License
- **Usage:** Hypothesis validation with statistical tests

#### Code Adaptations:
```python
# Mann-Whitney U test from SciPy documentation
# Used in notebooks/02_DataVisualization.ipynb for H1 and H3
from scipy.stats import mannwhitneyu

stat, p_value = mannwhitneyu(
    fraud_amounts, legit_amounts, alternative='two-sided'
)
```
- **Reference:** [scipy.stats.mannwhitneyu](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html)
```python
# Chi-squared test from SciPy documentation
# Used in notebooks/02_DataVisualization.ipynb for H2
from scipy.stats import chi2_contingency

contingency = pd.crosstab(df['Hour_bin'], df['Class'])
chi2, p_val, dof, expected = chi2_contingency(contingency)
```
- **Reference:** [scipy.stats.chi2_contingency](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html)

---

### Plotly (v5.17.0) — Interactive Visualisations
- **Source:** [Plotly Documentation](https://plotly.com/python/)
- **License:** MIT License
- **Usage:** Dashboard interactive charts

#### Code Adaptations:
```python
# Plotly Express histogram with marginal from Plotly documentation
# Used in notebooks/02_DataVisualization.ipynb and app_pages/page_fraud_study.py
import plotly.express as px

fig = px.histogram(
    df, x='Amount', color='Class',
    marginal='box', barmode='overlay',
    color_discrete_map={0: '#636EFA', 1: '#EF553B'},
    opacity=0.7
)
```
- **Reference:** [Plotly Express Histogram](https://plotly.com/python/histograms/)
```python
# Plotly Gauge indicator from Plotly Graph Objects documentation
# Used in app_pages/page_detector.py
import plotly.graph_objects as go

fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=proba * 100,
    gauge={
        'axis': {'range': [0, 100]},
        'steps': [
            {'range': [0, 30], 'color': '#d4edda'},
            {'range': [30, 70], 'color': '#fff3cd'},
            {'range': [70, 100], 'color': '#f8d7da'}
        ]
    }
))
```
- **Reference:** [Plotly Indicator](https://plotly.com/python/indicator/)
```python
# Plotly Heatmap for confusion matrix from Plotly documentation
# Used in app_pages/page_threshold_analysis.py
fig = go.Figure(data=go.Heatmap(
    z=[[tn, fp], [fn, tp]],
    x=['Predicted Legit', 'Predicted Fraud'],
    y=['Actual Legit', 'Actual Fraud'],
    colorscale='RdBu'
))
```
- **Reference:** [Plotly Heatmaps](https://plotly.com/python/heatmaps/)

---

### Seaborn (v0.12.0) — Statistical Visualisations
- **Source:** [Seaborn Documentation](https://seaborn.pydata.org/)
- **License:** BSD 3-Clause License
- **Usage:** Heatmaps, violin plots in notebooks

#### Code Adaptations:
```python
# Heatmap from Seaborn documentation
# Used in notebooks/02_DataVisualization.ipynb
import seaborn as sns

sns.heatmap(
    df[top_features].corr(),
    annot=True, cmap='RdBu_r', center=0, fmt='.2f'
)
```
- **Reference:** [seaborn.heatmap](https://seaborn.pydata.org/generated/seaborn.heatmap.html)
```python
# Violin plot with hue parameter from Seaborn documentation
# Adapted for Seaborn v0.14+ palette handling
# Used in notebooks/02_DataVisualization.ipynb
sns.violinplot(
    data=df, x='Class', y=feature,
    hue='Class', palette=['#636EFA', '#EF553B'],
    inner='box', legend=False
)
```
- **Reference:** [seaborn.violinplot](https://seaborn.pydata.org/generated/seaborn.violinplot.html)
- **Fix Reference:** [Stack Overflow — Seaborn palette error](https://stackoverflow.com/questions/76550417/)

---

### Kaggle API (v1.6.0) — Dataset Collection
- **Source:** [Kaggle API Documentation](https://github.com/Kaggle/kaggle-api)
- **License:** Apache License 2.0
- **Usage:** Downloading dataset from Kaggle endpoint

#### Code Adaptations:
```python
# Kaggle dataset download from API documentation
# Used in notebooks/01_DataCollection.ipynb
os.environ['KAGGLE_CONFIG_DIR'] = os.getcwd()
! kaggle datasets download -d mlg-ulb/creditcardfraud -p data/
```
- **Reference:** [Kaggle API Datasets](https://github.com/Kaggle/kaggle-api#datasets)

---

## Dashboard Pages (Streamlit)

### Page Structure Pattern
- **Source:** [Streamlit Multi-page Apps](https://docs.streamlit.io/develop/concepts/multipage-apps)
- **License:** Apache License 2.0
- **Usage:** Dashboard page organisation with sidebar navigation

#### Code Adaptations:
```python
# Multi-page pattern using dictionary routing
# Used in app.py lines 15-28
pages = {
    "📋 Project Summary": page_summary,
    "🔎 Fraud Pattern Study": page_fraud_study,
}
selection = st.sidebar.radio("Go to", list(pages.keys()))
pages[selection]()
```
- **Reference:** [Streamlit Navigation](https://docs.streamlit.io/develop/tutorials/multipage/st.page_link-nav)

### Interactive Widgets
```python
# Checkbox-controlled visualisations from Streamlit documentation
# Used in app_pages/page_fraud_study.py
if st.checkbox("Show Class Distribution"):
    fig = px.bar(...)
    st.plotly_chart(fig, use_container_width=True)
```
- **Reference:** [Streamlit Checkbox](https://docs.streamlit.io/develop/api-reference/widgets/st.checkbox)
```python
# Slider widget for threshold tuning from Streamlit documentation
# Used in app_pages/page_threshold_analysis.py
threshold = st.slider(
    "Decision Threshold",
    min_value=0.05, max_value=0.95,
    value=float(default_threshold), step=0.01
)
```
- **Reference:** [Streamlit Slider](https://docs.streamlit.io/develop/api-reference/widgets/st.slider)
```python
# File uploader for CSV batch processing from Streamlit documentation
# Used in app_pages/page_detector.py
uploaded = st.file_uploader("Choose a CSV file", type=['csv'])
```
- **Reference:** [Streamlit File Uploader](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader)
```python
# Tabs layout for organising content from Streamlit documentation
# Used in app_pages/page_ml_performance.py
tab1, tab2, tab3, tab4 = st.tabs([
    "Confusion Matrix", "ROC & PR Curves",
    "Feature Importance", "Hyperparameters"
])
```
- **Reference:** [Streamlit Tabs](https://docs.streamlit.io/develop/api-reference/layout/st.tabs)
```python
# Metric display from Streamlit documentation
# Used across multiple dashboard pages
col1, col2, col3 = st.columns(3)
col1.metric("Precision", f"{precision:.3f}")
col2.metric("Recall", f"{recall:.3f}")
col3.metric("F1-Score", f"{f1:.3f}")
```
- **Reference:** [Streamlit Metric](https://docs.streamlit.io/develop/api-reference/data/st.metric)
```python
# Expander for collapsible content from Streamlit documentation
# Used in app_pages/page_summary.py
with st.expander("ML Terminology Glossary"):
    st.markdown("**Binary Classification** — ...")
```
- **Reference:** [Streamlit Expander](https://docs.streamlit.io/develop/api-reference/layout/st.expander)

### Plotly Gauge Indicator
```python
# Gauge chart for fraud probability from Plotly documentation
# Used in app_pages/page_detector.py
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=proba * 100,
    title={'text': "Fraud Risk Score"},
    gauge={
        'axis': {'range': [0, 100]},
        'steps': [
            {'range': [0, 30], 'color': '#d4edda'},
            {'range': [30, 70], 'color': '#fff3cd'},
            {'range': [70, 100], 'color': '#f8d7da'}
        ]
    }
))
```
- **Reference:** [Plotly Indicator Gauge](https://plotly.com/python/indicator/)

### Plotly Heatmap for Confusion Matrix
```python
# Interactive confusion matrix from Plotly documentation
# Used in app_pages/page_threshold_analysis.py and page_ml_performance.py
fig = go.Figure(data=go.Heatmap(
    z=[[tn, fp], [fn, tp]],
    x=['Pred Legit', 'Pred Fraud'],
    y=['Actual Legit', 'Actual Fraud'],
    colorscale='RdBu',
    texttemplate="%{text}"
))
```
- **Reference:** [Plotly Heatmaps](https://plotly.com/python/heatmaps/)

---

## Utility Scripts

### fix_outputs.py — ML Artifact Generation
- **Source:** Custom script written to regenerate ML output files
- **Usage:** Generates SHAP explainer, feature importance, autoencoder model, and comparison results when switching between development machines
- **Libraries Used:** All libraries listed above (XGBoost, SHAP, TensorFlow, Scikit-learn)

---

## Deployment Optimisation

### Git Large File Handling
- **Source:** [GitHub Documentation — Removing sensitive data from a repository](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)
- **License:** GitHub Terms of Service
- **Usage:** Removing large CSV files from Git tracking while keeping them locally

#### Code Adaptations:
```bash
# Remove cached large files from Git tracking
# Used during deployment preparation
git rm --cached outputs/v1/X_train_resampled.csv
git rm --cached outputs/v1/X_train_engineered.csv
git rm --cached outputs/v1/X_train.csv
```
- **Reference:** [git rm --cached](https://git-scm.com/docs/git-rm)

### Streamlit Caching for Dashboard Performance
- **Source:** [Streamlit Caching Documentation](https://docs.streamlit.io/develop/concepts/architecture/caching)
- **Usage:** Pre-computing dashboard data to avoid loading large files at runtime

#### Code Adaptations:
```python
# Pre-compute small dashboard files from large training data
# Used in generate_dashboard_data.py
sample = pd.concat([
    df[df['Class'] == 0].sample(10000, random_state=42),
    df[df['Class'] == 1]
])
sample.to_csv("outputs/dashboard/data_sample.csv", index=False)
```
- **Reference:** [Streamlit Performance Best Practices](https://docs.streamlit.io/develop/concepts/architecture/caching)

### .gitignore for Large ML Artifacts
- **Source:** [GitHub .gitignore Documentation](https://docs.github.com/en/get-started/getting-started-with-git/ignoring-files)
- **Usage:** Excluding large intermediate CSV files from version control while keeping small model artifacts

#### Code Adaptations:
```gitignore
# Large output CSV files (regenerate via notebooks)
outputs/v1/X_train.csv
outputs/v1/X_train_resampled.csv
outputs/v1/X_train_engineered.csv
```
- **Reference:** [Gitignore Patterns](https://git-scm.com/docs/gitignore)

---

## Python Standard Library

### NumPy — Numerical Computing
```python
# Log transform from NumPy documentation
# Used in notebooks/04_FeatureEngineering.ipynb
X_train['Amount_log'] = np.log1p(X_train['Amount'])
```
- **Reference:** [numpy.log1p](https://numpy.org/doc/stable/reference/generated/numpy.log1p.html)

### Joblib — Model Serialisation
```python
# Model save/load pattern from Joblib documentation
# Used throughout notebooks and src/data_management.py
import joblib
joblib.dump(best_model, "outputs/v2/fraud_model_optimized.pkl")
model = joblib.load("outputs/v2/fraud_model_optimized.pkl")
```
- **Reference:** [Joblib Persistence](https://joblib.readthedocs.io/en/stable/persistence.html)

---

## Design Patterns

### CRISP-DM Methodology
- **Source:** [IBM CRISP-DM Guide](https://www.ibm.com/docs/en/spss-modeler/saas?topic=dm-crisp-help-overview)
- **Usage:** Project structure follows the 6-phase CRISP-DM lifecycle
- **Adaptation:** 7 Jupyter notebooks mapped to CRISP-DM phases (Business Understanding → Data Understanding → Data Preparation → Modelling → Evaluation → Deployment)

### Versioned Output Pattern
- **Source:** Code Institute learning materials
- **Usage:** Model artifacts saved in versioned folders (`outputs/v1/`, `outputs/v2/`, `outputs/v3/`)
- **Adaptation:** Each version represents a project milestone — v1 (baseline), v2 (optimised XGBoost), v3 (autoencoder)

---

## Community Resources

### Stack Overflow Solutions

| Issue | Solution | File |
|-------|----------|------|
| Seaborn v0.14+ palette error | Use `hue` parameter instead of `palette` dict | `notebooks/02_DataVisualization.ipynb` |

---

## License Compliance Summary

| Library | License | Commercial Use | Modification |
|---------|---------|----------------|--------------|
| Streamlit | Apache 2.0 | ✅ | ✅ |
| XGBoost | Apache 2.0 | ✅ | ✅ |
| Scikit-learn | BSD 3-Clause | ✅ | ✅ |
| Imbalanced-learn | MIT | ✅ | ✅ |
| SHAP | MIT | ✅ | ✅ |
| TensorFlow | Apache 2.0 | ✅ | ✅ |
| Plotly | MIT | ✅ | ✅ |
| Seaborn | BSD 3-Clause | ✅ | ✅ |
| SciPy | BSD 3-Clause | ✅ | ✅ |
| Pandas | BSD 3-Clause | ✅ | ✅ |
| NumPy | BSD 3-Clause | ✅ | ✅ |

All libraries used are open-source and permit commercial use, modification, and distribution.

---