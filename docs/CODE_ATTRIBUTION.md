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