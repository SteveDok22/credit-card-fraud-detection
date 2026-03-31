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