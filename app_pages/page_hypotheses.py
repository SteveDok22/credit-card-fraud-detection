"""Page 3: Project Hypotheses — statistical validation results."""
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, chi2_contingency
from src.data_management import load_raw_data


def page_hypotheses():
    """Display the hypothesis validation page."""

    st.title("🧪 Project Hypotheses & Validation")
    st.write("---")

    df = load_raw_data()