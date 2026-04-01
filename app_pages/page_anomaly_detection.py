"""Page 6: Anomaly Detection — answers BR3 with autoencoder results."""
import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
import joblib
import json
from sklearn.metrics import classification_report, f1_score