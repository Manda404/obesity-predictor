"""
main_app.py
=========================
Streamlit web application for the Obesity Predictor project.

Features
--------
- Interactive dashboard for model results and comparisons
- Displays MLflow metrics and drift monitoring
- Links to API prediction interface

Author: Rostand Surel
"""

import streamlit as st
from obesity_predictor.config.logger_config import logger
from obesity_predictor.core.utils.visualization import plot_confusion_matrix, plot_feature_importance
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Obesity Predictor Dashboard",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Obesity Predictor — Model Dashboard")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Configuration")
    model_choice = st.selectbox("Select model", ["CatBoost", "XGBoost", "LightGBM"])
    artifact_dir = Path("data/models")
    model_path = artifact_dir / f"{model_choice}_model.joblib"

    if model_path.exists():
        st.success(f"Model found: {model_path.name}")
    else:
        st.warning("Model not yet trained.")

st.subheader("📊 Model Overview")
st.markdown(
    "Visualize model performance, feature importance, and prediction results."
)

if model_path.exists():
    model = joblib.load(model_path)

    # Placeholder dataset for demonstration
    df = pd.read_csv("data/processed/validation_sample.csv") if Path("data/processed/validation_sample.csv").exists() else None
    if df is not None:
        X = df.drop(columns=["NObeyesdad"], errors="ignore")
        y_true = df["NObeyesdad"] if "NObeyesdad" in df.columns else None
        y_pred = model.predict(X)

        if y_true is not None:
            st.pyplot(plot_confusion_matrix(y_true, y_pred))
        st.pyplot(plot_feature_importance(model, feature_names=X.columns, top_n=15))
    else:
        st.info("No processed validation sample found yet.")
else:
    st.stop()

st.markdown("---")
st.write("✅ Built with Streamlit + MLflow + Loguru for full MLOps visibility.")
