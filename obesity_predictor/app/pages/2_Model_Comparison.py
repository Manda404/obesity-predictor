"""
2_Model_Comparison.py
=========================
Compare performance metrics across models and select the best one.

Author: Rostand Surel
"""

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from obesity_predictor.core.utils.visualization import plot_feature_importance

st.title("🤖 Model Comparison")
st.markdown("---")

artifact_dir = Path("data/models")
models = [f for f in artifact_dir.glob("*_model.joblib")]

if not models:
    st.warning("No trained models found in data/models/")
    st.stop()

results = []
for model_file in models:
    model_name = model_file.stem.replace("_model", "")
    model = joblib.load(model_file)
    results.append({"Model": model_name, "Params": len(model.get_params()), "File": str(model_file)})

df = pd.DataFrame(results)
st.dataframe(df)

selected = st.selectbox("Select model to inspect", df["Model"])
model_path = artifact_dir / f"{selected}_model.joblib"
model = joblib.load(model_path)

st.pyplot(plot_feature_importance(model, feature_names=[f"Feature_{i}" for i in range(10)]))