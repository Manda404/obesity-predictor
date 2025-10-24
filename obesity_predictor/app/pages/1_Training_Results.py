"""
1_Training_Results.py
=========================
Displays the training metrics and logged experiments from MLflow.

Author: Rostand Surel
"""

import streamlit as st
from obesity_predictor.core.utils.mlflow_utils import list_runs
from obesity_predictor.config.settings import settings
import pandas as pd

st.title("📈 Training Results")
st.markdown("---")

runs = list_runs(settings.experiment_name)

if runs:
    data = []
    for run in runs:
        data.append({
            "run_id": run.info.run_id,
            "model": run.data.params.get("model"),
            "accuracy": run.data.metrics.get("accuracy"),
            "f1_score": run.data.metrics.get("f1_score"),
            "date": run.info.start_time,
        })
    df = pd.DataFrame(data)
    st.dataframe(df)
else:
    st.info("No MLflow runs found yet.")
