"""
drift_detector.py
=========================
Detects data drift between reference (training) and
current (production) datasets using Evidently AI.

Author: Rostand Surel
"""

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DatasetDriftMetric
from evidently import ColumnMapping
import pandas as pd
from obesity_predictor.config.logger_config import logger
from pathlib import Path


class DriftDetector:
    """
    Wrapper around Evidently Report for drift detection.
    """

    def __init__(self, target_column: str):
        self.target_column = target_column
        self.report = Report(metrics=[DataDriftPreset(), DatasetDriftMetric()])

    def run(self, reference_df: pd.DataFrame, current_df: pd.DataFrame, output_path: str = "data/processed/drift_report.html"):
        """
        Run drift detection between reference and current datasets.

        Parameters
        ----------
        reference_df : pd.DataFrame
            Training or baseline dataset.
        current_df : pd.DataFrame
            New production dataset.
        output_path : str
            Path where the HTML report will be saved.
        """
        logger.info("[DriftDetector] Running data drift analysis...")

        mapping = ColumnMapping()
        mapping.target = self.target_column

        self.report.run(reference_data=reference_df, current_data=current_df, column_mapping=mapping)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self.report.save_html(output_path)

        logger.success(f"[DriftDetector] Drift report saved at {output_path}")

    def summarize(self) -> dict:
        """
        Return a dictionary summary of the drift results.
        """
        result = self.report.as_dict()
        drift_detected = result["metrics"][0]["result"]["dataset_drift"]
        num_drifted = result["metrics"][0]["result"]["n_drifted_columns"]
        total_columns = result["metrics"][0]["result"]["n_features"]

        summary = {
            "dataset_drift": drift_detected,
            "n_drifted_columns": num_drifted,
            "n_total_columns": total_columns,
            "drift_ratio": num_drifted / total_columns if total_columns else 0,
        }

        logger.info(f"[DriftDetector] Summary: {summary}")
        return summary