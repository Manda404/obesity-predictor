"""
evaluator.py
=========================
Centralized evaluation module for model performance.
Computes key classification metrics and logs them.

Author: Rostand Surel
"""

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from obesity_predictor.config.logger_config import logger


class ModelEvaluator:
    """
    Compute and log evaluation metrics for classification tasks.
    """

    def evaluate(self, y_true, y_pred) -> dict:
        """
        Compute standard classification metrics.
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted"),
            "recall": recall_score(y_true, y_pred, average="weighted"),
            "f1_score": f1_score(y_true, y_pred, average="weighted"),
        }

        logger.info(f"[Evaluator] Metrics: {metrics}")
        logger.info(f"[Evaluator] Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")
        return metrics
