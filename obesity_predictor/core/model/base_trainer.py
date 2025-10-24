"""
base_trainer.py
=========================
Abstract base class for all model trainers (CatBoost, XGBoost, LightGBM).

Provides a unified interface for:
- training
- evaluation
- logging to MLflow
- saving/loading model artifacts

Author: Rostand Surel
"""

from abc import ABC, abstractmethod
import mlflow
from obesity_predictor.config.logger_config import logger
from obesity_predictor.config.settings import settings


class BaseTrainer(ABC):
    """
    Abstract base trainer for all ML models in the project.
    """

    def __init__(self, model_name: str, params: dict):
        """
        Initialize the base trainer.

        Parameters
        ----------
        model_name : str
            Name of the model (e.g., 'CatBoost', 'XGBoost', 'LightGBM').
        params : dict
            Dictionary of hyperparameters.
        """
        self.model_name = model_name
        self.params = params
        self.model = None

    @abstractmethod
    def train(self, X_train, y_train, X_valid, y_valid):
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X):
        """Generate predictions."""
        pass

    @abstractmethod
    def save(self, path: str):
        """Save trained model to disk."""
        pass

    @abstractmethod
    def load(self, path: str):
        """Load a saved model."""
        pass

    def log_to_mlflow(self, metrics: dict, artifacts: dict = None):
        """
        Log model parameters, metrics, and optional artifacts to MLflow.
        """
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.experiment_name)

        with mlflow.start_run(run_name=self.model_name):
            mlflow.log_params(self.params)
            mlflow.log_metrics(metrics)

            if artifacts:
                for name, path in artifacts.items():
                    mlflow.log_artifact(path, artifact_path=name)

            mlflow.sklearn.log_model(self.model, artifact_path=self.model_name)
            logger.success(f"[MLflow] Logged model and metrics for {self.model_name}")
