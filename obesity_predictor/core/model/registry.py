"""
registry.py
=========================
Handles MLflow Model Registry operations:
- register best model
- retrieve latest version
- promote to production

Author: Rostand Surel
"""

import mlflow
from obesity_predictor.config.settings import settings
from obesity_predictor.config.logger_config import logger


class ModelRegistry:
    """
    Wrapper for MLflow model registry operations.
    """

    def __init__(self):
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        self.client = mlflow.tracking.MlflowClient()

    def register_model(self, model_uri: str):
        """
        Register the trained model in MLflow Model Registry.
        """
        logger.info(f"[Registry] Registering model {settings.model_name} from {model_uri}")
        result = mlflow.register_model(model_uri, settings.model_name)
        logger.success(f"[Registry] Registered model: {result.name}, version: {result.version}")
        return result

    def get_latest_version(self):
        """
        Retrieve latest registered version.
        """
        versions = self.client.get_latest_versions(settings.model_name)
        latest = max(versions, key=lambda v: int(v.version))
        logger.info(f"[Registry] Latest version: {latest.version}, stage={latest.current_stage}")
        return latest