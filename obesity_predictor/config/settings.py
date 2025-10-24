"""
settings.py
=========================
Centralized configuration management for the ObesityPredictor project.
Uses environment variables (.env) for dynamic setup of MLflow tracking,
artifacts directory, and other key parameters.

Author: Rostand Surel
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class Settings:
    """
    Dataclass to store key project configuration settings.
    """

    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    experiment_name: str = os.getenv("EXPERIMENT_NAME", "ObesityPredictor")
    model_name: str = os.getenv("MODEL_NAME", "ObesityPredictor-Best")
    artifact_dir: str = os.getenv("ARTIFACT_DIR", "data/models")
    target_column: str = os.getenv("TARGET_COLUMN", "NObeyesdad")
    test_size: float = float(os.getenv("TEST_SIZE", 0.2))
    random_state: int = int(os.getenv("RANDOM_STATE", 42))
    
    # NEW: Logging configuration
    log_dir: str = os.getenv("LOG_DIR", "logs")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


# Global settings instance
settings = Settings()
