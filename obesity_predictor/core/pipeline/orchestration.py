"""
orchestration.py
=========================
Global orchestration script:
- Trains and compares multiple ML models (CatBoost, XGBoost, LightGBM)
- Evaluates their performance
- Registers the best one in MLflow

Author: Rostand Surel
"""

import yaml
from obesity_predictor.config.logger_config import logger
from obesity_predictor.config.settings import settings
from obesity_predictor.core.model.catboost_trainer import CatBoostTrainer
from obesity_predictor.core.model.xgboost_trainer import XGBoostTrainer
from obesity_predictor.core.model.lightgbm_trainer import LightGBMTrainer
from obesity_predictor.core.model.comparator import ModelComparator
from obesity_predictor.core.model.registry import ModelRegistry
from obesity_predictor.core.pipeline.training_pipeline import TrainingPipeline
from pathlib import Path


def load_model_configs(config_path: str = "obesity_predictor/config/model_config.yaml"):
    """Load hyperparameters from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    """Main entry point for model training and comparison."""
    logger.info("========== Starting Model Orchestration ==========")
    config = load_model_configs()

    trainers = {
        "CatBoost": CatBoostTrainer(config["catboost"]),
        "XGBoost": XGBoostTrainer(config["xgboost"]),
        "LightGBM": LightGBMTrainer(config["lightgbm"]),
    }

    comparator = ModelComparator()
    results = {}

    # --- Run training for each model ---
    for name, trainer in trainers.items():
        pipeline = TrainingPipeline(trainer)
        metrics = pipeline.run()
        results[name] = metrics

    # --- Compare and register best model ---
    best_model_name, best_metrics = comparator.compare(trainers, None, None, None, None)
    logger.success(f"[Orchestration] Best model: {best_model_name} ({best_metrics})")

    # --- Register in MLflow ---
    registry = ModelRegistry()
    best_model_path = Path(settings.artifact_dir) / f"{best_model_name}_model.joblib"
    registry.register_model(str(best_model_path))

    logger.success("========== Model Orchestration Complete ==========")


if __name__ == "__main__":
    main()