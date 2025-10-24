"""
catboost_trainer.py
=========================
Trainer implementation for CatBoost model.

Author: Rostand Surel
"""

from catboost import CatBoostClassifier
from obesity_predictor.config.logger_config import logger
from obesity_predictor.core.model.base_trainer import BaseTrainer


class CatBoostTrainer(BaseTrainer):
    """
    Trainer for CatBoost multiclass classification model.
    """

    def __init__(self, params: dict):
        super().__init__(model_name="CatBoost", params=params)
        self.model = CatBoostClassifier(**params)

    def train(self, X_train, y_train, X_valid, y_valid):
        logger.info("[CatBoost] Training started...")
        self.model.fit(X_train, y_train, eval_set=(X_valid, y_valid), verbose=False)
        logger.success("[CatBoost] Training completed.")
        return self.model

    def predict(self, X):
        logger.info("[CatBoost] Generating predictions...")
        return self.model.predict(X)

    def save(self, path: str):
        self.model.save_model(path)
        logger.info(f"[CatBoost] Model saved at {path}")

    def load(self, path: str):
        self.model.load_model(path)
        logger.info(f"[CatBoost] Model loaded from {path}")