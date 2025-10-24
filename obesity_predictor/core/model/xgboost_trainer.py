"""
xgboost_trainer.py
=========================
Trainer implementation for XGBoost model.

Author: Rostand Surel
"""

import xgboost as xgb
from obesity_predictor.config.logger_config import logger
from obesity_predictor.core.model.base_trainer import BaseTrainer


class XGBoostTrainer(BaseTrainer):
    """
    Trainer for XGBoost multiclass classification model.
    """

    def __init__(self, params: dict):
        super().__init__(model_name="XGBoost", params=params)
        self.model = xgb.XGBClassifier(**params)

    def train(self, X_train, y_train, X_valid, y_valid):
        logger.info("[XGBoost] Training started...")
        self.model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        logger.success("[XGBoost] Training completed.")
        return self.model

    def predict(self, X):
        logger.info("[XGBoost] Generating predictions...")
        return self.model.predict(X)

    def save(self, path: str):
        self.model.save_model(path)
        logger.info(f"[XGBoost] Model saved at {path}")

    def load(self, path: str):
        self.model.load_model(path)
        logger.info(f"[XGBoost] Model loaded from {path}")
