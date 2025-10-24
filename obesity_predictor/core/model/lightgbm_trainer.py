"""
lightgbm_trainer.py
=========================
Trainer implementation for LightGBM model.

Author: Rostand Surel
"""

import lightgbm as lgb
from obesity_predictor.config.logger_config import logger
from obesity_predictor.core.model.base_trainer import BaseTrainer


class LightGBMTrainer(BaseTrainer):
    """
    Trainer for LightGBM multiclass classification model.
    """

    def __init__(self, params: dict):
        super().__init__(model_name="LightGBM", params=params)
        self.model = lgb.LGBMClassifier(**params)

    def train(self, X_train, y_train, X_valid, y_valid):
        logger.info("[LightGBM] Training started...")
        self.model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        logger.success("[LightGBM] Training completed.")
        return self.model

    def predict(self, X):
        logger.info("[LightGBM] Generating predictions...")
        return self.model.predict(X)

    def save(self, path: str):
        self.model.booster_.save_model(path)
        logger.info(f"[LightGBM] Model saved at {path}")

    def load(self, path: str):
        self.model = lgb.Booster(model_file=path)
        logger.info(f"[LightGBM] Model loaded from {path}")