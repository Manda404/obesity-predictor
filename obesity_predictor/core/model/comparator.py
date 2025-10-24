"""
comparator.py
=========================
Benchmarking utility to train and compare multiple models
on the same dataset, selecting the best one based on F1 score.

Author: Rostand Surel
"""

from obesity_predictor.config.logger_config import logger
from obesity_predictor.core.model.evaluator import ModelEvaluator


class ModelComparator:
    """
    Train, evaluate, and compare multiple ML models.
    """

    def __init__(self):
        self.evaluator = ModelEvaluator()
        self.results = {}

    def compare(self, trainers: dict, X_train, X_valid, y_train, y_valid):
        """
        Train and evaluate each model trainer.
        """
        for name, trainer in trainers.items():
            logger.info(f"[Comparator] Training model: {name}")
            model = trainer.train(X_train, y_train, X_valid, y_valid)
            preds = trainer.predict(X_valid)
            metrics = self.evaluator.evaluate(y_valid, preds)
            self.results[name] = metrics

        best_model_name = max(self.results, key=lambda k: self.results[k]["f1_score"])
        logger.success(f"[Comparator] Best model: {best_model_name} with F1={self.results[best_model_name]['f1_score']:.4f}")
        return best_model_name, self.results[best_model_name]
