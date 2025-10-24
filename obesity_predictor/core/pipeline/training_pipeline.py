"""
training_pipeline.py
=========================
Main pipeline for training a single ML model.

Responsible for:
- Loading data
- Preprocessing (fit + transform)
- Training the specified model
- Evaluating performance
- Logging to MLflow
- Saving artifacts (model + preprocessor)

Author: Rostand Surel
"""

from pathlib import Path
from obesity_predictor.config.logger_config import logger
from obesity_predictor.config.settings import settings
from obesity_predictor.core.data.loader import ObesityDataLoader
from obesity_predictor.core.data.splitter import ObesityDataSplitter
from obesity_predictor.core.preprocessing.train_preprocessor import TrainPreprocessor
from obesity_predictor.core.model.evaluator import ModelEvaluator
from obesity_predictor.core.utils.mlflow_utils import setup_mlflow
import joblib


class TrainingPipeline:
    """
    Pipeline to train and log a single model.
    """

    def __init__(self, trainer):
        self.trainer = trainer
        self.data_path = "data/raw/ObesityDataSet.csv"
        self.artifact_dir = Path(settings.artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        """
        Execute the full training pipeline.
        """
        logger.info(f"[Pipeline] Starting training for {self.trainer.model_name}...")

        # --- Load data ---
        loader = ObesityDataLoader(self.data_path)
        df = loader.load_data()

        # --- Split ---
        splitter = ObesityDataSplitter(df)
        X_train, X_valid, y_train, y_valid = splitter.split_data()

        # --- Preprocessing ---
        preprocessor = TrainPreprocessor()
        preprocessor.fit(X_train)
        X_train_t = preprocessor.transform(X_train)
        X_valid_t = preprocessor.transform(X_valid)

        # --- Train model ---
        model = self.trainer.train(X_train_t, y_train, X_valid_t, y_valid)

        # --- Evaluate ---
        evaluator = ModelEvaluator()
        preds = self.trainer.predict(X_valid_t)
        metrics = evaluator.evaluate(y_valid, preds)

        # --- Save artifacts ---
        model_path = self.artifact_dir / f"{self.trainer.model_name}_model.joblib"
        preproc_path = self.artifact_dir / f"{self.trainer.model_name}_preprocessor.joblib"

        joblib.dump(model, model_path)
        preprocessor.save(preproc_path)

        # --- Log to MLflow ---
        setup_mlflow()
        self.trainer.log_to_mlflow(metrics, artifacts={"model": str(model_path)})

        logger.success(f"[Pipeline] Training complete for {self.trainer.model_name}")
        return metrics
