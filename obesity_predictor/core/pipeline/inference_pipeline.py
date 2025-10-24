"""
inference_pipeline.py
=========================
Pipeline to load a registered model (or local artifact)
and perform predictions on new incoming data.

Author: Rostand Surel
"""

import pandas as pd
import joblib
from obesity_predictor.config.logger_config import logger
from obesity_predictor.config.settings import settings
from obesity_predictor.core.preprocessing.inference_preprocessor import InferencePreprocessor


class InferencePipeline:
    """
    End-to-end inference pipeline for production predictions.
    """

    def __init__(self, model_path: str, preprocessor_path: str):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model = None
        self.preprocessor = None

    def load(self):
        """
        Load model and preprocessor from disk.
        """
        logger.info("[InferencePipeline] Loading model and preprocessor...")
        self.model = joblib.load(self.model_path)
        self.preprocessor = InferencePreprocessor(self.preprocessor_path)
        self.preprocessor.load()
        logger.success("[InferencePipeline] Model and preprocessor loaded successfully.")

    def predict(self, input_data: pd.DataFrame):
        """
        Generate predictions on incoming data.

        Parameters
        ----------
        input_data : pd.DataFrame
            Raw user input data.

        Returns
        -------
        dict
            Predictions with labels.
        """
        if self.model is None or self.preprocessor is None:
            self.load()

        transformed = self.preprocessor.transform(input_data)
        preds = self.model.predict(transformed)
        logger.info(f"[InferencePipeline] Predictions generated: {preds}")
        return {"predictions": preds.tolist()}