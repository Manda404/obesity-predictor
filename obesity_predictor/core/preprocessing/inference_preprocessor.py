"""
inference_preprocessor.py
=========================
Applies the same transformations learned during training to new
data received in production for prediction.

Ensures input consistency between train-time and inference-time.

Author: Rostand Surel
"""

import pandas as pd
import joblib
from obesity_predictor.config.logger_config import logger
from obesity_predictor.core.preprocessing.base_preprocessor import BasePreprocessor


class InferencePreprocessor(BasePreprocessor):
    """
    Load a saved preprocessor and apply transformations
    on incoming data during inference.
    """

    def __init__(self, preprocessor_path: str):
        super().__init__()
        self.pipeline = None
        self.preprocessor_path = preprocessor_path

    def fit(self, data: pd.DataFrame):
        """
        Fit is not used in inference. Included for interface compatibility.
        """
        logger.warning("[InferencePreprocessor] fit() is not used in inference mode.")

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform incoming production data using the loaded pipeline.

        Parameters
        ----------
        data : pd.DataFrame
            Raw production data to be transformed.

        Returns
        -------
        pd.DataFrame
            Transformed dataset ready for model prediction.
        """
        self._check_fitted()
        logger.info("[InferencePreprocessor] Applying inference transformations...")

        data["BMI"] = data["Weight"] / (data["Height"] ** 2)
        data["Age_Group"] = pd.cut(data["Age"], bins=[0, 18, 30, 50, 100], labels=["Teen", "Young", "Adult", "Senior"])

        transformed_array = self.pipeline.transform(data)
        transformed_df = pd.DataFrame(transformed_array)
        logger.success(f"[InferencePreprocessor] Transformation complete. Shape: {transformed_df.shape}")

        return transformed_df

    def save(self, path: str):
        """
        Not used in inference (training handles save).
        """
        logger.warning("[InferencePreprocessor] save() is not used in inference mode.")

    def load(self, path: str = None):
        """
        Load preprocessing pipeline trained during training.

        Parameters
        ----------
        path : str
            Optional path override. If not provided, uses preprocessor_path.
        """
        preprocessor_file = path or self.preprocessor_path
        self.pipeline = joblib.load(preprocessor_file)
        self.fitted = True
        logger.info(f"[InferencePreprocessor] Loaded preprocessor from: {preprocessor_file}")
