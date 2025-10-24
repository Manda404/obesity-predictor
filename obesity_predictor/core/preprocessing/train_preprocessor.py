"""
train_preprocessor.py
=========================
Implements the preprocessing logic for training data:
- Encoding categorical variables
- Scaling numerical variables
- Generating derived features (e.g. BMI)
- Persisting encoders/scalers for future inference

Author: Rostand Surel
"""

import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from obesity_predictor.config.logger_config import logger
from obesity_predictor.core.preprocessing.base_preprocessor import BasePreprocessor
from obesity_predictor.config.settings import settings


class TrainPreprocessor(BasePreprocessor):
    """
    Preprocessor for training phase: fit and transform the data.
    """

    def __init__(self):
        super().__init__()
        self.pipeline = None
        self.feature_names = None

    def fit(self, data: pd.DataFrame):
        """
        Fit transformation pipeline on training data.

        Parameters
        ----------
        data : pd.DataFrame
            Raw input dataset.
        """
        logger.info("[TrainPreprocessor] Starting preprocessing fit...")

        # --- Derived features ---
        data["BMI"] = data["Weight"] / (data["Height"] ** 2)
        data["Age_Group"] = pd.cut(data["Age"], bins=[0, 18, 30, 50, 100], labels=["Teen", "Young", "Adult", "Senior"])

        # --- Identify types ---
        numeric_cols = data.select_dtypes(include="number").columns.tolist()
        categorical_cols = data.select_dtypes(exclude="number").columns.tolist()

        if settings.target_column in categorical_cols:
            categorical_cols.remove(settings.target_column)
        if settings.target_column in numeric_cols:
            numeric_cols.remove(settings.target_column)

        logger.debug(f"[TrainPreprocessor] Numeric cols: {numeric_cols}")
        logger.debug(f"[TrainPreprocessor] Categorical cols: {categorical_cols}")

        # --- Transformers ---
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

        self.pipeline = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols),
            ]
        )

        # Fit the pipeline
        self.pipeline.fit(data)
        self.feature_names = (
            numeric_cols + list(self.pipeline.named_transformers_["cat"].get_feature_names_out(categorical_cols))
        )

        self.fitted = True
        logger.success("[TrainPreprocessor] Preprocessing fit completed successfully.")

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted transformations to dataset.

        Parameters
        ----------
        data : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            Preprocessed dataset ready for training or inference.
        """
        self._check_fitted()
        logger.info("[TrainPreprocessor] Transforming dataset...")

        data["BMI"] = data["Weight"] / (data["Height"] ** 2)
        data["Age_Group"] = pd.cut(data["Age"], bins=[0, 18, 30, 50, 100], labels=["Teen", "Young", "Adult", "Senior"])

        transformed_array = self.pipeline.transform(data)
        transformed_df = pd.DataFrame(transformed_array, columns=self.feature_names)

        logger.success(f"[TrainPreprocessor] Transformation complete. Shape: {transformed_df.shape}")
        return transformed_df

    def save(self, path: str):
        """
        Save the fitted preprocessing pipeline.

        Parameters
        ----------
        path : str
            Destination path for the preprocessor file.
        """
        joblib.dump(self.pipeline, path)
        logger.info(f"[TrainPreprocessor] Preprocessing pipeline saved to: {path}")

    def load(self, path: str):
        """
        Load a previously saved preprocessing pipeline.

        Parameters
        ----------
        path : str
        """
        self.pipeline = joblib.load(path)
        self.fitted = True
        logger.info(f"[TrainPreprocessor] Loaded preprocessing pipeline from: {path}")
