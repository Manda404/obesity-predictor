"""
base_preprocessor.py
=========================
Abstract base class for preprocessing logic used in both training
and inference pipelines.

Defines the structure all preprocessors must follow:
- `fit()` for learning transformations on training data.
- `transform()` for applying transformations.
- `save()` / `load()` for persistence.

Author: Rostand Surel
"""

from abc import ABC, abstractmethod
import pandas as pd
from obesity_predictor.config.logger_config import logger


class BasePreprocessor(ABC):
    """
    Abstract base preprocessor class defining the interface
    for all preprocessors in the project.
    """

    def __init__(self):
        self.fitted = False

    @abstractmethod
    def fit(self, data: pd.DataFrame):
        """
        Learn transformation parameters from the dataset.

        Parameters
        ----------
        data : pd.DataFrame
            The training dataset used to learn preprocessing steps.
        """
        pass

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply learned transformations to the dataset.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset to transform.

        Returns
        -------
        pd.DataFrame
            The transformed dataset.
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """
        Save preprocessing artifacts (e.g., encoders, scalers).

        Parameters
        ----------
        path : str
            Path to save the preprocessor artifacts.
        """
        pass

    @abstractmethod
    def load(self, path: str):
        """
        Load preprocessing artifacts.

        Parameters
        ----------
        path : str
            Path from which to load the preprocessor artifacts.
        """
        pass

    def _check_fitted(self):
        """
        Utility method to ensure preprocessor has been fitted
        before transformation.
        """
        if not self.fitted:
            logger.error("[BasePreprocessor] Attempted to transform before fitting.")
            raise RuntimeError("Preprocessor must be fitted before calling transform().")