"""
splitter.py
=========================
Module to handle stratified train-test splitting.

Ensures consistent separation of data into training and testing
sets while maintaining class balance.

Author: Rostand Surel
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from obesity_predictor.config.logger_config import logger
from obesity_predictor.config.settings import settings


class ObesityDataSplitter:
    """
    Handles dataset splitting into train and test sets with stratification.

    Attributes
    ----------
    data : pd.DataFrame
        The complete dataset.
    target : str
        The target column name for prediction.
    test_size : float
        Fraction of the dataset to reserve for testing.
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target: str = settings.target_column,
        test_size: float = settings.test_size,
        random_state: int = settings.random_state,
    ):
        self.data = data
        self.target = target
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self, print_distribution: bool = True):
        """
        Perform the stratified train-test split.

        Parameters
        ----------
        print_distribution : bool
            If True, logs the class distribution in both sets.

        Returns
        -------
        (X_train, X_test, y_train, y_test) : tuple
            DataFrames and Series for training and testing.
        """
        logger.info("[SplitData] Splitting dataset into train and test sets...")
        X = self.data.drop(columns=[self.target])
        y = self.data[self.target]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state,
        )

        logger.success(
            f"[SplitData] Data split complete: "
            f"Train={X_train.shape[0]} | Test={X_test.shape[0]} | Features={X_train.shape[1]}"
        )

        if print_distribution:
            self._log_distribution(y_train, y_test)

        return X_train, X_test, y_train, y_test

    @staticmethod
    def _log_distribution(y_train, y_test):
        """
        Log the class distribution in both training and testing sets.

        Parameters
        ----------
        y_train : pd.Series
        y_test : pd.Series
        """
        logger.info("[SplitData] Class distribution in training set:")
        logger.info(y_train.value_counts(normalize=True))
        logger.info("[SplitData] Class distribution in testing set:")
        logger.info(y_test.value_counts(normalize=True))
