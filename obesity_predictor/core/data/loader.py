"""
loader.py
=========================
Module responsible for loading and inspecting the dataset.

This class centralizes dataset reading logic, provides
summary insights, and ensures consistent logging across
all data ingestion operations.

Author: Rostand Surel
"""

import pandas as pd
from obesity_predictor.config.logger_config import logger


class ObesityDataLoader:
    """
    Load and inspect the dataset for the ObesityPredictor project.

    Attributes
    ----------
    file_path : str
        Path to the CSV dataset file.
    data : pd.DataFrame
        The loaded dataset.
    """

    def __init__(self, file_path: str):
        """
        Initialize the data loader with the dataset path.

        Parameters
        ----------
        file_path : str
            Path to the CSV file containing the dataset.
        """
        self.file_path = file_path
        self.data = None

    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset into a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            The loaded dataset.
        """
        try:
            self.data = pd.read_csv(self.file_path)
            logger.success(f"[DataLoader] Data loaded successfully from: {self.file_path}")
            logger.info(f"[DataLoader] Shape: {self.data.shape[0]} rows × {self.data.shape[1]} columns")
            return self.data
        except FileNotFoundError:
            logger.error(f"[DataLoader] File not found: {self.file_path}")
            raise
        except pd.errors.EmptyDataError:
            logger.error("[DataLoader] The file is empty.")
            raise
        except Exception as e:
            logger.exception(f"[DataLoader] Unexpected error: {e}")
            raise

    def get_data(self) -> pd.DataFrame:
        """
        Return the loaded dataset.

        Returns
        -------
        pd.DataFrame
            The dataset previously loaded.
        """
        if self.data is None:
            logger.warning("[DataLoader] No data loaded yet. Call `load_data()` first.")
        return self.data

    def summarize(self, n: int = 5):
        """
        Print summary information about the dataset:
        - First rows
        - Missing values
        - Data types

        Parameters
        ----------
        n : int
            Number of rows to display from the head.
        """
        if self.data is None:
            logger.warning("[DataLoader] No data loaded yet. Call `load_data()` first.")
            return

        logger.info("[DataLoader] Showing dataset head:")
        print(self.data.head(n))

        logger.info("\n[DataLoader] Dataset info:")
        print(self.data.info())

        logger.info("\n[DataLoader] Missing values summary:")
        print(self.data.isna().sum())

        logger.info("\n[DataLoader] Unique value counts (top 10 columns):")
        print(self.data.nunique().head(10))