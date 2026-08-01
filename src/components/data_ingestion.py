"""
src/components/data_ingestion.py

Data ingestion: read CSV, split train/test with stratification by crop.
"""

import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception.exception import CustomException
from src.logging.logger import logging
from src.utils.common import save_object

DATA_PATH: str = os.path.join("data", "raw", "crop_production_india.csv")


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("data", "processed", "train.csv")
    test_data_path: str = os.path.join("data", "processed", "test.csv")
    raw_data_path: str = os.path.join("data", "processed", "crop_production_india.csv")


class DataIngestion:
    def __init__(self) -> None:
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """Read raw CSV, stratified split by crop, save train/test."""
        logging.info("Entered data ingestion")
        try:
            df = pd.read_csv(DATA_PATH)
            logging.info("Read dataset: %d rows, %d cols", len(df), len(df.columns))

            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)

            # Save raw copy
            df.to_csv(self.config.raw_data_path, index=False)
            logging.info("Raw data saved -> %s", self.config.raw_data_path)

            # Stratified split by crop
            logging.info("Performing stratified train/test split (stratify=crop)")
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42,
                stratify=df["crop"],
            )

            train_set.to_csv(self.config.train_data_path, index=False)
            test_set.to_csv(self.config.test_data_path, index=False)

            logging.info(
                "Split complete: train=%d, test=%d",
                len(train_set),
                len(test_set),
            )

            return (
                self.config.train_data_path,
                self.config.test_data_path,
            )

        except Exception as e:
            logging.error("Data ingestion failed: %s", e)
            raise CustomException(e, sys)