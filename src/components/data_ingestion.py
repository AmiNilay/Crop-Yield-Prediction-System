import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.logging.logger import logging
from src.exception.exception import CustomException
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    # Point to the new raw data file
    raw_data_path: str = os.path.join('data', 'raw', 'crop_production_india.csv')
    train_data_path: str = os.path.join('data', 'processed', 'train.csv')
    test_data_path: str = os.path.join('data', 'processed', 'test.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")
        try:
            # Read the large, real-world dataset
            df = pd.read_csv(self.ingestion_config.raw_data_path)
            logging.info(f"Raw data read successfully from {self.ingestion_config.raw_data_path}")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Split the data into training and testing sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data Ingestion is completed")
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            logging.error(f"Error in Data Ingestion: {e}")
            raise CustomException(e, sys)