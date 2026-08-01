import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception.exception import CustomException
from src.logging.logger import logging


class TrainPipeline:
    """Orchestrates the full training pipeline: ingest → transform → train."""

    def __init__(self) -> None:
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def run_pipeline(self) -> float:
        """Execute the complete pipeline.

        Returns:
            R² score on the held-out test set.
        """
        try:
            logging.info("=" * 60)
            logging.info("  TRAIN PIPELINE STARTED")
            logging.info("=" * 60)

            # Step 1: Ingest
            train_path, test_path = self.data_ingestion.initiate_data_ingestion()

            # Step 2: Transform
            train_arr, test_arr, _ = (
                self.data_transformation.initiate_data_transformation(
                    train_path, test_path
                )
            )

            # Step 3: Train
            r2_score = self.model_trainer.initiate_model_trainer(
                train_arr, test_arr
            )

            logging.info("=" * 60)
            logging.info("  TRAIN PIPELINE COMPLETED — R²=%.4f", r2_score)
            logging.info("=" * 60)

            print(f"\nTraining completed. R² Score: {r2_score:.4f}")
            return r2_score

        except Exception as e:
            logging.error("Train pipeline failed: %s", e)
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()