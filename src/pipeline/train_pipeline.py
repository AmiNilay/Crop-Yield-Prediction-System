import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception.exception import CustomException

class TrainPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def run_pipeline(self):
        try:
            train_data_path, test_data_path = self.data_ingestion.initiate_data_ingestion()
            train_arr, test_arr, _ = self.data_transformation.initiate_data_transformation(train_data_path, test_data_path)
            r2_score = self.model_trainer.initiate_model_trainer(train_arr, test_arr)
            print(f"Training completed. Model R2 Score: {r2_score}")
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()