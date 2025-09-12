import os
import sys
from dataclasses import dataclass

# Import new libraries
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
# ---

from sklearn.metrics import r2_score
from src.exception.exception import CustomException
from src.logging.logger import logging
from src.utils.common import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("models", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            # --- NEW: Hyperparameter Tuning with GridSearchCV ---
            
            # 1. Initialize the model
            xgb = XGBRegressor(random_state=42)
            
            # 2. Define a small parameter grid to search
            # For a real project, this grid would be much larger.
            param_grid = {
                'n_estimators': [50, 100],
                'learning_rate': [0.1, 0.05],
                'max_depth': [3, 5]
            }
            
            # 3. Set up GridSearchCV
            # cv=3 means 3-fold cross-validation. scoring='r2' is the metric to optimize.
            # n_jobs=-1 uses all available CPU cores to speed up the search.
            grid_search = GridSearchCV(
                estimator=xgb, 
                param_grid=param_grid, 
                cv=3, 
                n_jobs=-1, 
                verbose=2, 
                scoring='r2'
            )
            
            logging.info("Starting Hyperparameter Tuning with GridSearchCV")
            grid_search.fit(X_train, y_train)
            
            # 4. Get the best model from the search
            best_model = grid_search.best_estimator_
            logging.info(f"Best parameters found: {grid_search.best_params_}")

            # 5. Evaluate the best model on the test set
            y_pred = best_model.predict(X_test)
            score = r2_score(y_test, y_pred)
            logging.info(f"Model R2 Score on Test Set: {score}")

            if score < 0.1: # Kept the low threshold for the sample data
                 raise CustomException(Exception("Model performance is too low after tuning"), sys)

            # 6. Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            logging.info("Best model found via GridSearchCV has been saved.")
            return score

        except Exception as e:
            raise CustomException(e, sys)