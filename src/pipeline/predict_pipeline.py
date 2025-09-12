import sys
import pandas as pd
from src.exception.exception import CustomException
from src.utils.common import load_object

class PredictPipeline:
    def __init__(self):
        self.model_path = 'models/model.pkl'
        self.preprocessor_path = 'models/preprocessor.pkl'
        self.model = load_object(file_path=self.model_path)
        self.preprocessor = load_object(file_path=self.preprocessor_path)

    def predict(self, features):
        try:
            data_scaled = self.preprocessor.transform(features)
            preds = self.model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

# This class now creates a DataFrame with the EXACT column names the model was trained on.
class CustomData:
    def __init__(self,
                 crop: str,
                 state: str,
                 cost_of_cultivation_a2_fl: float,
                 cost_of_cultivation_c2: float,
                 cost_of_production_c2: float):
        self.crop = crop
        self.state = state
        self.cost_of_cultivation_a2_fl = cost_of_cultivation_a2_fl
        self.cost_of_cultivation_c2 = cost_of_cultivation_c2
        self.cost_of_production_c2 = cost_of_production_c2

    def get_data_as_dataframe(self):
        try:
            # --- THIS IS THE FIX ---
            # The dictionary keys now precisely match the feature names in data_transformation.py
            custom_data_input_dict = {
                "crop": [self.crop],
                "state": [self.state],
                "cost_of_cultivation_hectare_a2_fl": [self.cost_of_cultivation_a2_fl],
                "cost_of_cultivation_hectare_c2": [self.cost_of_cultivation_c2],
                "cost_of_production_quintal_c2": [self.cost_of_production_c2]
            }
            # --- END OF FIX ---
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)