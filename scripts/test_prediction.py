"""scripts/test_prediction.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

data = CustomData("RICE", "Punjab", "KHARIF", 2014, 1.0)
df = data.get_data_as_dataframe()
pipeline = PredictPipeline()
pred = pipeline.predict(df)
print(f"Prediction: {pred[0]:.2f} q/ha")
print("SUCCESS")