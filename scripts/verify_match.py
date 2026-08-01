"""scripts/verify_match.py — verify model and preprocessor feature counts match."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import xgboost as xgb

MODELS_DIR = Path("models")

# Load preprocessor
pp = joblib.load(str(MODELS_DIR / "preprocessor.pkl"))
pp_features = pp.transform(
    __import__("pandas").DataFrame(
        columns=[
            "crop", "state", "season", "crop_year", "area",
            "mean_temperature", "total_precipitation",
            "mean_relative_humidity", "mean_solar_radiation",
            "irrigation_coverage_pct", "npk_consumption_kg_per_ha",
        ]
    )
)
n_preprocessor = pp_features.shape[1] if hasattr(pp_features, "shape") else 0

# Load model
booster = xgb.Booster()
booster.load_model(str(MODELS_DIR / "model.json"))
n_model = int(booster.num_features())

# Load schema
with open(MODELS_DIR / "feature_schema.json") as f:
    schema = json.load(f)
n_schema = schema["n_output_features"]

print(f"Preprocessor outputs: {n_preprocessor} features")
print(f"Model expects:        {n_model} features")
print(f"Schema says:          {n_schema} features")

if n_preprocessor == n_model == n_schema:
    print("\nALL MATCH — ready to deploy!")
else:
    print("\nMISMATCH DETECTED — do not deploy!")
    sys.exit(1)