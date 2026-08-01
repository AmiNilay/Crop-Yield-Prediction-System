"""
scripts/compute_metrics.py

Standalone model evaluation script.

Loads the dataset, enriches with weather + irrigation + fertilizer,
transforms with the fitted preprocessor, predicts with the trained model,
and computes RMSE, R², MAE, MAPE metrics.

Usage:
    python scripts/compute_metrics.py
"""

import json
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HAS_XGB: bool = False
try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    pass

from src.components.fertilizer_data import (
    FERTILIZER_FEATURE,
    enrich_dataframe_with_fertilizer,
)
from src.components.irrigation_data import (
    IRRIGATION_FEATURE,
    enrich_dataframe_with_irrigation,
)
from src.components.weather_data import (
    PERIOD_LABEL,
    WEATHER_FEATURES,
    enrich_dataframe_with_weather,
)
from src.logging.logger import logging

MODELS_DIR: Path = PROJECT_ROOT / "models"
DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROC_DIR: Path = PROJECT_ROOT / "data" / "processed"
SCHEMA_PATH: Path = MODELS_DIR / "feature_schema.json"
METADATA_PATH: Path = MODELS_DIR / "model_metadata.json"
SEED: int = 42


# ---------------------------------------------------------------------------
#  Column cleaning (must match DataTransformation.clean_and_prepare_df)
# ---------------------------------------------------------------------------

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and rename yield column."""
    cleaned: List[str] = []
    for col in df.columns:
        new_col = col.strip().lower()
        new_col = re.sub(r"[^a-z0-9]+", "_", new_col)
        new_col = new_col.strip("_")
        cleaned.append(new_col)
    df.columns = cleaned

    rename_map: Dict[str, str] = {}
    for col in df.columns:
        if col.startswith("costof_"):
            rename_map[col] = "cost_of_" + col[len("costof_"):]
    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    yield_candidates = [c for c in df.columns if "yield" in c]
    if yield_candidates and yield_candidates[0] != "yield":
        df.rename(columns={yield_candidates[0]: "yield"}, inplace=True)

    return df


# ---------------------------------------------------------------------------
#  Main evaluation
# ---------------------------------------------------------------------------

def main() -> None:
    """Compute and display model evaluation metrics."""
    print()
    print("=" * 60)
    print("  MODEL EVALUATION")
    print("=" * 60)

    # ------------------------------------------------------------------
    #  1. Load schema
    # ------------------------------------------------------------------
    if not SCHEMA_PATH.exists():
        print("\n  ERROR: feature_schema.json not found.")
        print("  Run: python scripts/build_preprocessor.py")
        sys.exit(1)

    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)

    feature_cols: List[str] = schema["feature_columns"]
    output_feature_names: List[str] = schema.get("output_feature_names", [])
    crop_col: Optional[str] = schema.get("crop_column")
    state_col: Optional[str] = schema.get("state_column", "state")
    target_col: str = schema.get("target_column", "yield")

    print(f"\n  Features   : {len(feature_cols)} raw -> {len(output_feature_names)} encoded")
    print(f"  Crop col   : {crop_col}")
    print(f"  Seed       : {SEED}")
    print(f"  Weather    : {PERIOD_LABEL}")

    # ------------------------------------------------------------------
    #  2. Load preprocessor + model
    # ------------------------------------------------------------------
    pp_path = MODELS_DIR / "preprocessor.pkl"
    if not pp_path.exists():
        print("\n  ERROR: preprocessor.pkl not found.")
        sys.exit(1)

    try:
        preprocessor = joblib.load(str(pp_path))
    except Exception:
        import pickle
        with open(pp_path, "rb") as f:
            preprocessor = pickle.load(f)

    booster: Optional[Any] = None
    if HAS_XGB:
        for fname in ("model.json", "model.ubj"):
            path = MODELS_DIR / fname
            if path.exists():
                booster = xgb.Booster()
                booster.load_model(str(path))
                break

    if booster is None:
        pkl_path = MODELS_DIR / "model.pkl"
        if pkl_path.exists():
            model = joblib.load(str(pkl_path))
            if hasattr(model, "get_booster"):
                booster = model.get_booster()

    if booster is None:
        print("\n  ERROR: No model found.")
        sys.exit(1)

    # ------------------------------------------------------------------
    #  3. Load and prepare dataset
    # ------------------------------------------------------------------
    raw_path = DATA_RAW_DIR / "crop_production_india.csv"
    if not raw_path.exists():
        print(f"\n  ERROR: {raw_path} not found.")
        sys.exit(1)

    df = pd.read_csv(raw_path)
    df = clean_columns(df)

    # Enrich with weather data
    print("\n  Enriching with NASA POWER weather data...")
    try:
        df = enrich_dataframe_with_weather(df, state_col="state")
    except Exception as exc:
        print(f"  WARNING: Weather enrichment failed: {exc}")
        for feat in WEATHER_FEATURES:
            if feat not in df.columns:
                df[feat] = np.nan

    # Enrich with irrigation coverage
    print("  Enriching with irrigation coverage data...")
    try:
        df = enrich_dataframe_with_irrigation(df, state_col="state")
    except Exception as exc:
        print(f"  WARNING: Irrigation enrichment failed: {exc}")
        if IRRIGATION_FEATURE not in df.columns:
            df[IRRIGATION_FEATURE] = np.nan

    # Enrich with NPK fertilizer consumption
    print("  Enriching with NPK fertilizer data...")
    try:
        df = enrich_dataframe_with_fertilizer(df, state_col="state")
    except Exception as exc:
        print(f"  WARNING: Fertilizer enrichment failed: {exc}")
        if FERTILIZER_FEATURE not in df.columns:
            df[FERTILIZER_FEATURE] = np.nan

    print(f"\n  Dataset    : {len(df)} rows x {len(df.columns)} cols from {raw_path.name}")
    print(f"  Columns    : {list(df.columns)}")

    # Verify all feature columns are present
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        print(f"\n  ERROR: Missing columns: {missing_cols}")
        print(f"  Available: {list(df.columns)}")
        sys.exit(1)

    # ------------------------------------------------------------------
    #  4. Train/test split
    # ------------------------------------------------------------------
    if target_col not in df.columns:
        print(f"\n  ERROR: Target '{target_col}' not in dataset.")
        sys.exit(1)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED)
    print(f"  Split      : train={len(train_df)}, test={len(test_df)}")

    # ------------------------------------------------------------------
    #  5. Transform and predict
    # ------------------------------------------------------------------
    X_test = test_df[feature_cols]
    y_true = test_df[target_col].values.astype(np.float64)

    X_t = preprocessor.transform(X_test)
    if hasattr(X_t, "toarray"):
        X_t = X_t.toarray()
    X_arr = np.asarray(X_t, dtype=np.float32)

    dmat = xgb.DMatrix(X_arr, feature_names=output_feature_names or None)
    y_pred = booster.predict(dmat).astype(np.float64)

    # ------------------------------------------------------------------
    #  6. Compute metrics
    # ------------------------------------------------------------------
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    mask = y_true > 0.01
    mape = (
        float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
        if mask.any()
        else 0.0
    )

    print()
    print("-" * 40)
    print(f"  RMSE  : {rmse:>10.4f} q/ha")
    print(f"  R²    : {r2:>10.4f}")
    print(f"  MAE   : {mae:>10.4f} q/ha")
    print(f"  MAPE  : {mape:>10.2f}%")
    print(f"  N     : {len(y_true)}")
    print("-" * 40)

    # ------------------------------------------------------------------
    #  7. Per-crop metrics
    # ------------------------------------------------------------------
    per_crop: Dict[str, Dict[str, Any]] = {}
    if crop_col and crop_col in test_df.columns:
        print()
        print("  Per-Crop Breakdown:")
        print(f"  {'Crop':<25s} {'N':>4s} {'RMSE':>10s} {'MAE':>10s} {'R²':>8s}")
        print("  " + "-" * 55)

        for crop_name in sorted(test_df[crop_col].dropna().unique()):
            crop_mask = test_df[crop_col].values == crop_name
            n = int(crop_mask.sum())
            if n < 2:
                per_crop[crop_name] = {"n": n, "rmse": None, "mae": None, "r2": None}
                print(f"  {crop_name:<25s} {n:>4d} {'(skipped)':>10s} {'':>10s} {'':>8s}")
                continue
            yt = y_true[crop_mask]
            yp = y_pred[crop_mask]
            c_rmse = float(np.sqrt(mean_squared_error(yt, yp)))
            c_mae = float(mean_absolute_error(yt, yp))
            c_r2 = float(r2_score(yt, yp))
            per_crop[crop_name] = {
                "n": n,
                "rmse": round(c_rmse, 4),
                "mae": round(c_mae, 4),
                "r2": round(c_r2, 4),
            }
            print(f"  {crop_name:<25s} {n:>4d} {c_rmse:>10.4f} {c_mae:>10.4f} {c_r2:>8.4f}")

    # ------------------------------------------------------------------
    #  8. Feature importance (top 10)
    # ------------------------------------------------------------------
    if output_feature_names:
        try:
            raw_importance = booster.get_score(importance_type="gain")
            # XGBoost returns f0, f1, ... by default when trained via sklearn wrapper
            # Map to actual feature names
            mapped_importance = {}
            for key, val in raw_importance.items():
                if key.startswith("f") and key[1:].isdigit():
                    idx = int(key[1:])
                    if idx < len(output_feature_names):
                        mapped_importance[output_feature_names[idx]] = val
                    else:
                        mapped_importance[key] = val
                else:
                    mapped_importance[key] = val
            feat_imp = []
            for fname in output_feature_names:
                gain = mapped_importance.get(fname, 0.0)
                feat_imp.append({"feature": fname, "gain": gain})
            feat_imp.sort(key=lambda x: x["gain"], reverse=True)

            print()
            print("  Top 10 Features (XGBoost gain):")
            print(f"  {'Feature':<40s} {'Gain':>12s}")
            print("  " + "-" * 55)
            for item in feat_imp[:10]:
                print(f"  {item['feature']:<40s} {item['gain']:>12.2f}")
        except Exception:
            pass

    # ------------------------------------------------------------------
    #  9. Save metrics to metadata
    # ------------------------------------------------------------------
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    metadata["evaluation"] = {
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "mae": round(mae, 4),
        "mape": round(mape, 2),
        "n_samples": int(len(y_true)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "data_source": "train_test_split",
        "seed": SEED,
        "per_crop": per_crop,
        "weather_period": PERIOD_LABEL,
        "features": {
            "total_input": len(feature_cols),
            "total_encoded": len(output_feature_names),
            "weather": WEATHER_FEATURES,
            "irrigation": IRRIGATION_FEATURE,
            "fertilizer": FERTILIZER_FEATURE,
        },
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\n  Metrics saved -> {METADATA_PATH}")

    print()
    print("=" * 60)
    print("  EVALUATION COMPLETE")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()