"""
scripts/build_preprocessor.py

Builds (or rebuilds) the fitted preprocessor and feature schema.

Outputs:
  models/preprocessor.pkl        — fitted ColumnTransformer
  models/feature_schema.json     — metadata for dashboard + API

Run after dataset changes or when models/preprocessor.pkl is missing.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.components.fertilizer_data import (
    FERTILIZER_FEATURE,
    enrich_dataframe_with_fertilizer,
)
from src.components.irrigation_data import (
    IRRIGATION_FEATURE,
    enrich_dataframe_with_irrigation,
)
from src.components.weather_data import (
    WEATHER_FEATURES,
    enrich_dataframe_with_weather,
)
from src.utils.common import save_object

MODELS_DIR = PROJECT_ROOT / "models"
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROC_DIR = PROJECT_ROOT / "data" / "processed"
SCHEMA_PATH = MODELS_DIR / "feature_schema.json"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"

TARGET_COLUMN = "yield"
SEED = 42


# ---------------------------------------------------------------------------
#  Column cleaning (mirrors data_transformation.py)
# ---------------------------------------------------------------------------

import re


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = []
    for col in df.columns:
        new_col = col.strip().lower()
        new_col = re.sub(r"[^a-z0-9]+", "_", new_col)
        cleaned.append(new_col.strip("_"))
    df.columns = cleaned

    # Fix costof_ prefix
    rename_map = {}
    for col in df.columns:
        if col.startswith("costof_"):
            rename_map[col] = "cost_of_" + col[len("costof_"):]
    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    # Normalize yield column
    yield_candidates = [c for c in df.columns if "yield" in c]
    if yield_candidates and yield_candidates[0] != "yield":
        df.rename(columns={yield_candidates[0]: "yield"}, inplace=True)

    # Drop production and district if present
    for exc in ["production", "district"]:
        if exc in df.columns:
            df.drop(columns=[exc], inplace=True)

    return df


# ---------------------------------------------------------------------------
#  Find dataset
# ---------------------------------------------------------------------------

def find_dataset() -> pd.DataFrame:
    """Load dataset from raw or processed directory."""
    candidates = [
        DATA_RAW_DIR / "crop_production_india.csv",
        DATA_PROC_DIR / "train.csv",
    ]
    for path in candidates:
        if path.exists():
            print(f"  Loading: {path}")
            df = pd.read_csv(path)
            df = clean_columns(df)
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            return df
    raise FileNotFoundError(
        "No dataset found. Place CSV at data/raw/crop_production_india.csv"
    )


# ---------------------------------------------------------------------------
#  Enrichment
# ---------------------------------------------------------------------------

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add weather, irrigation, and fertilizer features."""
    if "state" not in df.columns:
        print("  WARNING: No 'state' column — skipping enrichment")
        return df

    # Weather
    missing_wx = [c for c in WEATHER_FEATURES if c not in df.columns]
    if missing_wx:
        print(f"  Enriching with weather ({len(missing_wx)} features)...")
        try:
            df = enrich_dataframe_with_weather(df, state_col="state")
        except Exception as exc:
            print(f"  WARNING: Weather enrichment failed: {exc}")
            for feat in WEATHER_FEATURES:
                if feat not in df.columns:
                    df[feat] = np.nan

    # Irrigation
    if IRRIGATION_FEATURE not in df.columns:
        print("  Enriching with irrigation coverage...")
        try:
            df = enrich_dataframe_with_irrigation(df, state_col="state")
        except Exception as exc:
            print(f"  WARNING: Irrigation enrichment failed: {exc}")
            df[IRRIGATION_FEATURE] = np.nan

    # Fertilizer
    if FERTILIZER_FEATURE not in df.columns:
        print("  Enriching with NPK fertilizer...")
        try:
            df = enrich_dataframe_with_fertilizer(df, state_col="state")
        except Exception as exc:
            print(f"  WARNING: Fertilizer enrichment failed: {exc}")
            df[FERTILIZER_FEATURE] = np.nan

    return df


# ---------------------------------------------------------------------------
#  Build preprocessor and schema
# ---------------------------------------------------------------------------

def build(df: pd.DataFrame) -> None:
    """Fit ColumnTransformer and extract feature schema."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Define feature columns ──
    # Categorical features
    cat_cols = []
    for col in ["crop", "state", "season"]:
        if col in df.columns:
            cat_cols.append(col)

    # Numerical features (in order)
    num_cols = []
    candidates = [
        "crop_year",
        "area",
        "mean_temperature",
        "total_precipitation",
        "mean_relative_humidity",
        "mean_solar_radiation",
        IRRIGATION_FEATURE,
        FERTILIZER_FEATURE,
    ]
    for col in candidates:
        if col in df.columns:
            num_cols.append(col)

    feature_columns = cat_cols + num_cols
    print(f"\n  Feature columns ({len(feature_columns)}): {feature_columns}")

    # ── Drop rows with missing features or target ──
    required = feature_columns + [TARGET_COLUMN]
    available = [c for c in required if c in df.columns]
    df_clean = df[available].dropna()
    print(f"  Rows after dropping NaN: {len(df_clean)} (dropped {len(df) - len(df_clean)})")

    if len(df_clean) < 10:
        raise ValueError(f"Only {len(df_clean)} valid rows — cannot build preprocessor")

    # ── Fit ColumnTransformer ──
    print("\n  Fitting ColumnTransformer...")
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop",
    )

    X = df_clean[feature_columns]
    preprocessor.fit(X)

    # ── Extract output feature names ──
    output_feature_names = []

    # OHE feature names
    ohe = preprocessor.named_transformers_["cat"]
    if hasattr(ohe, "get_feature_names_out"):
        ohe_names = list(ohe.get_feature_names_out(cat_cols))
    else:
        ohe_names = []
        for i, col in enumerate(cat_cols):
            categories = ohe.categories_[i]
            for cat in categories:
                ohe_names.append(f"{col}_{cat}")
    output_feature_names.extend(ohe_names)

    # Numerical feature names (pass through as-is)
    output_feature_names.extend(num_cols)

    print(f"  Output features after encoding: {len(output_feature_names)}")

    # ── Save preprocessor ──
    save_object(PREPROCESSOR_PATH, preprocessor)
    print(f"  Saved: {PREPROCESSOR_PATH}")

    # ── Extract unique values for dropdowns ──
    unique_values = {}
    for col in ["crop", "state", "season"]:
        if col in df_clean.columns:
            unique_values[f"unique_{col}"] = sorted(
                df_clean[col].dropna().unique().tolist()
            )

    # Years
    if "crop_year" in df_clean.columns:
        unique_values["unique_crop_year"] = sorted(
            df_clean["crop_year"].dropna().unique().tolist()
        )

    # Area statistics
    stats_area = {}
    if "area" in df_clean.columns:
        area = df_clean["area"].dropna()
        stats_area = {
            "min": round(float(area.min()), 1),
            "max": round(float(area.max()), 1),
            "mean": round(float(area.mean()), 1),
            "median": round(float(area.median()), 1),
            "std": round(float(area.std()), 1),
        }

    # ── Build schema ──
    schema = {
        "feature_columns": feature_columns,
        "categorical_columns": cat_cols,
        "numerical_columns": num_cols,
        "target_column": TARGET_COLUMN,
        "output_feature_names": output_feature_names,
        "n_output_features": len(output_feature_names),
        "crop_column": "crop" if "crop" in df_clean.columns else None,
        "state_column": "state" if "state" in df_clean.columns else None,
        "stats_area": stats_area,
        "n_training_samples": len(df_clean),
        **unique_values,
    }

    with open(SCHEMA_PATH, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, default=str)
    print(f"  Saved: {SCHEMA_PATH}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  BUILD COMPLETE")
    print("=" * 60)
    print(f"  Training samples:  {len(df_clean)}")
    print(f"  Input features:    {len(feature_columns)}")
    print(f"  Output features:   {len(output_feature_names)}")
    print(f"  Categorical:       {cat_cols}")
    print(f"  Numerical:         {num_cols}")
    print(f"  Crops:             {len(unique_values.get('unique_crop', []))}")
    print(f"  States:            {len(unique_values.get('unique_state', []))}")
    print(f"  Seasons:           {len(unique_values.get('unique_season', []))}")
    print(f"  Years:             {len(unique_values.get('unique_crop_year', []))}")
    print(f"  Area range:        {stats_area.get('min', 0)} - {stats_area.get('max', 0)} Ha")
    print(f"\n  Files written:")
    print(f"    {PREPROCESSOR_PATH}")
    print(f"    {SCHEMA_PATH}")
    print()


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  BUILD PREPROCESSOR & FEATURE SCHEMA")
    print("=" * 60)

    df = find_dataset()
    df = enrich(df)
    build(df)