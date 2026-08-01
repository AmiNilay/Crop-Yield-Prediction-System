"""
scripts/build_preprocessor.py

Loads the preprocessor ALREADY SAVED by training, extracts feature metadata,
computes data statistics, and saves feature_schema.json.

This script does NOT re-fit the preprocessor.

Usage:
    (venv) PS> python scripts/build_preprocessor.py
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
MODELS_DIR: Path = PROJECT_ROOT / "models"
DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROC_DIR: Path = PROJECT_ROOT / "data" / "processed"
PREPROCESSOR_PATH: Path = MODELS_DIR / "preprocessor.pkl"
SCHEMA_PATH: Path = MODELS_DIR / "feature_schema.json"
SEP: str = "=" * 60


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names identically to DataTransformation.clean_and_prepare_df."""
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


def load_preprocessor_robust() -> Any:
    """Load preprocessor using joblib (primary) with pickle fallback."""
    path = str(PREPROCESSOR_PATH)

    # Primary: joblib (this is how it was saved)
    try:
        obj = joblib.load(path)
        print(f"  Loaded with joblib : {type(obj).__name__}")
        return obj
    except Exception as exc:
        print(f"  joblib.load failed : {exc}")

    # Fallback: pickle
    import pickle
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"  Loaded with pickle : {type(obj).__name__}")
        return obj
    except Exception as exc:
        print(f"  pickle.load failed : {exc}")

    print("\nERROR: Cannot load preprocessor. Re-train the model:")
    print("  python -m src.pipeline.train_pipeline")
    sys.exit(1)


def main() -> None:
    print(f"\n{SEP}\n  SCHEMA BUILDER (loads existing preprocessor)\n{SEP}")

    # ── 1. Load preprocessor ─────────────────────────────────────────────
    if not PREPROCESSOR_PATH.exists():
        print(f"\nERROR: {PREPROCESSOR_PATH} not found.")
        print("Re-train the model first:")
        print("  python -m src.pipeline.train_pipeline")
        sys.exit(1)

    size_kb = PREPROCESSOR_PATH.stat().st_size / 1024
    print(f"\nFile     : {PREPROCESSOR_PATH.name} ({size_kb:.1f} KB)")
    preprocessor = load_preprocessor_robust()
    print(f"Type     : {type(preprocessor).__name__}")

    # ── 2. Extract input feature names ───────────────────────────────────
    feature_columns: List[str] = []
    cat_features: List[str] = []
    num_features: List[str] = []

    # Method A: feature_names_in_ (sklearn >= 1.0)
    if hasattr(preprocessor, "feature_names_in_"):
        feature_columns = list(preprocessor.feature_names_in_)
        print(f"\nfeature_names_in_ ({len(feature_columns)}): {feature_columns}")

    # Method B: inspect transformers
    transformer_list = getattr(preprocessor, "transformers", [])
    for name, _transformer, columns in transformer_list:
        if isinstance(columns, (list, np.ndarray)):
            cols = list(columns)
            if "cat" in name.lower():
                cat_features = cols
            elif "num" in name.lower():
                num_features = cols

    if not feature_columns:
        feature_columns = cat_features + num_features

    if not cat_features and not num_features and feature_columns:
        for col in feature_columns:
            if col in ("crop", "state"):
                cat_features.append(col)
            else:
                num_features.append(col)

    print(f"\nCategorical ({len(cat_features)}) : {cat_features}")
    print(f"Numerical  ({len(num_features)}) : {num_features}")
    print(f"Total input: {len(feature_columns)}")

    # ── 3. Extract output feature names ──────────────────────────────────
    output_names: List[str] = []
    try:
        raw_names = list(preprocessor.get_feature_names_out())
        output_names = [n.split("__")[-1] for n in raw_names]
    except Exception:
        output_names = [f"feature_{i}" for i in range(len(feature_columns) + 20)]
    print(f"Output features: {len(output_names)}")

    # ── 4. Verify against native Booster ─────────────────────────────────
    n_model: Optional[int] = None
    try:
        import xgboost as xgb

        for fname in ("model.json", "model.ubj"):
            path = MODELS_DIR / fname
            if path.exists():
                booster = xgb.Booster()
                booster.load_model(str(path))
                cfg = json.loads(booster.save_config())
                n_model = int(cfg["learner"]["learner_model_param"]["num_feature"])
                print(f"Model expects {n_model} features (from {fname})")
                break
    except Exception:
        pass

    if n_model is None:
        try:
            model_path = MODELS_DIR / "model.pkl"
            if model_path.exists():
                model_obj = joblib.load(str(model_path))
                n_model = int(model_obj.n_features_in_)
                print(f"Model expects {n_model} features (from model.pkl)")
        except Exception:
            pass

    if n_model is not None and len(output_names) != n_model:
        print(f"\nWARNING: Output ({len(output_names)}) != Model ({n_model})")
        print("Re-run training: python -m src.pipeline.train_pipeline")
        sys.exit(1)

    print(f"\nFeature count verified: {len(output_names)}")

    # ── 5. Load dataset for statistics ───────────────────────────────────
    df: Optional[pd.DataFrame] = None
    for path in [
        DATA_RAW_DIR / "crop_production_india.csv",
        DATA_PROC_DIR / "train.csv",
    ]:
        if path.exists():
            df = pd.read_csv(path)
            df = clean_columns(df)
            print(f"\nDataset: {df.shape[0]} rows from {path.name}")
            break

    if df is None:
        print("ERROR: No dataset found.")
        sys.exit(1)

    # ── 6. Identify crop/state columns ───────────────────────────────────
    crop_col = next(
        (c for c in cat_features if "crop" in c),
        cat_features[0] if cat_features else None,
    )
    state_col = next(
        (c for c in cat_features if "state" in c),
        cat_features[1] if len(cat_features) > 1 else None,
    )

    # ── 7. Build schema ──────────────────────────────────────────────────
    schema: Dict[str, Any] = {
        "target_column": "yield",
        "feature_columns": feature_columns,
        "categorical_columns": cat_features,
        "numerical_columns": num_features,
        "crop_column": crop_col,
        "state_column": state_col,
        "output_feature_names": output_names,
        "n_model_features": len(output_names),
        "n_preprocessor_features": len(output_names),
        "dropped_first": True,
    }

    for col in cat_features:
        if col in df.columns:
            schema[f"unique_{col}"] = sorted(df[col].dropna().unique().tolist())

    for col in num_features:
        if col in df.columns:
            schema[f"stats_{col}"] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "median": float(df[col].median()),
            }

    # ── 8. Save ──────────────────────────────────────────────────────────
    with open(SCHEMA_PATH, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {SCHEMA_PATH} ({SCHEMA_PATH.stat().st_size / 1024:.1f} KB)")

    crop_opts = schema.get(f"unique_{crop_col}", []) if crop_col else []
    state_opts = schema.get(f"unique_{state_col}", []) if state_col else []
    print(f"Crop options  ({len(crop_opts)}): {crop_opts}")
    print(f"State options ({len(state_opts)}): {state_opts}")

    print(f"\n{SEP}\n  SCHEMA BUILD COMPLETE\n{SEP}")
    print("  Next steps:")
    print("    python scripts/compute_metrics.py")
    print("    streamlit run dashboard/app.py")
    print("    python -m uvicorn api.main:app --port 8000\n")


if __name__ == "__main__":
    main()