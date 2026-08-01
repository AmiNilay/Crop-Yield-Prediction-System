import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import joblib

from src.exception.exception import CustomException
from src.logging.logger import logging

try:
    import xgboost as xgb  # type: ignore[import-untyped]

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import pickle as _pickle
except ImportError:
    _pickle = None  # type: ignore[assignment]


def save_object(file_path: str, obj: Any) -> None:
    """Save a Python object to disk using joblib.

    If the object is an XGBoost Booster, also saves native JSON and UBJSON
    formats for cross-version compatibility and SHAP support.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(obj, file_path)
        logging.info("Saved object -> %s", file_path)

        # Auto-save native XGBoost format
        if HAS_XGB:
            booster: Optional[Any] = None
            if isinstance(obj, xgb.Booster):
                booster = obj
            elif hasattr(obj, "get_booster"):
                try:
                    booster = obj.get_booster()
                except Exception:
                    pass

            if booster is not None:
                json_path = str(Path(file_path).with_suffix(".json"))
                ubj_path = str(Path(file_path).with_suffix(".ubj"))
                booster.save_model(json_path)
                booster.save_model(ubj_path)
                logging.info("Saved native Booster -> %s, %s", json_path, ubj_path)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str) -> Any:
    """Load a Python object from a joblib or pickle file."""
    try:
        obj = joblib.load(file_path)
        logging.info("Loaded object <- %s", file_path)
        return obj
    except Exception as e:
        raise CustomException(e, sys)


def load_booster_from_native(model_dir: str = "models") -> Optional[Any]:
    """Load an XGBoost Booster from native JSON/UBJSON (no pickle warnings)."""
    if not HAS_XGB:
        return None
    for fname in ("model.json", "model.ubj"):
        path = os.path.join(model_dir, fname)
        if os.path.exists(path):
            try:
                booster = xgb.Booster()
                booster.load_model(path)
                logging.info("Loaded native Booster <- %s", path)
                return booster
            except Exception:
                continue
    return None


def save_json(file_path: str, data: Dict[str, Any]) -> None:
    """Save a dictionary as formatted JSON."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logging.info("Saved JSON -> %s", file_path)
    except Exception as e:
        raise CustomException(e, sys)


def load_json(file_path: str) -> Dict[str, Any]:
    """Load a JSON file into a dictionary."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise CustomException(e, sys)