"""
src/pipeline/predict_pipeline.py

Production inference pipeline with optional farmer overrides.

If irrigation_pct or npk_consumption_kg_per_ha are provided by the user,
they replace the state-average values from the static datasets.
"""

import sys
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.components.fertilizer_data import (
    FERTILIZER_FEATURE,
    get_fertilizer_for_state,
)
from src.components.irrigation_data import (
    IRRIGATION_FEATURE,
    get_irrigation_for_state,
)
from src.components.weather_data import (
    WEATHER_FEATURES,
    enrich_dataframe_with_weather,
    get_weather_for_state,
)
from src.exception.exception import CustomException
from src.logging.logger import logging
from src.utils.common import load_booster_from_native, load_object

try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False


_FEATURE_COLUMNS = [
    "crop", "state", "season",
    "crop_year", "area",
    "mean_temperature", "total_precipitation",
    "mean_relative_humidity", "mean_solar_radiation",
    "irrigation_coverage_pct", "npk_consumption_kg_per_ha",
]


class PredictPipeline:
    """Production inference pipeline.

    Auto-enriches with weather, irrigation, and fertilizer.
    Supports optional farmer overrides for irrigation and NPK.
    """

    def __init__(self) -> None:
        self.preprocessor = load_object("models/preprocessor.pkl")
        self._booster = load_booster_from_native("models")
        self._sklearn_model: Optional[Any] = None
        if self._booster is not None:
            logging.info("PredictPipeline: using native Booster")
        else:
            logging.info("PredictPipeline: falling back to model.pkl")
            self._sklearn_model = load_object("models/model.pkl")

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        try:
            missing_weather = [c for c in WEATHER_FEATURES if c not in features.columns]
            if missing_weather:
                features = enrich_dataframe_with_weather(features, state_col="state")
            if IRRIGATION_FEATURE not in features.columns:
                from src.components.irrigation_data import enrich_dataframe_with_irrigation
                features = enrich_dataframe_with_irrigation(features, state_col="state")
            if FERTILIZER_FEATURE not in features.columns:
                from src.components.fertilizer_data import enrich_dataframe_with_fertilizer
                features = enrich_dataframe_with_fertilizer(features, state_col="state")

            X_transformed = self.preprocessor.transform(features)
            if hasattr(X_transformed, "toarray"):
                X_transformed = X_transformed.toarray()
            X_arr = np.asarray(X_transformed, dtype=np.float32)

            if self._booster is not None and HAS_XGB:
                dmat = xgb.DMatrix(X_arr)
                return self._booster.predict(dmat)
            else:
                return self._sklearn_model.predict(X_arr)
        except Exception as e:
            logging.error("Prediction failed: %s", e)
            raise CustomException(e, sys)


class CustomData:
    """Data transfer object for user inputs.

    Supports optional farmer overrides for irrigation and NPK.
    If override values are None, state averages are used.
    """

    def __init__(
        self,
        crop: str,
        state: str,
        season: str,
        crop_year: int,
        area: float = 1.0,
        irrigation_override: Optional[float] = None,
        npk_override: Optional[float] = None,
    ) -> None:
        self.crop = crop
        self.state = state
        self.season = season
        self.crop_year = crop_year
        self.area = area
        self.irrigation_override = irrigation_override
        self.npk_override = npk_override

    def get_data_as_dataframe(self) -> pd.DataFrame:
        """Return a single-row DataFrame with all required features.

        Weather is auto-fetched based on state + crop_year.
        Irrigation and NPK use farmer overrides if provided,
        otherwise fall back to state averages.
        """
        try:
            weather = get_weather_for_state(self.state, year=self.crop_year)

            # Irrigation: override or state average
            if self.irrigation_override is not None:
                irrigation = self.irrigation_override
                logging.info("Using farmer irrigation override: %.1f%%", irrigation)
            else:
                irrigation = get_irrigation_for_state(self.state)
                logging.info("Using state-average irrigation: %.1f%%", irrigation)

            # NPK: override or state average
            if self.npk_override is not None:
                npk = self.npk_override
                logging.info("Using farmer NPK override: %.1f kg/ha", npk)
            else:
                npk = get_fertilizer_for_state(self.state)
                logging.info("Using state-average NPK: %.1f kg/ha", npk)

            return pd.DataFrame(
                {
                    "crop": [self.crop],
                    "state": [self.state],
                    "season": [self.season],
                    "crop_year": [self.crop_year],
                    "area": [self.area],
                    "mean_temperature": [weather["mean_temperature"]],
                    "total_precipitation": [weather["total_precipitation"]],
                    "mean_relative_humidity": [weather["mean_relative_humidity"]],
                    "mean_solar_radiation": [weather["mean_solar_radiation"]],
                    "irrigation_coverage_pct": [irrigation],
                    "npk_consumption_kg_per_ha": [npk],
                },
                columns=_FEATURE_COLUMNS,
            )
        except Exception as e:
            raise CustomException(e, sys)