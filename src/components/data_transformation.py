"""
src/components/data_transformation.py

Data cleaning, feature enrichment (weather, irrigation, fertilizer),
and sklearn preprocessor construction.

Feature schema (new dataset):
  Categorical : crop, state, season
  Numerical   : crop_year, area, mean_temperature, total_precipitation,
                 mean_relative_humidity, mean_solar_radiation,
                 irrigation_coverage_pct, npk_consumption_kg_per_ha
  Target      : yield (Quintals/Hectare)
  Excluded    : production (leakage), district (too many OHE columns)
"""

import os
import re
import sys
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.components.fertilizer_data import (
    FERTILIZER_FEATURE,
    enrich_dataframe_with_fertilizer,
)
from src.components.irrigation_data import (
    IRRIGATION_FEATURE,
    enrich_dataframe_with_irrigation,
)
from src.components.weather_data import WEATHER_FEATURES, enrich_dataframe_with_weather
from src.exception.exception import CustomException
from src.logging.logger import logging
from src.utils.common import save_object

# ---------------------------------------------------------------------------
#  Constants — single source of truth for feature names
# ---------------------------------------------------------------------------
CATEGORICAL_FEATURES: List[str] = ["crop", "state", "season"]

NUMERICAL_FEATURES: List[str] = [
    # Dataset features (2)
    "crop_year",
    "area",
    # NASA POWER weather features (4)
    "mean_temperature",
    "total_precipitation",
    "mean_relative_humidity",
    "mean_solar_radiation",
    # Irrigation coverage (1)
    "irrigation_coverage_pct",
    # NPK fertilizer consumption (1)
    "npk_consumption_kg_per_ha",
]

TARGET_COLUMN: str = "yield"

# Columns that must NEVER be features (leakage, too-many-categories, etc.)
EXCLUDED_COLUMNS: List[str] = [
    "production",   # data leakage: yield = production / area
    "district",     # 646 unique values — would explode OHE
]


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("models", "preprocessor.pkl")


class DataTransformation:
    def __init__(self) -> None:
        self.config = DataTransformationConfig()

    # ------------------------------------------------------------------
    #  Column cleaning (shared by train & test)
    # ------------------------------------------------------------------

    @staticmethod
    def clean_and_prepare_df(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names and extract the target.

        Steps:
            1. Lowercase + strip whitespace
            2. Replace non-alphanumeric runs with ``_``
            3. Handle the known ``Costof`` -> ``cost_of`` typo (legacy compat)
            4. Rename the yield column to ``yield``
            5. Drop excluded columns (production, district)
        """
        logging.info("Original columns: %s", df.columns.tolist())

        cleaned: List[str] = []
        for col in df.columns:
            new_col = col.strip().lower()
            new_col = re.sub(r"[^a-z0-9]+", "_", new_col)
            new_col = new_col.strip("_")
            cleaned.append(new_col)
        df.columns = cleaned

        # Legacy: fix costof_ typo from old dataset
        rename_map: dict = {}
        for col in df.columns:
            if col.startswith("costof_"):
                rename_map[col] = "cost_of_" + col[len("costof_"):]
        if rename_map:
            df.rename(columns=rename_map, inplace=True)
            logging.info("Applied legacy typo fixes: %s", rename_map)

        # Rename yield column
        yield_candidates = [c for c in df.columns if "yield" in c]
        if not yield_candidates:
            raise KeyError(
                "No column containing 'yield' found after cleaning. "
                "Check the source CSV."
            )
        if yield_candidates[0] != TARGET_COLUMN:
            df.rename(columns={yield_candidates[0]: TARGET_COLUMN}, inplace=True)
            logging.info("Renamed '%s' -> '%s'", yield_candidates[0], TARGET_COLUMN)

        # Drop excluded columns (production, district) if present
        for exc_col in EXCLUDED_COLUMNS:
            if exc_col in df.columns:
                df.drop(columns=[exc_col], inplace=True)
                logging.info("Dropped excluded column: '%s'", exc_col)

        logging.info("Cleaned columns: %s", df.columns.tolist())

        return df

    # ------------------------------------------------------------------
    #  Feature enrichment
    # ------------------------------------------------------------------

    @staticmethod
    def enrich_with_weather(df: pd.DataFrame) -> pd.DataFrame:
        """Add NASA POWER weather features to the DataFrame."""
        return enrich_dataframe_with_weather(df, state_col="state")

    @staticmethod
    def enrich_with_irrigation(df: pd.DataFrame) -> pd.DataFrame:
        """Add irrigation coverage feature to the DataFrame."""
        return enrich_dataframe_with_irrigation(df, state_col="state")

    @staticmethod
    def enrich_with_fertilizer(df: pd.DataFrame) -> pd.DataFrame:
        """Add NPK fertilizer consumption feature to the DataFrame."""
        return enrich_dataframe_with_fertilizer(df, state_col="state")

    # ------------------------------------------------------------------
    #  Preprocessor builder
    # ------------------------------------------------------------------

    def get_data_transformer_object(self) -> ColumnTransformer:
        """Build the sklearn ColumnTransformer.

        Output column order: [categorical OHE, numerical scaled]
        This order MUST match between training, build_preprocessor.py,
        and inference.

        Categorical: OneHotEncoder with ``drop='first'`` and
        ``handle_unknown='ignore'`` for unseen categories at inference.

        Numerical: Median imputation + StandardScaler.
        Includes 2 dataset + 4 weather + 1 irrigation + 1 fertilizer = 8 numerical.
        """
        try:
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                            drop="first",
                        ),
                    ),
                ]
            )

            # CRITICAL: cat first, then num — must match build_preprocessor.py
            preprocessor = ColumnTransformer(
                transformers=[
                    ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
                    ("num", numerical_pipeline, NUMERICAL_FEATURES),
                ],
                remainder="drop",
            )

            logging.info(
                "Preprocessor built: cat=%s, num=%s (%d features), drop='first'",
                CATEGORICAL_FEATURES,
                NUMERICAL_FEATURES,
                len(NUMERICAL_FEATURES),
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------------
    #  Transformation entry point
    # ------------------------------------------------------------------

    def initiate_data_transformation(
        self, train_path: str, test_path: str
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """Clean data, enrich with weather + irrigation + fertilizer,
        fit preprocessor, transform, and save.
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train (%d rows) and test (%d rows)", len(train_df), len(test_df))

            # Clean both splits
            train_df = self.clean_and_prepare_df(train_df)
            test_df = self.clean_and_prepare_df(test_df)

            # Enrich with NASA POWER weather data
            logging.info("Enriching with NASA POWER weather data...")
            train_df = self.enrich_with_weather(train_df)
            test_df = self.enrich_with_weather(test_df)

            # Enrich with irrigation coverage
            logging.info("Enriching with irrigation coverage data...")
            train_df = self.enrich_with_irrigation(train_df)
            test_df = self.enrich_with_irrigation(test_df)

            # Enrich with NPK fertilizer consumption
            logging.info("Enriching with NPK fertilizer data...")
            train_df = self.enrich_with_fertilizer(train_df)
            test_df = self.enrich_with_fertilizer(test_df)

            # Verify target exists
            if TARGET_COLUMN not in train_df.columns:
                raise KeyError(f"Target '{TARGET_COLUMN}' not found after cleaning.")

            # Safety: verify production was dropped (leakage check)
            if "production" in train_df.columns:
                raise ValueError(
                    "LEAKAGE DETECTED: 'production' column still present after cleaning. "
                    "This column must be dropped because yield = production / area."
                )

            # Split features / target
            input_feature_train = train_df.drop(columns=[TARGET_COLUMN])
            target_train = train_df[TARGET_COLUMN]

            input_feature_test = test_df.drop(columns=[TARGET_COLUMN])
            target_test = test_df[TARGET_COLUMN]

            logging.info(
                "Input features (%d): %s",
                len(input_feature_train.columns),
                input_feature_train.columns.tolist(),
            )

            # Fit on train, transform both
            preprocessor_obj = self.get_data_transformer_object()

            train_arr = preprocessor_obj.fit_transform(input_feature_train)
            test_arr = preprocessor_obj.transform(input_feature_test)

            # Append target as last column
            train_arr = np.c_[train_arr, np.array(target_train)]
            test_arr = np.c_[test_arr, np.array(target_test)]

            # Save preprocessor
            save_object(
                file_path=self.config.preprocessor_obj_file_path,
                obj=preprocessor_obj,
            )
            logging.info("Preprocessor saved -> %s", self.config.preprocessor_obj_file_path)

            logging.info(
                "Transformed: train=%s, test=%s",
                train_arr.shape,
                test_arr.shape,
            )

            return (
                train_arr,
                test_arr,
                self.config.preprocessor_obj_file_path,
            )

        except Exception as e:
            logging.error("Data Transformation failed: %s", e)
            raise CustomException(e, sys)