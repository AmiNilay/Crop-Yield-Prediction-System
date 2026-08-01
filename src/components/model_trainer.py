import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV

from src.exception.exception import CustomException
from src.logging.logger import logging
from src.utils.common import save_json, save_object

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
R2_THRESHOLD: float = 0.1


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("models", "model.pkl")
    metadata_file_path: str = os.path.join("models", "model_metadata.json")


class ModelTrainer:
    """Train XGBRegressor with GridSearchCV, evaluate, and persist."""

    def __init__(self) -> None:
        self.config = ModelTrainerConfig()

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_metrics(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compute regression metrics."""
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        mask = y_true > 0.01
        mape = (
            float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
            if mask.any()
            else 0.0
        )
        return {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2": round(r2, 4),
            "mape": round(mape, 2),
        }

    # ------------------------------------------------------------------
    def initiate_model_trainer(
        self, train_array: np.ndarray, test_array: np.ndarray
    ) -> float:
        """Run GridSearchCV, evaluate, save model + metadata.

        Returns:
            R² score on the test set.
        """
        try:
            logging.info("Splitting train/test arrays")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # ------------------------------------------------------------------
            #  GridSearchCV
            # ------------------------------------------------------------------
            xgb_model = xgb.XGBRegressor(random_state=RANDOM_SEED)

            param_grid: Dict[str, list] = {
                "n_estimators": [50, 100],
                "learning_rate": [0.1, 0.05],
                "max_depth": [3, 5],
                "subsample": [0.8, 1.0],
            }

            grid_search = GridSearchCV(
                estimator=xgb_model,
                param_grid=param_grid,
                cv=3,
                n_jobs=-1,
                verbose=2,
                scoring="r2",
            )

            logging.info("Starting GridSearchCV (3-fold, seed=%d)", RANDOM_SEED)
            grid_search.fit(X_train, y_train)

            best_model: xgb.XGBRegressor = grid_search.best_estimator_
            best_params: Dict[str, Any] = grid_search.best_params_
            logging.info("Best params: %s", best_params)

            # ------------------------------------------------------------------
            #  Evaluate
            # ------------------------------------------------------------------
            y_pred = best_model.predict(X_test)
            metrics = self._compute_metrics(y_test, y_pred)
            logging.info("Test metrics: %s", metrics)

            if metrics["r2"] < R2_THRESHOLD:
                raise CustomException(
                    Exception(
                        f"R²={metrics['r2']:.4f} below threshold {R2_THRESHOLD}"
                    ),
                    sys,
                )

            # ------------------------------------------------------------------
            #  Save model (sklearn wrapper)
            # ------------------------------------------------------------------
            save_object(
                file_path=self.config.trained_model_file_path,
                obj=best_model,
            )
            # save_object already auto-saves model.json and model.ubj
            logging.info("Model saved -> %s (+ native formats)", self.config.trained_model_file_path)

            # ------------------------------------------------------------------
            #  Save metadata
            # ------------------------------------------------------------------
            metadata: Dict[str, Any] = {
                "model_version": "1.0.0",
                "created_at": datetime.now(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "model_type": "XGBRegressor",
                "framework": f"xgboost {xgb.__version__}",
                "random_seed": RANDOM_SEED,
                "hyperparameters": best_params,
                "training_data": {
                    "n_train": int(X_train.shape[0]),
                    "n_test": int(X_test.shape[0]),
                    "n_features": int(X_train.shape[1]),
                    "split": f"80/20 (seed={RANDOM_SEED})",
                },
                "evaluation": {
                    "test_metrics": metrics,
                    "grid_search_cv": 3,
                    "grid_search_best_score": round(
                        float(grid_search.best_score_), 4
                    ),
                },
                "files": {
                    "model_pickle": "model.pkl",
                    "model_json": "model.json",
                    "model_ubj": "model.ubj",
                    "preprocessor": "preprocessor.pkl",
                },
            }
            save_json(self.config.metadata_file_path, metadata)
            logging.info("Metadata saved -> %s", self.config.metadata_file_path)

            return metrics["r2"]

        except Exception as e:
            logging.error("Model training failed: %s", e)
            raise CustomException(e, sys)