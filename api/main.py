"""
api/main.py

FastAPI serving layer — v3.2 with forecast mode + farmer overrides.

Endpoints:
    GET  /                                  → Redirect to /docs
    GET  /api/v1/health                     → System health check
    GET  /api/v1/model/info                 → Model metadata
    GET  /api/v1/model/metrics              → Evaluation metrics
    GET  /api/v1/crops                      → List supported crops
    GET  /api/v1/states                     → List supported states
    GET  /api/v1/seasons                    → List supported seasons
    GET  /api/v1/weather?state=X&year=Y     → Historical weather (NASA POWER)
    GET  /api/v1/forecast?state=X&season=Y  → Seasonal forecast (Open-Meteo)
    GET  /api/v1/irrigation?state=X         → Irrigation coverage
    GET  /api/v1/fertilizer?state=X         → NPK consumption
    GET  /api/v1/fertilizer/all             → All states NPK
    GET  /api/v1/risk?state=X               → Weather risk for state
    GET  /api/v1/risk/all                   → Weather risk all states
    POST /api/v1/predict?forecast=true|false → Yield prediction
"""

import json
import logging
import os
import sys
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

HAS_XGB: bool = False
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    pass

from src.components.fertilizer_data import (
    FERTILIZER_FEATURE,
    get_all_fertilizer,
    get_fertilizer_for_state,
)
from src.components.irrigation_data import (
    IRRIGATION_FEATURE,
    get_all_irrigation,
    get_irrigation_for_state,
)
from src.components.weather_data import (
    PERIOD_LABEL,
    WEATHER_FEATURES,
    get_weather_for_state,
)
from src.components.forecast_data import (
    FORECAST_FEATURES,
    get_forecast_for_state,
    get_forecast_with_historical_comparison,
)
from src.components.weather_risk import (
    RiskAlert,
    RiskAssessment,
    assess_all_states,
    assess_state_risk,
)
from src.utils.geo_coords import get_coordinates

MODELS_DIR: Path = PROJECT_ROOT / "models"
SCHEMA_PATH: Path = MODELS_DIR / "feature_schema.json"
METADATA_PATH: Path = MODELS_DIR / "model_metadata.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("api")

API_TO_COLUMN: Dict[str, str] = {
    "crop": "crop",
    "state": "state",
    "season": "season",
    "crop_year": "crop_year",
    "area": "area",
}


# ---------------------------------------------------------------------------
#  Pydantic models
# ---------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    crop: str = Field(..., examples=["RICE"])
    state: str = Field(..., examples=["Punjab"])
    season: str = Field(..., examples=["KHARIF"])
    crop_year: int = Field(..., ge=1997, le=2015, examples=[2010])
    area: float = Field(1.0, ge=0.1, le=10000, description="Farm area in hectares")
    irrigation_pct: Optional[float] = Field(
        None, ge=0, le=100,
        description="Irrigation coverage %. None = use state average.",
    )
    npk_kg_per_ha: Optional[float] = Field(
        None, ge=0, le=500,
        description="NPK consumption kg/ha. None = use state average.",
    )


class PredictionResponse(BaseModel):
    predicted_yield: float
    crop: str
    state: str
    season: str
    crop_year: int
    area: float
    weather_source: str
    weather: Optional[Dict[str, float]] = None
    irrigation_pct: float
    irrigation_source: str
    npk_consumption: float
    npk_source: str
    forecast_confidence: Optional[str] = None
    forecast_spread: Optional[Dict[str, float]] = None
    forecast_warning: Optional[str] = None
    historical_yield: Optional[float] = None
    yield_delta: Optional[float] = None
    model_version: str = "3.2.0"
    unit: str = "Quintal/Hectare"


class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool
    preprocessor_loaded: bool
    schema_loaded: bool
    weather_cache_loaded: bool


class WeatherResponse(BaseModel):
    state: str
    latitude: float
    longitude: float
    period: str
    weather: Dict[str, float]
    source: str


class AlertModel(BaseModel):
    level: str
    category: str
    icon: str
    title: str
    message: str
    metric_name: str
    current_value: float
    threshold: float
    norm_value: float
    deviation_pct: float


class RiskResponse(BaseModel):
    state: str
    overall_risk: str
    overall_icon: str
    alerts: List[AlertModel]
    weather: Dict[str, float]
    summary: str


class RiskAllResponse(BaseModel):
    period: str
    states_assessed: int
    assessments: List[RiskResponse]


class IrrigationResponse(BaseModel):
    state: str
    irrigation_coverage_pct: float
    category: str
    description: str


class FertilizerResponse(BaseModel):
    state: str
    npk_consumption_kg_per_ha: float
    category: str
    description: str


class ForecastResponse(BaseModel):
    state: str
    season: str
    weather: Dict[str, float]
    forecast_spread: Dict[str, float]
    forecast_confidence: str
    coverage_pct: float
    forecast_horizon_days: int
    historical_baseline: Optional[Dict[str, float]] = None
    historical_delta: Optional[Dict[str, float]] = None
    historical_delta_pct: Optional[Dict[str, float]] = None
    warning: Optional[str] = None
    source: str
    attribution: str


# ---------------------------------------------------------------------------
#  Application state
# ---------------------------------------------------------------------------

class ModelState:
    def __init__(self) -> None:
        self.booster: Optional[Any] = None
        self.preprocessor: Optional[Any] = None
        self.schema: Optional[Dict[str, Any]] = None
        self.metadata: Optional[Dict[str, Any]] = None
        self.output_feature_names: List[str] = []
        self.feature_columns: List[str] = []
        self.weather_cache_loaded: bool = False

    def is_ready(self) -> bool:
        return self.booster is not None and self.preprocessor is not None


state = ModelState()


def _load_artifacts() -> None:
    if SCHEMA_PATH.exists():
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            state.schema = json.load(f)
        state.feature_columns = state.schema.get("feature_columns", [])
        state.output_feature_names = state.schema.get("output_feature_names", [])
        logger.info("Loaded feature_schema.json (%d features)", len(state.feature_columns))

    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            state.metadata = json.load(f)
        logger.info("Loaded model_metadata.json (v%s)", state.metadata.get("model_version"))

    pp_path = MODELS_DIR / "preprocessor.pkl"
    if pp_path.exists():
        try:
            state.preprocessor = joblib.load(str(pp_path))
            logger.info("Loaded preprocessor.pkl via joblib")
        except Exception:
            try:
                import pickle
                with open(pp_path, "rb") as f:
                    state.preprocessor = pickle.load(f)
                logger.info("Loaded preprocessor.pkl via pickle")
            except Exception as exc:
                logger.error("Cannot load preprocessor.pkl: %s", exc)

    if HAS_XGB:
        for fname in ("model.json", "model.ubj"):
            path = MODELS_DIR / fname
            if path.exists():
                try:
                    booster = xgb.Booster()
                    booster.load_model(str(path))
                    state.booster = booster
                    logger.info("Loaded %s", fname)
                    break
                except Exception:
                    continue

    if state.booster is None:
        pkl_path = MODELS_DIR / "model.pkl"
        if pkl_path.exists():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = joblib.load(str(pkl_path))
                if hasattr(model, "get_booster"):
                    state.booster = model.get_booster()
                    logger.info("Loaded model.pkl (extracted Booster)")
            except Exception as exc:
                logger.error("Cannot load model.pkl: %s", exc)

    from src.components.weather_data import CACHE_PATH
    state.weather_cache_loaded = CACHE_PATH.exists()
    if state.weather_cache_loaded:
        logger.info("Weather cache found: %s", CACHE_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model artifacts...")
    _load_artifacts()
    if state.is_ready():
        logger.info("Model ready.")
    else:
        logger.warning("Model NOT ready.")
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
#  App creation
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Crop Yield Prediction API",
    description=(
        "AI-powered crop yield prediction with historical weather "
        "(NASA POWER) and seasonal forecasts (Open-Meteo).\n\n"
        "**Dataset:** 50,000 district-level records, 15 crops, 33 states\n"
        "**Forecast:** Open-Meteo Seasonal Forecast API (CC-BY 4.0)\n"
        "**Overrides:** Farmers can supply their own irrigation and NPK values"
    ),
    version="3.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
#  Root redirect
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to Swagger UI."""
    return RedirectResponse(url="/docs")


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _assessment_to_response(a: RiskAssessment) -> RiskResponse:
    return RiskResponse(
        state=a.state,
        overall_risk=a.overall_risk,
        overall_icon=a.overall_icon,
        alerts=[
            AlertModel(
                level=al.level,
                category=al.category,
                icon=al.icon,
                title=al.title,
                message=al.message,
                metric_name=al.metric_name,
                current_value=al.current_value,
                threshold=al.threshold,
                norm_value=al.norm_value,
                deviation_pct=al.deviation_pct,
            )
            for al in a.alerts
        ],
        weather=a.weather,
        summary=a.summary,
    )


def _irrigation_category(pct: float) -> tuple:
    if pct >= 80:
        return "Very High", "Predominantly irrigated."
    elif pct >= 60:
        return "High", "Significant irrigation."
    elif pct >= 40:
        return "Moderate", "Partial irrigation."
    elif pct >= 20:
        return "Low", "Mostly rain-fed."
    return "Very Low", "Predominantly rain-fed."


def _fertilizer_category(kg_ha: float) -> tuple:
    if kg_ha >= 200:
        return "Very High", "Intensive chemical inputs."
    elif kg_ha >= 150:
        return "High", "Above-average use."
    elif kg_ha >= 100:
        return "Moderate", "Average consumption."
    elif kg_ha >= 50:
        return "Low", "Below-average use."
    return "Very Low", "Minimal inputs."


# ---------------------------------------------------------------------------
#  System endpoints
# ---------------------------------------------------------------------------

@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy" if state.is_ready() else "degraded",
        model_loaded=state.booster is not None,
        preprocessor_loaded=state.preprocessor is not None,
        schema_loaded=state.schema is not None,
        weather_cache_loaded=state.weather_cache_loaded,
    )


# ---------------------------------------------------------------------------
#  Model endpoints
# ---------------------------------------------------------------------------

@app.get("/api/v1/model/info", tags=["Model"])
async def model_info() -> Dict[str, Any]:
    if state.metadata:
        return state.metadata
    return {"message": "No model_metadata.json found."}


@app.get("/api/v1/model/metrics", tags=["Model"])
async def model_metrics() -> Dict[str, Any]:
    if state.metadata and "evaluation" in state.metadata:
        return state.metadata["evaluation"]
    raise HTTPException(status_code=404, detail="Metrics not available.")


# ---------------------------------------------------------------------------
#  Reference endpoints
# ---------------------------------------------------------------------------

@app.get("/api/v1/crops", tags=["Reference"])
async def list_crops() -> Dict[str, List[str]]:
    if not state.schema:
        raise HTTPException(status_code=503, detail="Schema not loaded.")
    return {"crops": state.schema.get("unique_crop", [])}


@app.get("/api/v1/states", tags=["Reference"])
async def list_states() -> Dict[str, List[str]]:
    if not state.schema:
        raise HTTPException(status_code=503, detail="Schema not loaded.")
    return {"states": state.schema.get("unique_state", [])}


@app.get("/api/v1/seasons", tags=["Reference"])
async def list_seasons() -> Dict[str, List[str]]:
    if not state.schema:
        raise HTTPException(status_code=503, detail="Schema not loaded.")
    return {"seasons": state.schema.get("unique_season", [])}


# ---------------------------------------------------------------------------
#  Weather & Forecast endpoints
# ---------------------------------------------------------------------------

@app.get("/api/v1/weather", response_model=WeatherResponse, tags=["Weather"])
async def get_weather(
    state_name: str = Query(..., alias="state"),
    year: Optional[int] = Query(None),
) -> WeatherResponse:
    """Get historical weather data from NASA POWER."""
    lat, lon = get_coordinates(state_name)
    try:
        if year:
            weather = get_weather_for_state(state_name, year=year)
            period = f"Year {year}"
        else:
            weather = get_weather_for_state(state_name)
            period = PERIOD_LABEL
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return WeatherResponse(
        state=state_name,
        latitude=lat,
        longitude=lon,
        period=period,
        weather=weather,
        source="NASA POWER",
    )


@app.get("/api/v1/forecast", response_model=ForecastResponse, tags=["Forecast"])
async def get_forecast(
    state_name: str = Query(..., alias="state"),
    season: str = Query(
        ..., description="KHARIF, RABI, SUMMER, AUTUMN, WINTER, WHOLE YEAR"
    ),
) -> ForecastResponse:
    """Get seasonal weather forecast with confidence intervals and historical delta.

    Data source: Open-Meteo Seasonal Forecast API (CC-BY 4.0).
    """
    try:
        fc = get_forecast_with_historical_comparison(state_name, season)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Forecast failed: {exc}")

    weather = {k: fc[k] for k in FORECAST_FEATURES if k in fc}

    return ForecastResponse(
        state=state_name,
        season=season.upper(),
        weather=weather,
        forecast_spread=fc.get("forecast_spread", {}),
        forecast_confidence=fc.get("forecast_confidence", "unknown"),
        coverage_pct=fc.get("coverage_pct", 0),
        forecast_horizon_days=fc.get("forecast_horizon_days", 0),
        historical_baseline=fc.get("historical_baseline"),
        historical_delta=fc.get("historical_delta"),
        historical_delta_pct=fc.get("historical_delta_pct"),
        warning=fc.get("warning"),
        source=fc.get("source", "Open-Meteo Seasonal Forecast"),
        attribution="Weather data by Open-Meteo.com",
    )


# ---------------------------------------------------------------------------
#  Agricultural input endpoints
# ---------------------------------------------------------------------------

@app.get("/api/v1/irrigation", response_model=IrrigationResponse, tags=["Inputs"])
async def get_irrigation(
    state_name: str = Query(..., alias="state"),
) -> IrrigationResponse:
    """Get irrigation coverage percentage for a state."""
    pct = get_irrigation_for_state(state_name)
    cat, desc = _irrigation_category(pct)
    return IrrigationResponse(
        state=state_name,
        irrigation_coverage_pct=pct,
        category=cat,
        description=desc,
    )


@app.get("/api/v1/fertilizer", response_model=FertilizerResponse, tags=["Inputs"])
async def get_fertilizer(
    state_name: str = Query(..., alias="state"),
) -> FertilizerResponse:
    """Get NPK fertilizer consumption for a state."""
    kg_ha = get_fertilizer_for_state(state_name)
    cat, desc = _fertilizer_category(kg_ha)
    return FertilizerResponse(
        state=state_name,
        npk_consumption_kg_per_ha=kg_ha,
        category=cat,
        description=desc,
    )


@app.get("/api/v1/fertilizer/all", tags=["Inputs"])
async def get_fertilizer_all() -> Dict[str, Any]:
    """Get NPK fertilizer consumption for all states."""
    data = get_all_fertilizer()
    result = {}
    for sn, kg_ha in sorted(data.items()):
        cat, _ = _fertilizer_category(kg_ha)
        result[sn] = {"npk_consumption_kg_per_ha": kg_ha, "category": cat}
    return {"states": result}


# ---------------------------------------------------------------------------
#  Risk assessment endpoints
# ---------------------------------------------------------------------------

@app.get("/api/v1/risk", response_model=RiskResponse, tags=["Risk Alerts"])
async def get_risk(
    state_name: str = Query(..., alias="state"),
) -> RiskResponse:
    """Get weather risk assessment for a state."""
    try:
        return _assessment_to_response(assess_state_risk(state_name))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/v1/risk/all", response_model=RiskAllResponse, tags=["Risk Alerts"])
async def get_risk_all() -> RiskAllResponse:
    """Get weather risk assessment for all states."""
    try:
        assessments = assess_all_states()
        return RiskAllResponse(
            period=PERIOD_LABEL,
            states_assessed=len(assessments),
            assessments=[_assessment_to_response(a) for a in assessments],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
#  Prediction endpoint
# ---------------------------------------------------------------------------

@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_yield(
    request: PredictionRequest,
    forecast: bool = Query(
        False,
        description="Use Open-Meteo forecast instead of historical weather",
    ),
) -> PredictionResponse:
    """Predict crop yield.

    Set ?forecast=true to use seasonal forecast weather.
    Optionally supply irrigation_pct and npk_kg_per_ha to override
    state averages with farmer-specific values.
    """
    if not state.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded.")

    # Validate inputs against schema
    if state.schema:
        valid_crops = [c.upper() for c in state.schema.get("unique_crop", [])]
        valid_states = state.schema.get("unique_state", [])
        valid_seasons = [s.upper() for s in state.schema.get("unique_season", [])]

        if valid_crops and request.crop.upper() not in valid_crops:
            raise HTTPException(
                status_code=422, detail=f"Unknown crop '{request.crop}'."
            )
        if valid_seasons and request.season.upper() not in valid_seasons:
            raise HTTPException(
                status_code=422, detail=f"Unknown season '{request.season}'."
            )
        if valid_states:
            match = next(
                (s for s in valid_states if s.lower() == request.state.lower()),
                None,
            )
            if match:
                request.state = match
            elif request.state not in valid_states:
                raise HTTPException(
                    status_code=422, detail=f"Unknown state '{request.state}'."
                )

    # Weather routing
    weather_source = "historical"
    forecast_confidence = None
    forecast_spread = None
    forecast_warning = None
    historical_yield = None
    yield_delta = None

    if forecast:
        try:
            fc = get_forecast_for_state(request.state, request.season)
            weather = {k: fc[k] for k in WEATHER_FEATURES if k in fc}
            weather_source = "forecast"
            forecast_confidence = fc.get("forecast_confidence")
            forecast_spread = fc.get("forecast_spread")
            forecast_warning = fc.get("warning")
        except Exception:
            weather = None
    else:
        try:
            weather = get_weather_for_state(
                request.state, year=request.crop_year
            )
        except Exception:
            weather = None

    # Irrigation: user override or state average
    if request.irrigation_pct is not None:
        irrigation_val = request.irrigation_pct
        irrigation_source = "user"
        logger.info(
            "API: Using farmer irrigation override: %.1f%%", irrigation_val
        )
    else:
        irrigation_val = get_irrigation_for_state(request.state)
        irrigation_source = "state-avg"
        logger.info(
            "API: Using state-average irrigation: %.1f%%", irrigation_val
        )

    # NPK: user override or state average
    if request.npk_kg_per_ha is not None:
        npk_val = request.npk_kg_per_ha
        npk_source = "user"
        logger.info("API: Using farmer NPK override: %.1f kg/ha", npk_val)
    else:
        npk_val = get_fertilizer_for_state(request.state)
        npk_source = "state-avg"
        logger.info("API: Using state-average NPK: %.1f kg/ha", npk_val)

    # Build input row
    row: Dict[str, Any] = {}
    for api_key, col_name in API_TO_COLUMN.items():
        row[col_name] = getattr(request, api_key, None)
    if weather:
        for feat in WEATHER_FEATURES:
            row[feat] = weather.get(feat)
    row[IRRIGATION_FEATURE] = irrigation_val
    row[FERTILIZER_FEATURE] = npk_val

    input_df = pd.DataFrame([row], columns=state.feature_columns)

    try:
        X_t = state.preprocessor.transform(input_df)
        if hasattr(X_t, "toarray"):
            X_t = X_t.toarray()
        X_arr = np.asarray(X_t, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        dmat = xgb.DMatrix(
            X_arr, feature_names=state.output_feature_names or None
        )
        prediction = max(0.0, float(state.booster.predict(dmat)[0]))

        # If forecast mode, also compute historical baseline for delta
        if forecast:
            try:
                hist_weather = get_weather_for_state(
                    request.state, year=request.crop_year
                )
                hist_row = dict(row)
                for feat in WEATHER_FEATURES:
                    hist_row[feat] = hist_weather.get(feat)
                hist_df = pd.DataFrame(
                    [hist_row], columns=state.feature_columns
                )
                hist_X = state.preprocessor.transform(hist_df)
                if hasattr(hist_X, "toarray"):
                    hist_X = hist_X.toarray()
                hist_arr = np.asarray(hist_X, dtype=np.float32)
                hist_dmat = xgb.DMatrix(
                    hist_arr,
                    feature_names=state.output_feature_names or None,
                )
                historical_yield = max(
                    0.0, float(state.booster.predict(hist_dmat)[0])
                )
                yield_delta = round(prediction - historical_yield, 4)
            except Exception:
                pass

        version = "3.2.0"
        if state.metadata:
            version = state.metadata.get("model_version", version)

        return PredictionResponse(
            predicted_yield=round(prediction, 4),
            crop=request.crop,
            state=request.state,
            season=request.season,
            crop_year=request.crop_year,
            area=request.area,
            weather_source=weather_source,
            weather=weather,
            irrigation_pct=round(irrigation_val, 1),
            irrigation_source=irrigation_source,
            npk_consumption=round(npk_val, 1),
            npk_source=npk_source,
            forecast_confidence=forecast_confidence,
            forecast_spread=forecast_spread,
            forecast_warning=forecast_warning,
            historical_yield=(
                round(historical_yield, 4) if historical_yield else None
            ),
            yield_delta=yield_delta,
            model_version=version,
        )
    except Exception as exc:
        logger.error("Prediction failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {exc}"
        )