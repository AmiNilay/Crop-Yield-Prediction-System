"""
src/components/forecast_data.py

Open-Meteo Seasonal Forecast API for crop yield prediction.

Provides seasonal weather forecasts compatible with the NASA POWER
weather_data.py schema. Supports all Indian cropping seasons.

Data Source : Open-Meteo Seasonal Forecast API (ECMWF SEAS5 / EC46)
License     : CC-BY 4.0 — attribution required
Endpoint    : https://seasonal-api.open-meteo.com/v1/seasonal
Free tier   : 10,000 calls/day (non-commercial)
Resolution  : 36 km, 6-hourly, 51 ensemble members

Attribution (required):
    "Weather data by Open-Meteo.com"

Usage:
    from src.components.forecast_data import (
        fetch_seasonal_forecast,
        get_forecast_for_state,
        get_forecast_with_historical_comparison,
        FORECAST_FEATURES,
    )

    # Direct lat/lon
    wx = fetch_seasonal_forecast(30.7, 76.8, "KHARIF")

    # State-based (cached)
    wx = get_forecast_for_state("Punjab", "RABI")

    # With historical delta
    wx = get_forecast_with_historical_comparison("Punjab", "RABI")
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

from src.utils.geo_coords import get_coordinates

logger = logging.getLogger(__name__)

# ===========================================================================
#  Constants
# ===========================================================================

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
CACHE_DIR: Path = PROJECT_ROOT / "data" / "processed"
CACHE_PATH: Path = CACHE_DIR / "forecast_cache.json"

FORECAST_API_URL: str = "https://seasonal-api.open-meteo.com/v1/seasonal"
FORECAST_DAYS: int = 180
REQUEST_TIMEOUT: int = 60
RATE_LIMIT_SLEEP: float = 1.0

# Cache TTL: 30 days
CACHE_TTL_SECONDS: int = 30 * 24 * 3600

# Unit conversion: Open-Meteo solar radiation W/m² → NASA POWER MJ/m²/day
WM2_TO_MJM2DAY: float = 0.0864

# Schema-compatible feature names (must match weather_data.py)
FORECAST_FEATURES: List[str] = [
    "mean_temperature",
    "total_precipitation",
    "mean_relative_humidity",
    "mean_solar_radiation",
]

# Season → month numbers
SEASON_MONTHS: Dict[str, List[int]] = {
    "KHARIF":     [6, 7, 8, 9, 10],
    "RABI":       [11, 12, 1, 2, 3],
    "SUMMER":     [3, 4, 5],
    "AUTUMN":     [10, 11],
    "WINTER":     [12, 1, 2],
    "WHOLE YEAR": list(range(1, 13)),
}

# Forecast confidence thresholds (days from today to mid-season)
HORIZON_HIGH_DAYS: int = 45
HORIZON_MEDIUM_DAYS: int = 120

# Open-Meteo variable names → internal names
_API_VAR_MAP: Dict[str, str] = {
    "temperature_2m": "temperature_2m",
    "precipitation": "precipitation",
    "relative_humidity_2m": "relative_humidity_2m",
    "shortwave_radiation": "shortwave_radiation",
}


# ===========================================================================
#  Cache Management
# ===========================================================================

def _load_cache() -> Dict[str, Any]:
    """Load forecast cache from disk."""
    if not CACHE_PATH.exists():
        return {}
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            cache = json.load(f)
        logger.info("Loaded forecast cache: %d entries", len(cache))
        return cache
    except Exception as exc:
        logger.warning("Failed to load forecast cache: %s", exc)
        return {}


def _save_cache(cache: Dict[str, Any]) -> None:
    """Persist forecast cache to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, default=str)
        logger.info("Saved forecast cache: %d entries", len(cache))
    except Exception as exc:
        logger.error("Failed to save forecast cache: %s", exc)


def _is_cache_valid(entry: Dict[str, Any]) -> bool:
    """Check if a cache entry is within TTL."""
    cached_at = entry.get("cached_at")
    if not cached_at:
        return False
    try:
        cached_dt = datetime.fromisoformat(str(cached_at))
        age_seconds = (datetime.now() - cached_dt).total_seconds()
        return age_seconds < CACHE_TTL_SECONDS
    except Exception:
        return False


# Module-level cache
_forecast_cache: Optional[Dict[str, Any]] = None


def _get_cache() -> Dict[str, Any]:
    """Get or initialize module-level forecast cache."""
    global _forecast_cache
    if _forecast_cache is None:
        _forecast_cache = _load_cache()
    return _forecast_cache


# ===========================================================================
#  Season Logic
# ===========================================================================

def _normalize_season(season: str) -> str:
    """Normalize season name to canonical uppercase form."""
    s = season.strip().upper()
    aliases = {
        "WHOLE_YEAR": "WHOLE YEAR",
        "WHOLEYEAR": "WHOLE YEAR",
        "ANNUAL": "WHOLE YEAR",
    }
    return aliases.get(s, s)


def _get_season_months(season: str) -> List[int]:
    """Get month numbers for a season."""
    season = _normalize_season(season)
    return SEASON_MONTHS.get(season, list(range(1, 13)))


def _compute_confidence(horizon_days: int) -> str:
    """Estimate forecast confidence from horizon."""
    if horizon_days <= HORIZON_HIGH_DAYS:
        return "high"
    elif horizon_days <= HORIZON_MEDIUM_DAYS:
        return "medium"
    return "low"


def _compute_horizon(
    timestamps: List[str],
    season_months: List[int],
) -> int:
    """Compute median forecast horizon (days from today to mid-season dates)."""
    today = datetime.now().date()
    relevant: List[int] = []

    for ts in timestamps:
        try:
            ts_clean = ts.replace("Z", "").split("+")[0]
            dt = datetime.fromisoformat(ts_clean)
            if dt.month in season_months:
                relevant.append((dt.date() - today).days)
        except Exception:
            continue

    if not relevant:
        return FORECAST_DAYS

    return max(0, int(np.median(relevant)))


# ===========================================================================
#  Open-Meteo API
# ===========================================================================

def _build_api_params(lat: float, lon: float) -> Dict[str, str]:
    """Build Open-Meteo API query parameters."""
    variables = ",".join(_API_VAR_MAP.keys())
    return {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "six_hourly": variables,
        "forecast_days": str(FORECAST_DAYS),
        "timezone": "Asia/Kolkata",
    }


def _fetch_raw_forecast(lat: float, lon: float) -> Dict[str, Any]:
    """Fetch raw forecast data from Open-Meteo.

    Tries six_hourly parameter first; falls back to hourly.
    Raises requests.RequestException on network/API failure.
    """
    params = _build_api_params(lat, lon)

    logger.info(
        "Open-Meteo fetch: (%.4f, %.4f), %d days, vars=%s",
        lat, lon, FORECAST_DAYS, params.get("six_hourly", ""),
    )

    response = requests.get(
        FORECAST_API_URL, params=params, timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()

    # Fallback: if six_hourly not in response, retry with hourly
    if "six_hourly" not in data and "hourly" not in data:
        logger.info("six_hourly not in response — retrying with hourly parameter")
        params["hourly"] = params.pop("six_hourly")
        response = requests.get(
            FORECAST_API_URL, params=params, timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

    return data


def _extract_series(data: Dict[str, Any]) -> Dict[str, List]:
    """Extract time-series arrays from Open-Meteo response.

    Handles both 'six_hourly' and 'hourly' response keys.
    Fills missing variables with empty lists.
    """
    for key in ("six_hourly", "hourly"):
        if key in data:
            raw = data[key]
            return {
                "time": raw.get("time", []),
                "temperature_2m": raw.get("temperature_2m", []),
                "precipitation": raw.get("precipitation", []),
                "relative_humidity_2m": raw.get("relative_humidity_2m", []),
                "shortwave_radiation": raw.get("shortwave_radiation", []),
            }

    raise ValueError(
        "Open-Meteo response missing both 'six_hourly' and 'hourly' keys. "
        f"Response keys: {list(data.keys())}"
    )


# ===========================================================================
#  Aggregation: 6-hourly → Seasonal Statistics
# ===========================================================================

def _safe_float_list(arr: List, indices: List[int]) -> np.ndarray:
    """Extract valid floats at given indices, filtering None/NaN."""
    values = []
    for i in indices:
        if i < len(arr) and arr[i] is not None:
            try:
                v = float(arr[i])
                if not np.isnan(v):
                    values.append(v)
            except (ValueError, TypeError):
                continue
    return np.array(values) if values else np.array([np.nan])


def _aggregate_to_season(
    series: Dict[str, List],
    season: str,
) -> Dict[str, Any]:
    """Aggregate 6-hourly forecast to seasonal features.

    Precipitation: annualized (mean daily × 365.25) to match NASA POWER.
    Temperature:   mean °C
    Humidity:      mean %
    Radiation:     mean W/m² × 0.0864 → MJ/m²/day

    Returns dict with 4 features + forecast metadata.
    """
    timestamps = series["time"]
    season_months = _get_season_months(season)

    if not timestamps:
        raise ValueError("Empty timestamp array from Open-Meteo")

    # Filter indices to season months
    season_indices: List[int] = []
    filtered_ts: List[str] = []

    for i, ts in enumerate(timestamps):
        try:
            ts_clean = ts.replace("Z", "").split("+")[0]
            dt = datetime.fromisoformat(ts_clean)
            if dt.month in season_months:
                season_indices.append(i)
                filtered_ts.append(ts)
        except Exception:
            continue

    n_points = len(season_indices)
    if n_points == 0:
        raise ValueError(
            f"No forecast data for season '{season}' (months {season_months}). "
            "Season may be entirely beyond the {FORECAST_DAYS}-day forecast horizon."
        )

    # Extract filtered values
    temp = _safe_float_list(series["temperature_2m"], season_indices)
    precip = _safe_float_list(series["precipitation"], season_indices)
    humidity = _safe_float_list(series["relative_humidity_2m"], season_indices)
    radiation = _safe_float_list(series["shortwave_radiation"], season_indices)

    # ── Compute features ──

    # Temperature: mean (°C)
    mean_temp = float(np.nanmean(temp))

    # Precipitation: annualize to match NASA POWER format
    # Each 6h value = mm per 6h. Sum = total mm for the period.
    # Annualize: (total / n_days) × 365.25
    n_days = n_points / 4.0  # 4 readings per day
    season_total_mm = float(np.nansum(precip))
    if n_days > 0:
        mean_daily_mm = season_total_mm / n_days
        total_precip = mean_daily_mm * 365.25
    else:
        total_precip = 0.0

    # Humidity: mean (%)
    mean_humidity = float(np.nanmean(humidity))

    # Radiation: mean W/m² → MJ/m²/day
    mean_radiation_wm2 = float(np.nanmean(radiation))
    mean_radiation = mean_radiation_wm2 * WM2_TO_MJM2DAY

    # ── Spread (temporal std dev) ──
    spread = {
        "mean_temperature": round(float(np.nanstd(temp)), 2),
        "total_precipitation": round(float(np.nanstd(precip) * 365.25 / max(n_days, 1) * np.sqrt(4)), 1),
        "mean_relative_humidity": round(float(np.nanstd(humidity)), 1),
        "mean_solar_radiation": round(float(np.nanstd(radiation) * WM2_TO_MJM2DAY), 2),
    }

    # ── Confidence & coverage ──
    total_possible = len(timestamps)
    coverage_pct = (n_points / total_possible * 100) if total_possible > 0 else 0.0
    horizon_days = _compute_horizon(filtered_ts, season_months)
    confidence = _compute_confidence(horizon_days)

    # ── Warning ──
    warning = None
    if coverage_pct < 30:
        warning = (
            f"Low forecast coverage ({coverage_pct:.0f}%). "
            f"Season extends beyond the {FORECAST_DAYS}-day forecast horizon."
        )
    elif horizon_days > HORIZON_MEDIUM_DAYS:
        warning = (
            f"Forecast horizon is {horizon_days} days — "
            f"accuracy drops significantly beyond {HORIZON_MEDIUM_DAYS} days."
        )
    elif horizon_days > HORIZON_HIGH_DAYS:
        warning = (
            f"Forecast horizon is {horizon_days} days — "
            f"treat as indicative trend, not precise prediction."
        )

    return {
        # Core features (drop-in compatible with weather_data.py)
        "mean_temperature": round(mean_temp, 2),
        "total_precipitation": round(total_precip, 1),
        "mean_relative_humidity": round(mean_humidity, 1),
        "mean_solar_radiation": round(mean_radiation, 2),
        # Forecast metadata
        "forecast_spread": spread,
        "forecast_confidence": confidence,
        "coverage_pct": round(coverage_pct, 1),
        "forecast_horizon_days": horizon_days,
        "data_points": n_points,
        "season_total_precipitation_mm": round(season_total_mm, 1),
        "warning": warning,
        "source": "Open-Meteo Seasonal Forecast (ECMWF SEAS5)",
        "attribution": "Weather data by Open-Meteo.com",
    }


# ===========================================================================
#  Public API
# ===========================================================================

def fetch_seasonal_forecast(
    lat: float,
    lon: float,
    season: str,
    months_ahead: int = 6,
) -> Dict[str, Any]:
    """Fetch seasonal weather forecast for coordinates and season.

    Direct API call — no caching. Use get_forecast_for_state() for
    cached state-based lookups.

    Args:
        lat: Latitude
        lon: Longitude
        season: KHARIF, RABI, SUMMER, AUTUMN, WINTER, WHOLE YEAR
        months_ahead: Forecast horizon (default 6)

    Returns:
        Dict with 4 weather features (same keys as weather_data.py)
        plus forecast_spread, forecast_confidence, coverage_pct,
        forecast_horizon_days, warning, source.
    """
    season = _normalize_season(season)

    if season not in SEASON_MONTHS:
        raise ValueError(
            f"Unknown season '{season}'. Valid: {list(SEASON_MONTHS.keys())}"
        )

    raw = _fetch_raw_forecast(lat, lon)
    series = _extract_series(raw)
    result = _aggregate_to_season(series, season)

    result["latitude"] = lat
    result["longitude"] = lon
    result["season"] = season
    result["cached_at"] = datetime.now().isoformat()

    return result


def get_forecast_for_state(
    state: str,
    season: str,
    months_ahead: int = 6,
) -> Dict[str, Any]:
    """Get seasonal forecast for an Indian state (cached).

    Cache key: {state}_{season}_{YYYY-MM} — invalidates monthly.
    TTL: 30 days.

    Args:
        state: Indian state name
        season: Season name
        months_ahead: Forecast horizon (default 6)

    Returns:
        Dict with weather features + forecast metadata.
    """
    season = _normalize_season(season)
    cache = _get_cache()

    # Monthly cache key
    year_month = datetime.now().strftime("%Y-%m")
    cache_key = f"{state}_{season}_{year_month}"

    # Check cache
    if cache_key in cache and _is_cache_valid(cache[cache_key]):
        logger.debug("Forecast cache hit: %s", cache_key)
        entry = cache[cache_key]
        entry["cache_hit"] = True
        return entry

    # Fetch fresh
    lat, lon = get_coordinates(state)
    result = fetch_seasonal_forecast(lat, lon, season, months_ahead)
    result["state"] = state
    result["cache_key"] = cache_key
    result["cache_hit"] = False

    # Store in cache
    cache[cache_key] = result
    _save_cache(cache)

    logger.info(
        "Forecast cached: %s → %s (%s, %.0f%% coverage)",
        cache_key,
        result["forecast_confidence"],
        season,
        result["coverage_pct"],
    )

    return result


def get_forecast_with_historical_comparison(
    state: str,
    season: str,
) -> Dict[str, Any]:
    """Get forecast + delta vs NASA POWER historical baseline.

    Returns forecast values with historical_delta dict showing
    the difference for each feature.

    Returns:
        Forecast dict augmented with:
          - historical_baseline: dict of historical weather values
          - historical_delta: dict of (forecast - historical) per feature
          - historical_delta_pct: dict of percentage differences
    """
    from src.components.weather_data import get_weather_for_state

    forecast = get_forecast_for_state(state, season)

    # Get historical baseline (trailing 365 days)
    try:
        historical = get_weather_for_state(state)
    except Exception as exc:
        logger.warning("Historical baseline unavailable for %s: %s", state, exc)
        forecast["historical_baseline"] = None
        forecast["historical_delta"] = None
        return forecast

    # Compute deltas
    delta: Dict[str, float] = {}
    delta_pct: Dict[str, float] = {}

    for feat in FORECAST_FEATURES:
        fc_val = forecast.get(feat, 0.0) or 0.0
        hist_val = historical.get(feat, 0.0) or 0.0

        diff = fc_val - hist_val
        delta[feat] = round(diff, 2)

        if abs(hist_val) > 0.01:
            delta_pct[feat] = round(diff / abs(hist_val) * 100, 1)
        else:
            delta_pct[feat] = 0.0

    forecast["historical_baseline"] = historical
    forecast["historical_delta"] = delta
    forecast["historical_delta_pct"] = delta_pct

    return forecast


# ===========================================================================
#  Utility
# ===========================================================================

def get_cache_stats() -> Dict[str, Any]:
    """Return forecast cache statistics."""
    cache = _get_cache()
    valid = sum(1 for v in cache.values() if _is_cache_valid(v))
    return {
        "total_entries": len(cache),
        "valid_entries": valid,
        "expired_entries": len(cache) - valid,
        "cache_file": str(CACHE_PATH),
        "cache_size_kb": round(CACHE_PATH.stat().st_size / 1024, 1) if CACHE_PATH.exists() else 0,
        "ttl_days": CACHE_TTL_SECONDS // (24 * 3600),
    }


def clear_cache() -> None:
    """Clear in-memory forecast cache."""
    global _forecast_cache
    _forecast_cache = None
    logger.info("Forecast cache cleared from memory")