"""
src/components/weather_data.py

NASA POWER weather data fetcher with year-aware caching.

Supports two modes:
  1. Year-specific mode (training): cache key = "{state}_{year}"
     Used when crop_year column exists in the DataFrame.
     Pre-computes weather for all unique (state, year) combos in batch.
     ~30 states × 18 years = ~540 API calls, NOT 50K.

  2. Trailing-365 mode (dashboard/API): cache key = "{state}_LATEST"
     Used when crop_year is not present (single-row predictions).

Cache persistence: data/processed/weather_cache.json

Usage:
    from src.components.weather_data import (
        enrich_dataframe_with_weather,
        get_weather_for_state,
        WEATHER_FEATURES,
        PERIOD_LABEL,
    )

    # Training (batch, year-aware)
    df = enrich_dataframe_with_weather(df, state_col="state")

    # Dashboard (single state, trailing 365 days)
    weather = get_weather_for_state("Punjab")
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import requests

from src.utils.geo_coords import get_coordinates

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
CACHE_DIR: Path = PROJECT_ROOT / "data" / "processed"
CACHE_PATH: Path = CACHE_DIR / "weather_cache.json"

# NASA POWER API
POWER_BASE: str = "https://power.larc.nasa.gov/api/temporal/daily/point"
POWER_PARAMS: Dict[str, str] = {
    "T2M": "mean_temperature",
    "PRECTOTCORR": "total_precipitation",
    "RH2M": "mean_relative_humidity",
    "ALLSKY_SFC_SW_DWN": "mean_solar_radiation",
}
POWER_COMMUNITY: str = "AG"

WEATHER_FEATURES: List[str] = list(POWER_PARAMS.values())

# Trailing period for dashboard/API mode
_TRAILING_DAYS: int = 365
_TODAY: datetime = datetime.now()
_TRAILING_START: datetime = _TODAY - timedelta(days=_TRAILING_DAYS)
PERIOD_LABEL: str = f"{_TRAILING_START.strftime('%Y-%m-%d')} to {_TODAY.strftime('%Y-%m-%d')}"

# API request settings
_REQUEST_TIMEOUT: int = 60
_RATE_LIMIT_SLEEP: float = 1.0  # seconds between API calls (respect NASA rate limits)

# Normalization constants (daily -> annual/mean)
_DAILY_TO_MM_YEAR: float = 365.25  # mm/day -> mm/year


# ---------------------------------------------------------------------------
#  Cache management
# ---------------------------------------------------------------------------

def _load_cache() -> Dict[str, Dict[str, float]]:
    """Load weather cache from disk."""
    if not CACHE_PATH.exists():
        logger.info("No weather cache found at %s — starting fresh", CACHE_PATH)
        return {}

    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            cache = json.load(f)
        logger.info("Loaded weather cache: %d entries from %s", len(cache), CACHE_PATH)
        return cache
    except Exception as exc:
        logger.warning("Failed to load weather cache: %s — starting fresh", exc)
        return {}


def _save_cache(cache: Dict[str, Dict[str, float]]) -> None:
    """Persist weather cache to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, default=str)
        logger.info("Saved weather cache: %d entries to %s", len(cache), CACHE_PATH)
    except Exception as exc:
        logger.error("Failed to save weather cache: %s", exc)


# Module-level cache
_weather_cache: Optional[Dict[str, Dict[str, float]]] = None


def _get_cache() -> Dict[str, Dict[str, float]]:
    """Get or initialize the module-level weather cache."""
    global _weather_cache
    if _weather_cache is None:
        _weather_cache = _load_cache()
    return _weather_cache


def clear_cache() -> None:
    """Clear the in-memory cache (useful for testing)."""
    global _weather_cache
    _weather_cache = None


# ---------------------------------------------------------------------------
#  NASA POWER API fetcher
# ---------------------------------------------------------------------------

def _fetch_nasa_power(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
) -> Dict[str, float]:
    """Fetch daily weather data from NASA POWER and aggregate.

    Args:
        lat: Latitude
        lon: Longitude
        start_date: YYYYMMDD format
        end_date: YYYYMMDD format

    Returns:
        Dict with 4 weather features:
          - mean_temperature (°C)
          - total_precipitation (mm/year)
          - mean_relative_humidity (%)
          - mean_solar_radiation (MJ/m²/day)
    """
    params_str = ",".join(POWER_PARAMS.keys())

    url = (
        f"{POWER_BASE}"
        f"?parameters={params_str}"
        f"&community={POWER_COMMUNITY}"
        f"&longitude={lon:.4f}&latitude={lat:.4f}"
        f"&start={start_date}&end={end_date}"
        f"&format=JSON"
    )

    response = requests.get(url, timeout=_REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()

    properties = data.get("properties", {}).get("parameter", {})

    # Extract daily values for each parameter
    result: Dict[str, float] = {}
    for nasa_param, feature_name in POWER_PARAMS.items():
        daily_data = properties.get(nasa_param, {})

        if not daily_data:
            logger.warning("No data for parameter %s", nasa_param)
            result[feature_name] = np.nan
            continue

        # Filter out fill values (-999, -999.0, etc.)
        values = []
        for date_key, val in daily_data.items():
            if val is not None and val > -990:
                values.append(float(val))

        if not values:
            result[feature_name] = np.nan
            continue

        if nasa_param == "PRECTOTCORR":
            # Daily precipitation (mm/day) -> annual total (mm/year)
            result[feature_name] = float(np.mean(values)) * _DAILY_TO_MM_YEAR
        else:
            result[feature_name] = float(np.mean(values))

    return result


def _fetch_and_cache(
    cache_key: str,
    state: str,
    start_date: str,
    end_date: str,
    cache: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """Fetch weather for a (state, period) combo and cache it.

    Args:
        cache_key: Cache key (e.g., "Punjab_2005" or "Punjab_LATEST")
        state: Indian state name
        start_date: YYYYMMDD
        end_date: YYYYMMDD
        cache: The mutable cache dict

    Returns:
        Weather feature dict
    """
    lat, lon = get_coordinates(state)

    logger.info(
        "Fetching NASA POWER for %s (%.2f, %.2f) [%s -> %s]",
        state, lat, lon, start_date, end_date,
    )

    features = _fetch_nasa_power(lat, lon, start_date, end_date)

    # Store in cache
    cache[cache_key] = {
        **features,
        "_lat": lat,
        "_lon": lon,
        "_start": start_date,
        "_end": end_date,
        "_cached_at": datetime.now().isoformat(),
    }

    return features


def _extract_features(cache_entry: Dict[str, Any]) -> Dict[str, float]:
    """Extract just the 4 weather features from a cache entry."""
    return {k: cache_entry[k] for k in WEATHER_FEATURES if k in cache_entry}


# ---------------------------------------------------------------------------
#  Public API — single state fetch
# ---------------------------------------------------------------------------

def get_weather_for_state(
    state: str,
    year: Optional[int] = None,
    force_refresh: bool = False,
) -> Dict[str, float]:
    """Get weather data for a single state.

    Args:
        state: Indian state name
        year: Specific year (e.g., 2005). If None, uses trailing 365 days.
        force_refresh: If True, bypass cache and fetch fresh data.

    Returns:
        Dict with 4 weather features.
    """
    cache = _get_cache()

    if year is not None:
        cache_key = f"{state}_{year}"
        start_date = f"{year}0101"
        end_date = f"{year}1231"
    else:
        cache_key = f"{state}_LATEST"
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=_TRAILING_DAYS)
        start_date = start_dt.strftime("%Y%m%d")
        end_date = end_dt.strftime("%Y%m%d")

    # Check cache
    if not force_refresh and cache_key in cache:
        logger.debug("Weather cache hit: %s", cache_key)
        return _extract_features(cache[cache_key])

    # Fetch fresh
    features = _fetch_and_cache(cache_key, state, start_date, end_date, cache)

    # Persist after each fetch (so partial progress is saved)
    _save_cache(cache)

    return features


# ---------------------------------------------------------------------------
#  Public API — batch enrichment for DataFrames
# ---------------------------------------------------------------------------

def enrich_dataframe_with_weather(
    df: pd.DataFrame,
    state_col: str = "state",
) -> pd.DataFrame:
    """Add weather features to a DataFrame by looking up state (+ year).

    This function is optimized for batch training:
      - Detects if crop_year column exists
      - Pre-computes weather for all unique (state, year) combos
      - Maps features to each row via merge

    Args:
        df: Input DataFrame with at least a state column.
            If crop_year column exists, uses year-specific weather.
            Otherwise, uses trailing-365-day weather.
        state_col: Name of the state column.

    Returns:
        Enriched DataFrame with 4 additional weather feature columns.
    """
    cache = _get_cache()

    # Determine mode
    has_year = "crop_year" in df.columns

    if has_year:
        return _enrich_year_specific(df, state_col, cache)
    else:
        return _enrich_trailing(df, state_col, cache)


def _enrich_year_specific(
    df: pd.DataFrame,
    state_col: str,
    cache: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """Year-specific weather enrichment (training mode).

    Pre-fetches weather for all unique (state, crop_year) combos,
    then merges features onto the DataFrame.
    """
    # Find unique (state, year) combos
    df["_state_year_key"] = (
        df[state_col].astype(str).str.strip()
        + "_"
        + df["crop_year"].astype(int).astype(str)
    )
    unique_keys: Set[str] = set(df["_state_year_key"].unique())

    # Identify which keys need fetching
    to_fetch: List[Tuple[str, str, int]] = []
    for key in sorted(unique_keys):
        if key not in cache:
            parts = key.rsplit("_", 1)
            state_name = parts[0]
            year = int(parts[1])
            to_fetch.append((key, state_name, year))

    logger.info(
        "Weather enrichment: %d unique (state, year) combos, "
        "%d already cached, %d to fetch",
        len(unique_keys), len(unique_keys) - len(to_fetch), len(to_fetch,
        ),
    )

    # Batch fetch with rate limiting
    fetched_count = 0
    for cache_key, state_name, year in to_fetch:
        start_date = f"{year}0101"
        end_date = f"{year}1231"

        try:
            _fetch_and_cache(cache_key, state_name, start_date, end_date, cache)
            fetched_count += 1

            # Rate limit: sleep between API calls
            if fetched_count % 10 == 0:
                logger.info("  Fetched %d / %d weather combos...", fetched_count, len(to_fetch))
                _save_cache(cache)  # checkpoint

            time.sleep(_RATE_LIMIT_SLEEP)

        except Exception as exc:
            logger.error(
                "Weather fetch failed for %s (%d): %s — filling with NaN",
                cache_key, year, exc,
            )
            cache[cache_key] = {feat: np.nan for feat in WEATHER_FEATURES}

    # Save cache after all fetches
    if fetched_count > 0:
        _save_cache(cache)
        logger.info("Weather cache updated: %d new entries saved", fetched_count)

    # Build lookup DataFrame from cache
    cache_records = []
    for key in unique_keys:
        if key in cache:
            record = {"_state_year_key": key}
            for feat in WEATHER_FEATURES:
                record[feat] = cache[key].get(feat, np.nan)
            cache_records.append(record)

    if not cache_records:
        logger.warning("No weather cache entries found — filling all with NaN")
        for feat in WEATHER_FEATURES:
            df[feat] = np.nan
    else:
        cache_df = pd.DataFrame(cache_records)
        df = df.merge(cache_df, on="_state_year_key", how="left")
        logger.info("Weather merge complete: %d rows enriched", len(df))

    # Clean up temp column
    df.drop(columns=["_state_year_key"], inplace=True)

    # Verify all weather features are present
    for feat in WEATHER_FEATURES:
        if feat not in df.columns:
            df[feat] = np.nan
            logger.warning("Weather feature '%s' missing after merge — filled with NaN", feat)

    nans = df[WEATHER_FEATURES].isna().any(axis=1).sum()
    if nans > 0:
        logger.warning(
            "%d / %d rows have NaN weather features (states outside geo_coords coverage)",
            nans, len(df),
        )

    return df


def _enrich_trailing(
    df: pd.DataFrame,
    state_col: str,
    cache: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """Trailing-365-day weather enrichment (dashboard/API fallback).

    Used when crop_year column is not present.
    Fetches current trailing-year weather per state.
    """
    unique_states = df[state_col].dropna().unique().tolist()
    logger.info(
        "Trailing-365 weather enrichment for %d unique states", len(unique_states),
    )

    state_weather: Dict[str, Dict[str, float]] = {}
    for state_name in unique_states:
        state_name_clean = str(state_name).strip()
        try:
            state_weather[state_name_clean] = get_weather_for_state(state_name_clean)
        except Exception as exc:
            logger.error("Weather fetch failed for %s: %s", state_name_clean, exc)
            state_weather[state_name_clean] = {feat: np.nan for feat in WEATHER_FEATURES}

    for feat in WEATHER_FEATURES:
        df[feat] = df[state_col].astype(str).str.strip().map(
            lambda s, f=feat: state_weather.get(s, {}).get(f, np.nan)
        )

    return df


# ---------------------------------------------------------------------------
#  Cache management utilities
# ---------------------------------------------------------------------------

def get_cache_stats() -> Dict[str, Any]:
    """Return statistics about the weather cache."""
    cache = _get_cache()

    # Separate year-specific vs LATEST entries
    year_entries = {k: v for k, v in cache.items() if not k.endswith("_LATEST")} if cache else {}
    latest_entries = {k: v for k, v in cache.items() if k.endswith("_LATEST")} if cache else {}

    states_in_cache = set()
    years_in_cache = set()
    for key in cache:
        if key.endswith("_LATEST"):
            states_in_cache.add(key.replace("_LATEST", ""))
        else:
            parts = key.rsplit("_", 1)
            if len(parts) == 2:
                states_in_cache.add(parts[0])
                try:
                    years_in_cache.add(int(parts[1]))
                except ValueError:
                    pass

    return {
        "total_entries": len(cache),
        "year_specific": len(cache) - len(latest_entries),
        "latest": len(latest_entries),
        "states": sorted(states_in_cache),
        "years": sorted(years_in_cache),
        "cache_file": str(CACHE_PATH),
        "cache_size_kb": CACHE_PATH.stat().st_size / 1024 if CACHE_PATH.exists() else 0,
    }


def preload_all_state_year_combos(
    df: pd.DataFrame,
    state_col: str = "state",
) -> None:
    """Pre-compute weather for all unique (state, year) combos in a DataFrame.

    Call this before training to warm the cache and avoid repeated fetches.

    Args:
        df: DataFrame with state and crop_year columns.
        state_col: Name of the state column.
    """
    if "crop_year" not in df.columns:
        logger.warning("No crop_year column — cannot preload year-specific weather")
        return

    cache = _get_cache()

    combos = (
        df[[state_col, "crop_year"]]
        .drop_duplicates()
        .sort_values([state_col, "crop_year"])
    )

    logger.info(
        "Preloading weather for %d (state, year) combos...",
        len(combos),
    )

    fetched = 0
    skipped = 0
    failed = 0

    for _, row in combos.iterrows():
        state_name = str(row[state_col]).strip()
        year = int(row["crop_year"])
        cache_key = f"{state_name}_{year}"

        if cache_key in cache:
            skipped += 1
            continue

        try:
            _fetch_and_cache(cache_key, state_name, f"{year}0101", f"{year}1231", cache)
            fetched += 1
            time.sleep(_RATE_LIMIT_SLEEP)

            if fetched % 20 == 0:
                logger.info("  Preloaded %d / %d (skipped %d cached)", fetched, len(combos), skipped)
                _save_cache(cache)

        except Exception as exc:
            failed += 1
            logger.error("Preload failed for %s (%d): %s", state_name, year, exc)

    _save_cache(cache)
    logger.info(
        "Preload complete: fetched=%d, cached=%d, failed=%d",
        fetched, skipped, failed,
    )