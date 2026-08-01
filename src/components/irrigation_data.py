"""
src/components/irrigation_data.py

State-level irrigation coverage data for Indian agriculture.

Sources:
    - Agriculture Census 2015-16 (Ministry of Agriculture)
    - NITI Aayog Irrigation Statistics
    - Ministry of Water Resources Annual Reports

Irrigation coverage = (Net irrigated area / Net sown area) * 100

The data is static (updated yearly at most) and stored as a CSV.
No API keys, no network calls, no rate limits.

Usage:
    from src.components.irrigation_data import (
        get_irrigation_for_state,
        enrich_dataframe_with_irrigation,
    )

    pct = get_irrigation_for_state("Punjab")   # 94.0
    df = enrich_dataframe_with_irrigation(df, state_col="state")
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
IRRIGATION_CSV: Path = PROJECT_ROOT / "data" / "raw" / "irrigation_coverage.csv"

IRRIGATION_FEATURE: str = "irrigation_coverage_pct"
DEFAULT_IRRIGATION: float = 40.0  # national approximate fallback

# Map source CSV state names to normalized names for fuzzy matching
_STATE_ALIASES: Dict[str, str] = {
    "odisha": "Odisha",
    "orissa": "Odisha",
    "tamilnadu": "Tamil Nadu",
    "tamil nadu": "Tamil Nadu",
    "jammu & kashmir": "Jammu and Kashmir",
    "jammu and kashmir": "Jammu and Kashmir",
    "uttaranchal": "Uttarakhand",
}


# ---------------------------------------------------------------------------
#  Data loading
# ---------------------------------------------------------------------------

def _load_irrigation_data() -> Dict[str, float]:
    """Load irrigation coverage from CSV into a dict keyed by state name.

    Returns:
        Dict mapping state name to irrigation coverage percentage.
    """
    if not IRRIGATION_CSV.exists():
        logger.warning("Irrigation CSV not found: %s — using defaults", IRRIGATION_CSV)
        return {}

    try:
        df = pd.read_csv(IRRIGATION_CSV)
        # Ensure columns are clean
        df.columns = [c.strip().lower() for c in df.columns]

        if "state" not in df.columns or "irrigation_coverage_pct" not in df.columns:
            logger.warning("Irrigation CSV missing required columns")
            return {}

        # Build lookup dict
        data: Dict[str, float] = {}
        for _, row in df.iterrows():
            state_name = str(row["state"]).strip()
            pct = float(row["irrigation_coverage_pct"])
            data[state_name] = pct

        logger.info("Loaded irrigation data for %d states", len(data))
        return data

    except Exception as exc:
        logger.error("Failed to load irrigation CSV: %s", exc)
        return {}


def _normalize_state_name(state: str) -> str:
    """Normalize state name for fuzzy matching."""
    if not state:
        return ""
    clean = state.strip()
    lower = clean.lower()
    if lower in _STATE_ALIASES:
        return _STATE_ALIASES[lower]
    return clean


# Module-level cache
_irrigation_cache: Optional[Dict[str, float]] = None


def _get_cache() -> Dict[str, float]:
    """Get or initialize the module-level irrigation cache."""
    global _irrigation_cache
    if _irrigation_cache is None:
        _irrigation_cache = _load_irrigation_data()
    return _irrigation_cache


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def get_irrigation_for_state(state: str) -> float:
    """Get irrigation coverage percentage for a state.

    Args:
        state: Indian state name.

    Returns:
        Irrigation coverage percentage (0-100).
        Returns DEFAULT_IRRIGATION if state not found.
    """
    cache = _get_cache()
    if not cache:
        return DEFAULT_IRRIGATION

    normalized = _normalize_state_name(state)

    # Exact match
    if normalized in cache:
        return cache[normalized]

    # Case-insensitive match
    lower = normalized.lower()
    for key, val in cache.items():
        if key.lower() == lower:
            return val

    # Partial match
    for key, val in cache.items():
        if lower in key.lower() or key.lower() in lower:
            return val

    logger.debug("Irrigation data not found for '%s' — using default %.0f%%", state, DEFAULT_IRRIGATION)
    return DEFAULT_IRRIGATION


def get_all_irrigation() -> Dict[str, float]:
    """Get irrigation coverage for all states in the dataset.

    Returns:
        Dict mapping state name to irrigation coverage percentage.
    """
    return dict(_get_cache())


def enrich_dataframe_with_irrigation(
    df: pd.DataFrame,
    state_col: str = "state",
) -> pd.DataFrame:
    """Add irrigation coverage column to a DataFrame by mapping state names.

    Args:
        df: Input DataFrame with a state column.
        state_col: Name of the state column.

    Returns:
        Enriched DataFrame with 1 additional column: irrigation_coverage_pct.
    """
    if state_col not in df.columns:
        logger.warning("State column '%s' not found — filling with default", state_col)
        df[IRRIGATION_FEATURE] = DEFAULT_IRRIGATION
        return df

    unique_states = df[state_col].dropna().unique().tolist()
    logger.info("Mapping irrigation coverage for %d unique states", len(unique_states))

    state_irrigation: Dict[str, float] = {}
    for state_name in unique_states:
        state_irrigation[state_name] = get_irrigation_for_state(str(state_name))

    df[IRRIGATION_FEATURE] = df[state_col].map(state_irrigation).fillna(DEFAULT_IRRIGATION)

    logger.info("Irrigation enrichment complete: added %s", IRRIGATION_FEATURE)
    return df


def get_supported_states() -> List[str]:
    """Return list of all states with irrigation data."""
    return sorted(_get_cache().keys())