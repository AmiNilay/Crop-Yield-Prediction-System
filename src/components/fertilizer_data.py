"""
src/components/fertilizer_data.py

State-level NPK fertilizer consumption data for Indian agriculture.

Sources:
    - Department of Fertilizers, Ministry of Chemicals and Fertilizers
    - Fertilizer Statistics (FAI - Fertiliser Association of India)

Metric: NPK consumption in kg per hectare of net sown area.
    N = Nitrogen, P = Phosphorus, K = Potassium

The data is static (updated yearly) and stored as a CSV.
No API keys, no network calls, no rate limits.

Usage:
    from src.components.fertilizer_data import (
        get_fertilizer_for_state,
        enrich_dataframe_with_fertilizer,
    )

    kg_ha = get_fertilizer_for_state("Punjab")   # 248.0
    df = enrich_dataframe_with_fertilizer(df, state_col="state")
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
FERTILIZER_CSV: Path = PROJECT_ROOT / "data" / "raw" / "fertilizer_npk.csv"

FERTILIZER_FEATURE: str = "npk_consumption_kg_per_ha"
DEFAULT_FERTILIZER: float = 120.0  # national approximate fallback

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

def _load_fertilizer_data() -> Dict[str, float]:
    """Load NPK consumption from CSV into a dict keyed by state name."""
    if not FERTILIZER_CSV.exists():
        logger.warning("Fertilizer CSV not found: %s — using defaults", FERTILIZER_CSV)
        return {}

    try:
        df = pd.read_csv(FERTILIZER_CSV)
        df.columns = [c.strip().lower() for c in df.columns]

        if "state" not in df.columns or "npk_consumption_kg_per_ha" not in df.columns:
            logger.warning("Fertilizer CSV missing required columns")
            return {}

        data: Dict[str, float] = {}
        for _, row in df.iterrows():
            state_name = str(row["state"]).strip()
            val = float(row["npk_consumption_kg_per_ha"])
            data[state_name] = val

        logger.info("Loaded NPK fertilizer data for %d states", len(data))
        return data

    except Exception as exc:
        logger.error("Failed to load fertilizer CSV: %s", exc)
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
_fertilizer_cache: Optional[Dict[str, float]] = None


def _get_cache() -> Dict[str, float]:
    """Get or initialize the module-level fertilizer cache."""
    global _fertilizer_cache
    if _fertilizer_cache is None:
        _fertilizer_cache = _load_fertilizer_data()
    return _fertilizer_cache


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def get_fertilizer_for_state(state: str) -> float:
    """Get NPK consumption (kg/ha) for a state.

    Args:
        state: Indian state name.

    Returns:
        NPK consumption in kg per hectare.
        Returns DEFAULT_FERTILIZER if state not found.
    """
    cache = _get_cache()
    if not cache:
        return DEFAULT_FERTILIZER

    normalized = _normalize_state_name(state)

    if normalized in cache:
        return cache[normalized]

    lower = normalized.lower()
    for key, val in cache.items():
        if key.lower() == lower:
            return val

    for key, val in cache.items():
        if lower in key.lower() or key.lower() in lower:
            return val

    logger.debug(
        "Fertilizer data not found for '%s' — using default %.0f kg/ha",
        state, DEFAULT_FERTILIZER,
    )
    return DEFAULT_FERTILIZER


def get_all_fertilizer() -> Dict[str, float]:
    """Get NPK consumption for all states in the dataset."""
    return dict(_get_cache())


def enrich_dataframe_with_fertilizer(
    df: pd.DataFrame,
    state_col: str = "state",
) -> pd.DataFrame:
    """Add NPK consumption column to a DataFrame by mapping state names.

    Args:
        df: Input DataFrame with a state column.
        state_col: Name of the state column.

    Returns:
        Enriched DataFrame with 1 additional column: npk_consumption_kg_per_ha.
    """
    if state_col not in df.columns:
        logger.warning("State column '%s' not found — filling with default", state_col)
        df[FERTILIZER_FEATURE] = DEFAULT_FERTILIZER
        return df

    unique_states = df[state_col].dropna().unique().tolist()
    logger.info("Mapping NPK fertilizer consumption for %d unique states", len(unique_states))

    state_fertilizer: Dict[str, float] = {}
    for state_name in unique_states:
        state_fertilizer[state_name] = get_fertilizer_for_state(str(state_name))

    df[FERTILIZER_FEATURE] = df[state_col].map(state_fertilizer).fillna(DEFAULT_FERTILIZER)

    logger.info("Fertilizer enrichment complete: added %s", FERTILIZER_FEATURE)
    return df


def get_supported_states() -> List[str]:
    """Return list of all states with fertilizer data."""
    return sorted(_get_cache().keys())