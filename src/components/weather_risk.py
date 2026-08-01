"""
src/components/weather_risk.py

Weather Risk Alert Engine for Indian agriculture.

Analyzes NASA POWER weather data against agronomic thresholds to generate
risk alerts for:
    - DROUGHT: Low rainfall + low humidity
    - HEAT_STRESS: Elevated temperature
    - EXCESS_RAIN: Flooding / waterlogging risk
    - COLD_STRESS: Temperature below crop comfort zone
    - SOLAR_DEFICIENCY: Insufficient radiation for photosynthesis

No new data sources — consumes the existing weather cache.
No model retraining — purely analytical.

Usage:
    from src.components.weather_risk import assess_state_risk, assess_all_states

    assessment = assess_state_risk("Rajasthan")
    print(assessment.overall_risk)  # "HIGH"
    for alert in assessment.alerts:
        print(f"  {alert.icon} {alert.title}: {alert.message}")
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.components.weather_data import (
    WEATHER_FEATURES,
    get_weather_for_state,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Agronomic thresholds (Indian context)
#
#  Each rule: (metric, operator, threshold, level, category, title_template)
#
#  Thresholds are based on Indian Agricultural Research Institute (ICAR)
#  guidelines for rainfed and irrigated agriculture across major agro-
#  climatic zones.
# ---------------------------------------------------------------------------

# Threshold definitions: (metric, comparison, threshold, alert_level, category)
_RULES: List[Tuple[str, str, float, str, str]] = [
    # --- DROUGHT ---
    ("total_precipitation", "<", 400.0, "CRITICAL", "DROUGHT"),
    ("total_precipitation", "<", 600.0, "HIGH", "DROUGHT"),
    ("total_precipitation", "<", 800.0, "MODERATE", "DROUGHT"),
    ("mean_relative_humidity", "<", 35.0, "HIGH", "DROUGHT"),
    ("mean_relative_humidity", "<", 45.0, "MODERATE", "DROUGHT"),

    # --- EXCESS RAIN ---
    ("total_precipitation", ">", 2500.0, "CRITICAL", "EXCESS_RAIN"),
    ("total_precipitation", ">", 2000.0, "HIGH", "EXCESS_RAIN"),
    ("total_precipitation", ">", 1600.0, "MODERATE", "EXCESS_RAIN"),

    # --- HEAT STRESS ---
    ("mean_temperature", ">", 30.0, "CRITICAL", "HEAT_STRESS"),
    ("mean_temperature", ">", 28.0, "HIGH", "HEAT_STRESS"),
    ("mean_temperature", ">", 26.5, "MODERATE", "HEAT_STRESS"),

    # --- COLD STRESS ---
    ("mean_temperature", "<", 16.0, "CRITICAL", "COLD_STRESS"),
    ("mean_temperature", "<", 19.0, "HIGH", "COLD_STRESS"),
    ("mean_temperature", "<", 21.0, "MODERATE", "COLD_STRESS"),

    # --- SOLAR DEFICIENCY ---
    ("mean_solar_radiation", "<", 13.0, "HIGH", "SOLAR_DEFICIENCY"),
    ("mean_solar_radiation", "<", 15.0, "MODERATE", "SOLAR_DEFICIENCY"),
]

# Severity ordering for overall risk aggregation
_LEVEL_ORDER: Dict[str, int] = {
    "OPTIMAL": 0,
    "LOW": 1,
    "MODERATE": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}

# Icons for each level
_LEVEL_ICONS: Dict[str, str] = {
    "OPTIMAL": "\U0001f7e2",   # green circle
    "LOW": "\U0001f7e1",       # yellow circle
    "MODERATE": "\U0001f7e0",  # orange circle
    "HIGH": "\U0001f534",      # red circle
    "CRITICAL": "\U0001f534",  # red circle (same, differentiated by text)
}

# Category display names
_CATEGORY_NAMES: Dict[str, str] = {
    "DROUGHT": "Drought Risk",
    "EXCESS_RAIN": "Excess Rainfall",
    "HEAT_STRESS": "Heat Stress",
    "COLD_STRESS": "Cold Stress",
    "SOLAR_DEFICIENCY": "Low Solar Radiation",
}

# Metric display info
_METRIC_INFO: Dict[str, Dict[str, str]] = {
    "total_precipitation": {"label": "Rainfall", "unit": "mm/yr", "fmt": ".0f"},
    "mean_temperature": {"label": "Temperature", "unit": "\u00b0C", "fmt": ".1f"},
    "mean_relative_humidity": {"label": "Humidity", "unit": "%", "fmt": ".0f"},
    "mean_solar_radiation": {"label": "Solar Radiation", "unit": "MJ/m\u00b2/day", "fmt": ".1f"},
}

# Reference norms (Indian agricultural averages, approximate)
# Used when computing deviation for alert messages
_DEFAULT_NORMS: Dict[str, float] = {
    "total_precipitation": 1100.0,
    "mean_temperature": 25.5,
    "mean_relative_humidity": 62.0,
    "mean_solar_radiation": 17.5,
}


# ---------------------------------------------------------------------------
#  Data classes
# ---------------------------------------------------------------------------

@dataclass
class RiskAlert:
    """A single risk alert."""
    level: str              # "CRITICAL", "HIGH", "MODERATE"
    category: str           # "DROUGHT", "HEAT_STRESS", etc.
    icon: str               # Emoji icon
    title: str              # "DROUGHT ALERT"
    message: str            # "Rainfall 320mm vs norm 1100mm (-71%)"
    metric_name: str        # "total_precipitation"
    current_value: float    # 320.0
    threshold: float        # 400.0 (the rule threshold that triggered)
    norm_value: float       # 1100.0 (reference norm)
    deviation_pct: float    # -70.9


@dataclass
class RiskAssessment:
    """Complete risk assessment for a state."""
    state: str
    overall_risk: str                           # "CRITICAL", "HIGH", "MODERATE", "LOW", "OPTIMAL"
    overall_icon: str                           # Emoji
    alerts: List[RiskAlert] = field(default_factory=list)
    weather: Dict[str, float] = field(default_factory=dict)
    summary: str = ""                           # One-line human summary


# ---------------------------------------------------------------------------
#  Norm computation
# ---------------------------------------------------------------------------

def _compute_state_norms() -> Dict[str, float]:
    """Compute reference norms from all cached state weather data.

    Falls back to hardcoded Indian averages if cache is empty.
    """
    from src.components.weather_data import _load_cache

    cache = _load_cache()
    if not cache:
        return dict(_DEFAULT_NORMS)

    # Collect values from all cached entries that have weather data
    collected: Dict[str, List[float]] = {feat: [] for feat in WEATHER_FEATURES}
    for state_name, entry in cache.items():
        for feat in WEATHER_FEATURES:
            val = entry.get(feat)
            if val is not None and isinstance(val, (int, float)) and val > 0:
                collected[feat].append(float(val))

    # Compute medians (more robust than mean for small samples)
    norms: Dict[str, float] = {}
    for feat in WEATHER_FEATURES:
        vals = collected[feat]
        if vals:
            norms[feat] = round(float(np.median(vals)), 1)
        else:
            norms[feat] = _DEFAULT_NORMS[feat]

    return norms


# ---------------------------------------------------------------------------
#  Assessment engine
# ---------------------------------------------------------------------------

def _compare(value: float, operator: str, threshold: float) -> bool:
    """Evaluate a threshold comparison."""
    if operator == "<":
        return value < threshold
    elif operator == ">":
        return value > threshold
    elif operator == "<=":
        return value <= threshold
    elif operator == ">=":
        return value >= threshold
    return False


def assess_state_risk(
    state: str,
    weather: Optional[Dict[str, float]] = None,
    norms: Optional[Dict[str, float]] = None,
) -> RiskAssessment:
    """Assess weather risk for a single state.

    Args:
        state: Indian state name.
        weather: Pre-fetched weather dict. If None, fetches from cache/API.
        norms: Reference norms dict. If None, computed from all cached data.

    Returns:
        RiskAssessment with alerts, overall risk level, and summary.
    """
    if weather is None:
        weather = get_weather_for_state(state)

    if norms is None:
        norms = _compute_state_norms()

    alerts: List[RiskAlert] = []
    max_level_score = 0
    max_level = "OPTIMAL"

    # Evaluate each rule
    for metric, operator, threshold, level, category in _RULES:
        value = weather.get(metric)
        if value is None:
            continue

        if _compare(value, operator, threshold):
            # This rule fired
            norm_val = norms.get(metric, _DEFAULT_NORMS.get(metric, 0))
            dev_pct = ((value - norm_val) / norm_val * 100) if norm_val > 0 else 0.0

            info = _METRIC_INFO.get(metric, {"label": metric, "unit": "", "fmt": ".1f"})
            fmt = info["fmt"]

            message = (
                f"{info['label']} {value:{fmt}}{info['unit']} "
                f"vs norm {norm_val:{fmt}}{info['unit']} "
                f"({dev_pct:+.0f}%)"
            )

            alert = RiskAlert(
                level=level,
                category=category,
                icon=_LEVEL_ICONS.get(level, "\u26a0\ufe0f"),
                title=f"{_CATEGORY_NAMES[category].upper()}",
                message=message,
                metric_name=metric,
                current_value=value,
                threshold=threshold,
                norm_value=norm_val,
                deviation_pct=round(dev_pct, 1),
            )
            alerts.append(alert)

            # Track highest severity
            level_score = _LEVEL_ORDER.get(level, 0)
            if level_score > max_level_score:
                max_level_score = level_score
                max_level = level

    # Deduplicate: keep only the highest-severity alert per category
    best_by_category: Dict[str, RiskAlert] = {}
    for alert in alerts:
        existing = best_by_category.get(alert.category)
        if existing is None or _LEVEL_ORDER.get(alert.level, 0) > _LEVEL_ORDER.get(existing.level, 0):
            best_by_category[alert.category] = alert

    deduplicated = sorted(
        best_by_category.values(),
        key=lambda a: _LEVEL_ORDER.get(a.level, 0),
        reverse=True,
    )

    # Generate summary
    if not deduplicated:
        summary = f"All indicators within normal range for {state}."
        overall = "OPTIMAL"
    else:
        top_alerts = [f"{a.icon} {a.title}" for a in deduplicated[:2]]
        summary = f"{state}: {' | '.join(top_alerts)}"
        overall = max_level

    return RiskAssessment(
        state=state,
        overall_risk=overall,
        overall_icon=_LEVEL_ICONS.get(overall, "\u26a0\ufe0f"),
        alerts=deduplicated,
        weather=weather,
        summary=summary,
    )


def assess_all_states(
    states: Optional[List[str]] = None,
) -> List[RiskAssessment]:
    """Assess weather risk for all states.

    Args:
        states: List of state names. If None, uses all states from schema.

    Returns:
        List of RiskAssessment, sorted by risk severity (highest first).
    """
    if states is None:
        try:
            from src.utils.geo_coords import get_supported_states
            states = get_supported_states()
        except Exception:
            states = []

    # Pre-compute norms once
    norms = _compute_state_norms()

    assessments: List[RiskAssessment] = []
    for state in sorted(states):
        try:
            assessment = assess_state_risk(state, norms=norms)
            assessments.append(assessment)
        except Exception as exc:
            logger.warning("Risk assessment failed for %s: %s", state, exc)

    # Sort by severity (CRITICAL first)
    assessments.sort(
        key=lambda a: _LEVEL_ORDER.get(a.overall_risk, 0),
        reverse=True,
    )

    return assessments


def get_risk_summary_table(
    assessments: Optional[List[RiskAssessment]] = None,
) -> List[Dict[str, Any]]:
    """Convert assessments to a flat table for dashboard display.

    Returns:
        List of dicts with keys: State, Risk Level, Icon, Temp, Rain,
        Humidity, Solar, Alerts, Summary.
    """
    if assessments is None:
        assessments = assess_all_states()

    rows: List[Dict[str, Any]] = []
    for a in assessments:
        wx = a.weather
        alert_text = " | ".join(
            f"{al.icon} {al.title}: {al.message}" for al in a.alerts
        ) if a.alerts else "\U0001f7e2 Optimal"

        rows.append({
            "State": a.state,
            "Risk Level": a.overall_risk,
            "Icon": a.overall_icon,
            "Temp (\u00b0C)": f"{wx.get('mean_temperature', 0):.1f}",
            "Rain (mm)": f"{wx.get('total_precipitation', 0):.0f}",
            "Humidity (%)": f"{wx.get('mean_relative_humidity', 0):.0f}",
            "Solar (MJ)": f"{wx.get('mean_solar_radiation', 0):.1f}",
            "Alerts": alert_text,
            "Summary": a.summary,
        })

    return rows