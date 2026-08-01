"""
src/utils/geo_coords.py

Maps Indian states to approximate centroid latitude/longitude coordinates
for NASA POWER API weather data queries.

Falls back to all-India centroid (20.5937, 78.9629) for unknown states.
"""

from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------------
#  State centroids (latitude, longitude)
#  Source: Survey of India approximate administrative centroids
# ---------------------------------------------------------------------------
_STATE_COORDINATES: Dict[str, Tuple[float, float]] = {
    "Andhra Pradesh": (15.9129, 79.7400),
    "Arunachal Pradesh": (28.2180, 94.7278),
    "Assam": (26.2006, 92.9376),
    "Bihar": (25.0961, 85.3131),
    "Chhattisgarh": (21.2787, 81.8661),
    "Goa": (15.2993, 74.1240),
    "Gujarat": (22.2587, 71.1924),
    "Haryana": (29.0588, 76.0856),
    "Himachal Pradesh": (31.1048, 77.1734),
    "Jammu and Kashmir": (33.7782, 76.5762),
    "Jharkhand": (23.6102, 85.2799),
    "Karnataka": (15.3173, 75.7139),
    "Kerala": (10.8505, 76.2711),
    "Madhya Pradesh": (22.9734, 78.6569),
    "Maharashtra": (19.7515, 75.7139),
    "Manipur": (24.6637, 93.9063),
    "Meghalaya": (25.4670, 91.3662),
    "Mizoram": (23.1645, 92.9376),
    "Nagaland": (26.1584, 94.5624),
    "Odisha": (20.9517, 85.0985),
    "Orissa": (20.9517, 85.0985),  # Legacy name
    "Punjab": (31.1471, 75.3412),
    "Rajasthan": (27.0238, 74.2179),
    "Sikkim": (27.5330, 88.5122),
    "Tamil Nadu": (11.1271, 78.6569),
    "Telangana": (18.1124, 79.0193),
    "Tripura": (23.9408, 91.9882),
    "Uttar Pradesh": (26.8467, 80.9462),
    "Uttarakhand": (30.0668, 79.0193),
    "West Bengal": (22.9868, 87.8550),
}

# All-India centroid (fallback)
_INDIA_CENTROID: Tuple[float, float] = (20.5937, 78.9629)


def get_coordinates(state: str) -> Tuple[float, float]:
    """Look up latitude and longitude for an Indian state.

    Args:
        state: State name (case-insensitive, partial match supported).

    Returns:
        Tuple of (latitude, longitude).

    Examples:
        >>> get_coordinates("Punjab")
        (31.1471, 75.3412)
        >>> get_coordinates("Unknown State")
        (20.5937, 78.9629)
    """
    if not state:
        return _INDIA_CENTROID

    state_clean = state.strip()

    # Exact match
    if state_clean in _STATE_COORDINATES:
        lat, lon = _STATE_COORDINATES[state_clean]
        return lat, lon

    # Case-insensitive match
    state_lower = state_clean.lower()
    for key, coords in _STATE_COORDINATES.items():
        if key.lower() == state_lower:
            return coords

    # Partial match
    for key, coords in _STATE_COORDINATES.items():
        if state_lower in key.lower() or key.lower() in state_lower:
            return coords

    return _INDIA_CENTROID


def get_supported_states() -> list:
    """Return list of all mapped state names."""
    return sorted(_STATE_COORDINATES.keys())