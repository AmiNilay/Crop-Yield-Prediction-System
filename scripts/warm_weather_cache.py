"""
scripts/warm_weather_cache.py

Pre-compute weather for all unique (state, year) combos.
Run once before training to warm the cache (~10 min, ~540 API calls).
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import pandas as pd
from src.components.weather_data import (
    preload_all_state_year_combos,
    get_cache_stats,
    CACHE_PATH,
)


def main():
    df = pd.read_csv("data/raw/crop_production_india.csv")
    combos = df[["state", "crop_year"]].drop_duplicates()

    print()
    print(f"  Dataset     : {len(df):,} rows")
    print(f"  States      : {df['state'].nunique()}")
    print(f"  Year range  : {df['crop_year'].min()} — {df['crop_year'].max()}")
    print(f"  Unique combos: {len(combos)}")
    print()

    # Check existing cache
    if CACHE_PATH.exists():
        existing = get_cache_stats()
        print(f"  Existing cache: {existing['total_entries']} entries")
        print(f"  Already cached: {existing['total_entries']} / {len(combos)} combos")
        print(f"  Remaining     : {len(combos) - existing['total_entries']} to fetch")
    else:
        print(f"  No cache found — will fetch all {len(combos)} combos")

    print()
    print("  Starting preload (1 sec rate limit between API calls)...")
    print()

    preload_all_state_year_combos(df, state_col="state")

    stats = get_cache_stats()
    print()
    print(f"  Cache stats:")
    print(f"    Total entries : {stats['total_entries']}")
    print(f"    States        : {len(stats['states'])}")
    print(f"    Years         : {len(stats['years'])}")
    print(f"    Cache file    : {stats['cache_file']}")
    print(f"    Cache size    : {stats['cache_size_kb']:.1f} KB")
    print()


if __name__ == "__main__":
    main()