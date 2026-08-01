import sys, os
sys.path.insert(0, ".")
os.chdir(".")

from src.components.forecast_data import (
    get_forecast_for_state,
    get_forecast_with_historical_comparison,
)

print("Fetching Punjab KHARIF forecast...")
wx = get_forecast_for_state("Punjab", "KHARIF")
for k in ["mean_temperature", "total_precipitation", "mean_relative_humidity", "mean_solar_radiation"]:
    print(f"  {k}: {wx[k]}")
print(f"  confidence: {wx['forecast_confidence']}")
print(f"  coverage: {wx['coverage_pct']}%")
print(f"  horizon: {wx['forecast_horizon_days']} days")
print(f"  spread: {wx['forecast_spread']}")
if wx.get("warning"):
    print(f"  warning: {wx['warning']}")

print()
print("Fetching with historical comparison...")
wx2 = get_forecast_with_historical_comparison("Punjab", "KHARIF")
if wx2.get("historical_delta"):
    print("  Delta vs historical:")
    for k, v in wx2["historical_delta"].items():
        pct = wx2["historical_delta_pct"].get(k, 0)
        print(f"    {k}: {v:+.2f} ({pct:+.1f}%)")
else:
    print("  No historical delta available")