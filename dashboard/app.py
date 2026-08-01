"""
dashboard/app.py

AI-Powered Crop Yield Prediction System — Streamlit Dashboard
Version 3.2.1 — Bugfixes for checkbox, button placement, badge dedup

Fixes applied:
  BUG-1: Removed ☐ unicode from checkbox label (was rendering double checkbox)
  BUG-2: Moved Refresh Weather next to weather card, not between badges
  P1:    Removed redundant sidebar irrigation/NPK badges
  P2:    Compressed forecast warning to caption
  P3:    Consistent icons: 👨‍🌾 Your input / 🏛️ State default
"""

# =========================================================================
#  SECTION 1 — Imports & Environment
# =========================================================================

import json as json_lib
import math
import os
import pickle
import re
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.components.weather_data import (
    PERIOD_LABEL,
    WEATHER_FEATURES,
    enrich_dataframe_with_weather,
    get_weather_for_state,
)
from src.components.forecast_data import (
    FORECAST_FEATURES,
    get_forecast_for_state,
    get_forecast_with_historical_comparison,
)
from src.components.weather_risk import (
    assess_all_states,
    assess_state_risk,
)
from src.components.irrigation_data import (
    IRRIGATION_FEATURE,
    get_irrigation_for_state,
    enrich_dataframe_with_irrigation,
)
from src.components.fertilizer_data import (
    FERTILIZER_FEATURE,
    get_fertilizer_for_state,
    enrich_dataframe_with_fertilizer,
)
from src.utils.geo_coords import get_coordinates

HAS_XGB: bool
try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

HAS_SHAP: bool
try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

MODELS_DIR: Path = PROJECT_ROOT / "models"
DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROC_DIR: Path = PROJECT_ROOT / "data" / "processed"
SCHEMA_PATH: Path = MODELS_DIR / "feature_schema.json"

GAUGE_MAX_BY_CROP: Dict[str, float] = {
    "SUGARCANE": 1000.0, "POTATO": 400.0, "ONION": 500.0,
    "BANANA": 400.0, "RICE": 60.0, "WHEAT": 60.0,
    "MAIZE": 60.0, "COTTON": 50.0, "JOWAR": 50.0,
    "BAJRA": 50.0, "GRAM": 30.0, "GROUNDNUT": 50.0,
    "SESAMUM": 20.0, "URAD": 20.0, "MOONG(GREEN GRAM)": 20.0,
    "RAPESEED &MUSTARD": 30.0, "ARHAR/TUR": 30.0,
    "DRY CHILLIES": 80.0, "SUNFLOWER": 40.0,
}


# =========================================================================
#  SECTION 2 — Page Configuration & Styles
# =========================================================================

st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="\U0001f33e",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Source+Serif+4:ital,opsz,wght@0,8..60,300;0,8..60,400;0,8..60,600;1,8..60,400&display=swap" rel="stylesheet">
<style>
    .stApp { background-color: #0c1a0c; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #142014 0%, #0c180c 100%);
        border-right: 1px solid rgba(200,230,201,0.07);
    }
    h1,h2,h3,h4,h5,h6 {
        font-family: 'DM Serif Display', Georgia, serif !important;
        color: #e8f5e9 !important;
    }
    p,li,span,label,.stMarkdown,.stCaption {
        font-family: 'Source Serif 4', Georgia, serif;
        color: #c8e6c9;
    }
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.025);
        border: 1px solid rgba(200,230,201,0.08);
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
    }
    [data-testid="stMetricValue"] {
        font-family: 'DM Serif Display', serif !important;
        color: #f9a825 !important;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'Source Serif 4', serif !important;
        color: #a5d6a7 !important;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 2rem; }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Source Serif 4', serif;
        color: #81c784;
        padding: 0.6rem 1rem;
    }
    .stTabs [aria-selected="true"] {
        color: #f9a825 !important;
        border-bottom-color: #f9a825 !important;
    }
    .yield-card {
        background: linear-gradient(135deg, rgba(56,142,60,0.12) 0%, rgba(27,94,32,0.08) 100%);
        border: 1px solid rgba(76,175,80,0.25);
        border-radius: 14px;
        padding: 2.5rem 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .yield-card h2 {
        font-size: 3.2rem !important;
        color: #f9a825 !important;
        margin-bottom: 0 !important;
    }
    .yield-card p {
        color: #a5d6a7 !important;
        font-size: 1.15rem;
        margin-top: 0.3rem;
    }
    .weather-card {
        background: linear-gradient(135deg, rgba(30,136,229,0.08) 0%, rgba(13,71,161,0.06) 100%);
        border: 1px solid rgba(66,165,245,0.18);
        border-radius: 10px;
        padding: 0.8rem 1rem;
    }
    .forecast-card {
        background: linear-gradient(135deg, rgba(156,39,176,0.08) 0%, rgba(103,58,183,0.06) 100%);
        border: 1px solid rgba(186,104,200,0.18);
        border-radius: 10px;
        padding: 0.8rem 1rem;
    }
    .risk-badge {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.15rem 0;
    }
    .risk-CRITICAL { background: rgba(244,67,54,0.2); color: #ef9a9a; border: 1px solid rgba(244,67,54,0.3); }
    .risk-HIGH { background: rgba(255,87,34,0.2); color: #ffab91; border: 1px solid rgba(255,87,34,0.3); }
    .risk-MODERATE { background: rgba(255,193,7,0.2); color: #ffe082; border: 1px solid rgba(255,193,7,0.3); }
    .risk-LOW { background: rgba(76,175,80,0.15); color: #a5d6a7; border: 1px solid rgba(76,175,80,0.2); }
    .risk-OPTIMAL { background: rgba(76,175,80,0.1); color: #81c784; border: 1px solid rgba(76,175,80,0.15); }
    hr { border-color: rgba(200,230,201,0.08) !important; }
    .stAlert { font-family: 'Source Serif 4', serif; }
</style>
""",
    unsafe_allow_html=True,
)

plt.style.use("dark_background")
plt.rcParams.update({
    "figure.facecolor": "#0c1a0c", "axes.facecolor": "#142014",
    "axes.edgecolor": "#2e4a2e", "text.color": "#c8e6c9",
    "axes.labelcolor": "#c8e6c9", "xtick.color": "#81c784",
    "ytick.color": "#81c784", "grid.color": "#1e3a1e", "font.family": "serif",
})


# =========================================================================
#  SECTION 3 — Helpers
# =========================================================================

COLUMN_DISPLAY: Dict[str, str] = {
    "crop": "Crop", "state": "State", "season": "Season",
    "crop_year": "Year", "area": "Area (Ha)",
    "yield": "Yield (q/ha)", "mean_temperature": "Avg Temp (\u00b0C)",
    "total_precipitation": "Rainfall (mm/yr)",
    "mean_relative_humidity": "Humidity (%)",
    "mean_solar_radiation": "Solar Rad (MJ/m\u00b2/day)",
    "irrigation_coverage_pct": "Irrigation (%)",
    "npk_consumption_kg_per_ha": "NPK (kg/ha)",
}


def friendly_name(col: str) -> str:
    return COLUMN_DISPLAY.get(col, col.replace("_", " ").title())


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = []
    for col in df.columns:
        new_col = col.strip().lower()
        new_col = re.sub(r"[^a-z0-9]+", "_", new_col)
        cleaned.append(new_col.strip("_"))
    df.columns = cleaned
    rename_map = {}
    for col in df.columns:
        if col.startswith("costof_"):
            rename_map[col] = "cost_of_" + col[len("costof_"):]
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
    yield_candidates = [c for c in df.columns if "yield" in c]
    if yield_candidates and yield_candidates[0] != "yield":
        df.rename(columns={yield_candidates[0]: "yield"}, inplace=True)
    for exc in ["production", "district"]:
        if exc in df.columns:
            df.drop(columns=[exc], inplace=True)
    return df


def _risk_badge_html(level: str, text: str) -> str:
    return f'<span class="risk-badge risk-{level}">{text}</span>'


def _category_icon(pct: float, t=(80, 60, 40, 20)) -> Tuple[str, str]:
    if pct >= t[0]:
        return "Very High", "\U0001f7e2"
    elif pct >= t[1]:
        return "High", "\U0001f7e2"
    elif pct >= t[2]:
        return "Moderate", "\U0001f7e0"
    elif pct >= t[3]:
        return "Low", "\U0001f534"
    return "Very Low", "\U0001f534"


def _fert_cat(kg_ha: float) -> Tuple[str, str]:
    if kg_ha >= 200:
        return "Very High", "\U0001f7e2"
    elif kg_ha >= 150:
        return "High", "\U0001f7e2"
    elif kg_ha >= 100:
        return "Moderate", "\U0001f7e0"
    elif kg_ha >= 50:
        return "Low", "\U0001f534"
    return "Very Low", "\U0001f534"


def _confidence_icon(conf: str) -> str:
    return {"high": "\U0001f7e2", "medium": "\U0001f7e0", "low": "\U0001f534"}.get(conf, "")


def _delta_arrow(val: float) -> str:
    if val > 0.5:
        return "\u2b06\ufe0f"
    elif val < -0.5:
        return "\u2b07\ufe0f"
    return "\u27a1\ufe0f"


def _fmt_int(val: float) -> str:
    """Format number as int if close to whole, else 1 decimal place."""
    if abs(val - round(val)) < 0.05:
        return f"{int(round(val))}"
    return f"{val:.1f}"


def _source_badge_user() -> str:
    return "\U0001f468\u200d\U0001f33e Your input"


def _source_badge_state() -> str:
    return "\U0001f3db\ufe0f State default"


# =========================================================================
#  SECTION 4 — Cached Resources
# =========================================================================

@st.cache_data
def load_schema():
    if not SCHEMA_PATH.exists():
        return None
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json_lib.load(f)


@st.cache_resource
def load_preprocessor():
    path = MODELS_DIR / "preprocessor.pkl"
    if not path.exists():
        return None
    try:
        return joblib.load(str(path))
    except Exception:
        pass
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _fix_xgb_base_score(booster):
    try:
        config = json_lib.loads(booster.save_config())
        bs = config.get("learner", {}).get("learner_model_param", {}).get("base_score", "")
        if not isinstance(bs, str) or not bs.strip().startswith("["):
            return booster
        inner = bs.strip().strip("[]").split(",")[0].strip()
        bs_float = float(inner)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
            tmp_path = tmp.name
        booster.save_model(tmp_path)
        with open(tmp_path, "r") as f:
            model_data = json_lib.load(f)
        try:
            model_data["learner"]["learner_model_param"]["base_score"] = str(bs_float)
        except (KeyError, TypeError):
            os.unlink(tmp_path)
            return booster
        with open(tmp_path, "w") as f:
            json_lib.dump(model_data, f)
        fixed = xgb.Booster()
        fixed.load_model(tmp_path)
        os.unlink(tmp_path)
        return fixed
    except Exception:
        return booster


@st.cache_resource
def load_booster():
    if not HAS_XGB:
        return None
    for fname in ("model.json", "model.ubj"):
        path = MODELS_DIR / fname
        if path.exists():
            try:
                booster = xgb.Booster()
                booster.load_model(str(path))
                return _fix_xgb_base_score(booster)
            except Exception:
                continue
    return None


@st.cache_resource
def load_shap_explainer():
    if not HAS_SHAP:
        return None
    booster = load_booster()
    if booster is None:
        return None
    try:
        return shap.TreeExplainer(booster)
    except Exception:
        pass
    for mo in ("raw", "margin"):
        try:
            return shap.TreeExplainer(booster, model_output=mo)
        except Exception:
            continue
    preprocessor_local = load_preprocessor()
    schema_local = load_schema()
    if preprocessor_local and schema_local:
        try:
            feat_cols = schema_local["feature_columns"]
            for csv_name in ("train.csv", "test.csv", "crop_production_india.csv"):
                for csv_dir in (DATA_PROC_DIR, DATA_RAW_DIR):
                    csv_path = csv_dir / csv_name
                    if csv_path.exists():
                        bg_df = pd.read_csv(csv_path)
                        bg_df = clean_columns(bg_df)
                        for enrich_fn, feat in [
                            (enrich_dataframe_with_weather, WEATHER_FEATURES),
                            (enrich_dataframe_with_irrigation, IRRIGATION_FEATURE),
                            (enrich_dataframe_with_fertilizer, FERTILIZER_FEATURE),
                        ]:
                            need = feat if isinstance(feat, str) else [c for c in feat if c not in bg_df.columns]
                            if need and "state" in bg_df.columns:
                                try:
                                    bg_df = enrich_fn(bg_df, state_col="state")
                                except Exception:
                                    pass
                        if all(c in bg_df.columns for c in feat_cols):
                            X_bg = preprocessor_local.transform(bg_df[feat_cols].head(30))
                            if hasattr(X_bg, "toarray"):
                                X_bg = X_bg.toarray()
                            X_bg = np.asarray(X_bg, dtype=np.float32)
                            out_names = schema_local.get("output_feature_names", [])

                            def _predict(X):
                                dm = xgb.DMatrix(X, feature_names=out_names or None)
                                return booster.predict(dm)

                            return shap.KernelExplainer(_predict, shap.kmeans(X_bg, 10))
        except Exception:
            pass
    return None


@st.cache_data
def load_dataset():
    for path in (DATA_RAW_DIR / "crop_production_india.csv", DATA_PROC_DIR / "train.csv"):
        if path.exists():
            try:
                df = pd.read_csv(path)
                df = clean_columns(df)
                if "state" in df.columns:
                    missing_wx = [c for c in WEATHER_FEATURES if c not in df.columns]
                    if missing_wx:
                        try:
                            df = enrich_dataframe_with_weather(df, state_col="state")
                        except Exception:
                            for feat in WEATHER_FEATURES:
                                if feat not in df.columns:
                                    df[feat] = np.nan
                    if IRRIGATION_FEATURE not in df.columns:
                        try:
                            df = enrich_dataframe_with_irrigation(df, state_col="state")
                        except Exception:
                            df[IRRIGATION_FEATURE] = np.nan
                    if FERTILIZER_FEATURE not in df.columns:
                        try:
                            df = enrich_dataframe_with_fertilizer(df, state_col="state")
                        except Exception:
                            df[FERTILIZER_FEATURE] = np.nan
                return df
            except Exception:
                continue
    return None


def predict_yield(booster, preprocessor, input_df, feature_names):
    try:
        X_t = preprocessor.transform(input_df)
        if hasattr(X_t, "toarray"):
            X_t = X_t.toarray()
        X_arr = np.asarray(X_t, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        dmat = xgb.DMatrix(X_arr, feature_names=feature_names or None)
        return float(booster.predict(dmat)[0]), X_arr
    except Exception:
        return None, None


@st.cache_data
def compute_evaluation_metrics():
    try:
        schema_local = load_schema()
        preprocessor_local = load_preprocessor()
        booster_local = load_booster()
        if not all((schema_local, preprocessor_local, booster_local)):
            return None
        feat_cols = schema_local["feature_columns"]
        out_names = schema_local.get("output_feature_names", [])
        crop_col_local = schema_local.get("crop_column")
        df = None
        source_name = ""
        for csv_name, csv_dir in [
            ("test.csv", DATA_PROC_DIR),
            ("train.csv", DATA_PROC_DIR),
            ("crop_production_india.csv", DATA_RAW_DIR),
        ]:
            p = csv_dir / csv_name
            if p.exists():
                candidate = pd.read_csv(p)
                candidate = clean_columns(candidate)
                if "state" in candidate.columns:
                    for enrich_fn, feat in [
                        (enrich_dataframe_with_weather, WEATHER_FEATURES),
                        (enrich_dataframe_with_irrigation, IRRIGATION_FEATURE),
                        (enrich_dataframe_with_fertilizer, FERTILIZER_FEATURE),
                    ]:
                        need = feat if isinstance(feat, str) else [c for c in feat if c not in candidate.columns]
                        if need:
                            try:
                                candidate = enrich_fn(candidate, state_col="state")
                            except Exception:
                                pass
                if (
                    len(candidate) > 5
                    and "yield" in candidate.columns
                    and all(c in candidate.columns for c in feat_cols)
                ):
                    df = candidate
                    source_name = csv_name
                    break
        if df is None:
            return None
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        X_raw = df[feat_cols]
        y_true = df["yield"].values.astype(np.float64)
        X_t = preprocessor_local.transform(X_raw)
        if hasattr(X_t, "toarray"):
            X_t = X_t.toarray()
        X_arr = np.asarray(X_t, dtype=np.float32)
        dmat = xgb.DMatrix(X_arr, feature_names=out_names or None)
        y_pred = booster_local.predict(dmat).astype(np.float64)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        mae = float(mean_absolute_error(y_true, y_pred))
        mask = y_true > 0.01
        mape = (
            float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
            if mask.any()
            else 0.0
        )
        per_crop = {}
        if crop_col_local and crop_col_local in df.columns:
            for cn in sorted(df[crop_col_local].dropna().unique()):
                cm = df[crop_col_local].values == cn
                n = int(cm.sum())
                if n < 2:
                    per_crop[cn] = {"n": n, "rmse": None, "mae": None, "r2": None}
                    continue
                yt, yp = y_true[cm], y_pred[cm]
                per_crop[cn] = {
                    "n": n,
                    "rmse": round(float(np.sqrt(mean_squared_error(yt, yp))), 4),
                    "mae": round(float(mean_absolute_error(yt, yp)), 4),
                    "r2": round(float(r2_score(yt, yp)), 4),
                }
        return {
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "mae": round(mae, 4),
            "mape": round(mape, 2),
            "n_samples": int(len(y_true)),
            "data_source": source_name,
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist(),
            "per_crop": per_crop,
        }
    except Exception:
        return None


# =========================================================================
#  SECTION 5 — Load Everything
# =========================================================================

schema = load_schema()
preprocessor = load_preprocessor()
booster = load_booster()
explainer = load_shap_explainer()
dataset = load_dataset()
metrics = compute_evaluation_metrics()

SCHEMA_OK = schema is not None
BOOSTER_OK = booster is not None
PREPROC_OK = preprocessor is not None
SHAP_OK = explainer is not None
DATA_OK = dataset is not None
METRICS_OK = metrics is not None

feature_cols = schema["feature_columns"] if SCHEMA_OK else []
cat_cols = schema.get("categorical_columns", []) if SCHEMA_OK else []
num_cols = schema.get("numerical_columns", []) if SCHEMA_OK else []
target_col = schema.get("target_column", "") if SCHEMA_OK else ""
output_feature_names = schema.get("output_feature_names", []) if SCHEMA_OK else []
crop_col = schema.get("crop_column") if SCHEMA_OK else None
state_col = schema.get("state_column") if SCHEMA_OK else None


# =========================================================================
#  SECTION 6 — Sidebar
# =========================================================================

predict_btn = False
user_inputs: Dict[str, Any] = {}
irrigation_source = "state-avg"
npk_source = "state-avg"
use_forecast = False

with st.sidebar:
    st.markdown("## \U0001f33e Crop Yield Predictor")
    st.caption("50K district-level records \u00b7 Historical + Forecast")
    st.divider()

    if not SCHEMA_OK:
        st.error("Schema not found.")
    elif not BOOSTER_OK:
        st.error("Model not found.")
    elif not PREPROC_OK:
        st.error("Preprocessor not found.")
    else:
        # ── Weather Source Toggle ──
        weather_source = st.radio(
            "\U0001f4ca Weather Data Source",
            options=[
                "Historical Baseline (NASA POWER)",
                "Next-Season Forecast (Open-Meteo)",
            ],
            index=0,
        )
        use_forecast = weather_source.startswith("Next-Season")

        st.divider()

        # ── Crop & Location ──
        st.markdown("#### \U0001f331 Crop & Location")
        for col in cat_cols:
            options = schema.get(f"unique_{col}", [])
            if options:
                user_inputs[col] = st.selectbox(
                    friendly_name(col), options=options, index=0
                )
            else:
                user_inputs[col] = st.text_input(friendly_name(col), value="")

        # Year slider
        year_options = schema.get("unique_crop_year", list(range(1997, 2015)))
        if not year_options:
            year_options = list(range(1997, 2015))
        year_val = st.slider(
            "Crop Year",
            int(min(year_options)),
            int(max(year_options)),
            int(max(year_options)),
            step=1,
        )
        user_inputs["crop_year"] = int(year_val)

        selected_state = user_inputs.get("state", "")
        selected_season = user_inputs.get("season", "KHARIF")
        selected_year = user_inputs.get("crop_year", None)

        st.divider()

        # ── Weather Preview + Refresh (grouped together) ──
        if selected_state:
            try:
                if use_forecast:
                    wx_preview = get_forecast_for_state(
                        selected_state, selected_season
                    )
                    lat, lon = get_coordinates(selected_state)
                    conf_icon = _confidence_icon(
                        wx_preview.get("forecast_confidence", "")
                    )
                    st.markdown(
                        f'<div class="forecast-card">'
                        f'<small>\U0001f52e Open-Meteo ({lat:.1f}, {lon:.1f})</small><br>'
                        f'<small>T: {wx_preview["mean_temperature"]:.1f}\u00b0C | '
                        f'R: {_fmt_int(wx_preview["total_precipitation"])}mm/yr | '
                        f'H: {wx_preview["mean_relative_humidity"]:.0f}%</small><br>'
                        f'<small>{conf_icon} {wx_preview.get("forecast_confidence", "N/A")} '
                        f'(coverage: {wx_preview.get("coverage_pct", 0):.0f}%)</small>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    # BUG FIX P2: compress forecast warning to caption
                    if wx_preview.get("warning"):
                        st.caption(f"\u26a0\ufe0f {wx_preview['warning']}")
                else:
                    wx_preview = get_weather_for_state(
                        str(selected_state), year=selected_year
                    )
                    lat, lon = get_coordinates(str(selected_state))
                    st.markdown(
                        f'<div class="weather-card">'
                        f'<small>\U0001f570\ufe0f NASA POWER ({lat:.1f}, {lon:.1f}) '
                        f"\u2014 {selected_year}</small><br>"
                        f'<small>T: {wx_preview["mean_temperature"]:.1f}\u00b0C | '
                        f'R: {_fmt_int(wx_preview["total_precipitation"])}mm | '
                        f'H: {wx_preview["mean_relative_humidity"]:.0f}%</small>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                # Risk badge (historical only)
                if not use_forecast:
                    risk_result = assess_state_risk(
                        str(selected_state), weather=wx_preview
                    )
                    st.markdown(
                        _risk_badge_html(
                            risk_result.overall_risk,
                            f"{risk_result.overall_icon} {risk_result.overall_risk} RISK",
                        ),
                        unsafe_allow_html=True,
                    )

                # BUG FIX 2: Refresh button grouped with weather card
                if st.button(
                    "\U0001f504 Refresh weather data",
                    use_container_width=True,
                    help="Re-fetch weather from API and clear cached data",
                ):
                    try:
                        load_dataset.clear()
                        compute_evaluation_metrics.clear()
                        if use_forecast:
                            from src.components.forecast_data import clear_cache

                            clear_cache()
                        else:
                            get_weather_for_state(
                                str(selected_state),
                                year=selected_year,
                                force_refresh=True,
                            )
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Failed: {exc}")

            except Exception:
                pass

        st.divider()

        # ── Your Farm ──
        st.markdown("#### \U0001f69c Your Farm")

        # Area — always user input
        area_val = st.number_input(
            "Area (Hectares)",
            min_value=0.1,
            max_value=10000.0,
            value=1.0,
            step=0.5,
            format="%.1f",
            help="Total cultivated area of your farm",
        )
        user_inputs["area"] = float(area_val)

        # Get state averages for defaults
        state_irr_avg = 40.0
        state_npk_avg = 120.0
        if selected_state:
            try:
                state_irr_avg = get_irrigation_for_state(str(selected_state))
            except Exception:
                pass
            try:
                state_npk_avg = get_fertilizer_for_state(str(selected_state))
            except Exception:
                pass

        # BUG FIX 1: Removed the ☐ unicode character from the label.
        # The old label was "☐ Override state averages" which rendered
        # a literal empty checkbox next to the real Streamlit checkbox.
        override_irr_npk = st.checkbox(
            "Override state averages",
            value=False,
            help="Check to enter your own irrigation and fertilizer values instead of state defaults",
        )

        if override_irr_npk:
            user_irr = st.slider(
                "Irrigation Coverage (%)",
                min_value=0,
                max_value=100,
                value=int(round(state_irr_avg)),
                step=5,
                help="Percentage of your farm that is irrigated",
            )
            st.caption(f"State default: {state_irr_avg:.0f}%")
            user_inputs[IRRIGATION_FEATURE] = float(user_irr)
            irrigation_source = "user"

            user_npk = st.number_input(
                "NPK Fertilizer (kg/ha)",
                min_value=0.0,
                max_value=500.0,
                value=state_npk_avg,
                step=10.0,
                format="%.0f",
                help="Your NPK consumption per hectare",
            )
            st.caption(f"State default: {state_npk_avg:.0f} kg/ha")
            user_inputs[FERTILIZER_FEATURE] = float(user_npk)
            npk_source = "user"
        else:
            user_inputs[IRRIGATION_FEATURE] = state_irr_avg
            user_inputs[FERTILIZER_FEATURE] = state_npk_avg
            irrigation_source = "state-avg"
            npk_source = "state-avg"

        # Auto-fill weather features (from selected source)
        if selected_state:
            try:
                if use_forecast:
                    wx = get_forecast_for_state(selected_state, selected_season)
                else:
                    wx = get_weather_for_state(
                        str(selected_state), year=selected_year
                    )
                for feat in WEATHER_FEATURES:
                    user_inputs[feat] = wx.get(feat, 0.0)
            except Exception:
                for feat in WEATHER_FEATURES:
                    user_inputs[feat] = 0.0

        st.divider()

        # BUG FIX 2: Predict button at bottom, prominent
        predict_btn = st.button(
            "\U0001f52e Predict Yield",
            type="primary",
            use_container_width=True,
        )


# =========================================================================
#  SECTION 7 — Main Content
# =========================================================================

st.markdown("# \U0001f33e AI-Powered Crop Yield Prediction")
st.caption(
    "District-level data (50K rows) \u00b7 15 crops \u00b7 33 states "
    "\u00b7 Historical + Forecast \u00b7 Farmer overrides"
)
st.divider()

if not SCHEMA_OK or not BOOSTER_OK or not PREPROC_OK:
    st.error(
        "**Setup required.**\n\n```powershell\n"
        "python scripts/build_preprocessor.py\n```"
    )
    st.stop()

tab_predict, tab_shap, tab_data, tab_info = st.tabs(
    ["Prediction", "SHAP Explainability", "Data Insights", "Model Info"]
)


# =====================  TAB 1 — PREDICTION  ==============================
with tab_predict:
    if predict_btn:
        input_df = pd.DataFrame([user_inputs], columns=feature_cols)
        prediction, X_arr = predict_yield(
            booster, preprocessor, input_df, output_feature_names
        )

        # Historical baseline for delta
        hist_pred = None
        if use_forecast and prediction is not None:
            hist_inputs = dict(user_inputs)
            try:
                hist_wx = get_weather_for_state(
                    str(selected_state), year=selected_year
                )
                for feat in WEATHER_FEATURES:
                    hist_inputs[feat] = hist_wx.get(feat, 0.0)
                hist_df = pd.DataFrame([hist_inputs], columns=feature_cols)
                hist_pred, _ = predict_yield(
                    booster, preprocessor, hist_df, output_feature_names
                )
                if hist_pred is not None:
                    hist_pred = max(0.0, hist_pred)
            except Exception:
                pass

        if prediction is not None:
            prediction = max(0.0, prediction)

            col_left, col_right = st.columns([1, 2])

            with col_left:
                # Yield card with optional delta
                delta_html = ""
                if use_forecast and hist_pred is not None:
                    delta = prediction - hist_pred
                    if abs(delta) > 1.0:
                        arrow = _delta_arrow(delta)
                        delta_html = (
                            f'<p style="font-size:0.9rem;color:#ffe082;'
                            f'margin-top:0.5rem;">'
                            f"{arrow} {delta:+.2f} q/ha vs historical baseline"
                            f"</p>"
                        )

                source_badge = (
                    "\U0001f52e Forecast" if use_forecast else "\U0001f570\ufe0f Historical"
                )

                st.markdown(
                    f'<div class="yield-card">'
                    f"<h2>{prediction:,.2f}</h2>"
                    f"<p>Quintal / Hectare</p>"
                    f"{delta_html}"
                    f'<p style="font-size:0.8rem;opacity:0.6;'
                    f'margin-top:0.5rem;">{source_badge}</p>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

                crop_name = str(user_inputs.get("crop", "N/A"))
                state_name_full = str(user_inputs.get("state", "N/A"))
                st.metric("Crop", crop_name)
                st.metric("State", state_name_full, help=state_name_full)
                st.metric("Season", str(user_inputs.get("season", "N/A")))
                st.metric("Year", str(int(user_inputs.get("crop_year", 0))))
                st.metric("Area", f"{float(user_inputs.get('area', 0)):,.1f} Ha")

            with col_right:
                crop_upper = crop_name.upper()
                gauge_max = GAUGE_MAX_BY_CROP.get(crop_upper, 100.0)
                if prediction > gauge_max:
                    gauge_max = math.ceil(prediction * 1.3 / 10) * 10

                fig_gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=round(prediction, 2),
                        number={
                            "suffix": " q/ha",
                            "font": {"size": 30, "color": "#f9a825"},
                        },
                        gauge={
                            "axis": {
                                "range": [0, gauge_max],
                                "tickwidth": 2,
                                "tickcolor": "#4a7c59",
                                "dtick": gauge_max / 5,
                            },
                            "bar": {"color": "#f9a825", "thickness": 0.35},
                            "bgcolor": "#142014",
                            "borderwidth": 0,
                            "steps": [
                                {
                                    "range": [0, gauge_max * 0.33],
                                    "color": "rgba(244,67,54,0.12)",
                                },
                                {
                                    "range": [gauge_max * 0.33, gauge_max * 0.66],
                                    "color": "rgba(255,193,7,0.12)",
                                },
                                {
                                    "range": [gauge_max * 0.66, gauge_max],
                                    "color": "rgba(76,175,80,0.12)",
                                },
                            ],
                        },
                        title={"text": "Predicted Yield", "font": {"size": 18}},
                    )
                )
                fig_gauge.update_layout(
                    height=350,
                    margin=dict(t=60, b=20, l=30, r=30),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={"family": "Source Serif 4, serif", "color": "#c8e6c9"},
                )
                st.plotly_chart(fig_gauge, width="stretch")

            # ── Climate context ──
            if selected_state:
                try:
                    if use_forecast:
                        fc = get_forecast_with_historical_comparison(
                            selected_state, selected_season
                        )
                        spread = fc.get("forecast_spread", {})
                        delta = fc.get("historical_delta", {})

                        st.markdown(
                            f"#### \U0001f52e Climate Forecast \u2014 "
                            f"{selected_state} (Next {selected_season})"
                        )
                        conf_icon = _confidence_icon(
                            fc.get("forecast_confidence", "")
                        )
                        st.caption(
                            f"{conf_icon} Confidence: "
                            f"{fc.get('forecast_confidence', 'N/A')} "
                            f"\u00b7 Coverage: {fc.get('coverage_pct', 0):.0f}% "
                            f"(share of forecast days with valid ensemble data) "
                            f"\u00b7 Horizon: "
                            f"{fc.get('forecast_horizon_days', 0)} days "
                            f"\u00b7 Source: Open-Meteo (CC-BY 4.0)"
                        )
                        if fc.get("warning"):
                            st.warning(fc["warning"])

                        wc1, wc2, wc3, wc4 = st.columns(4)

                        tv = fc["mean_temperature"]
                        ts = spread.get("mean_temperature", 0)
                        td = (delta.get("mean_temperature", 0) if delta else 0)
                        wc1.metric(
                            "Temperature",
                            f"{tv:.1f} \u00b0C \u00b1 {ts:.1f}",
                            delta=(
                                f"{td:+.1f}\u00b0C vs hist"
                                if abs(td) > 0.3
                                else None
                            ),
                            delta_color="inverse" if td > 2 else "normal",
                            help="Temperature \u00b1 ensemble std deviation",
                        )

                        rv = fc["total_precipitation"]
                        rs = spread.get("total_precipitation", 0)
                        rd = (delta.get("total_precipitation", 0) if delta else 0)
                        wc2.metric(
                            "Rainfall",
                            f"{_fmt_int(rv)}mm/yr \u00b1 {_fmt_int(rs)}",
                            delta=(
                                f"{rd:+.0f}mm vs hist" if abs(rd) > 50 else None
                            ),
                            delta_color="off",
                            help="Annualized precipitation \u00b1 ensemble std deviation",
                        )

                        hv = fc["mean_relative_humidity"]
                        hs = spread.get("mean_relative_humidity", 0)
                        hd = (delta.get("mean_relative_humidity", 0) if delta else 0)
                        wc3.metric(
                            "Humidity",
                            f"{hv:.0f}% \u00b1 {hs:.0f}",
                            delta=(
                                f"{hd:+.1f}% vs hist" if abs(hd) > 2 else None
                            ),
                            help="Relative humidity \u00b1 ensemble std deviation",
                        )

                        sv = fc["mean_solar_radiation"]
                        ss = spread.get("mean_solar_radiation", 0)
                        sd = (delta.get("mean_solar_radiation", 0) if delta else 0)
                        wc4.metric(
                            "Solar Radiation",
                            f"{sv:.1f} MJ/m\u00b2/d \u00b1 {ss:.1f}",
                            delta=(
                                f"{sd:+.1f} vs hist" if abs(sd) > 0.5 else None
                            ),
                            help="Shortwave irradiance \u00b1 ensemble std deviation",
                        )

                    else:
                        wx = get_weather_for_state(
                            str(selected_state), year=selected_year
                        )
                        lat, lon = get_coordinates(str(selected_state))
                        st.markdown(
                            f"#### Climate Context \u2014 "
                            f"{selected_state} ({selected_year})"
                        )
                        st.caption(f"NASA POWER ({lat:.2f}\u00b0N, {lon:.2f}\u00b0E)")
                        wc1, wc2, wc3, wc4 = st.columns(4)
                        wc1.metric("Avg Temperature", f"{wx['mean_temperature']:.1f} \u00b0C")
                        wc2.metric("Annual Rainfall", f"{_fmt_int(wx['total_precipitation'])} mm")
                        wc3.metric("Relative Humidity", f"{wx['mean_relative_humidity']:.0f}%")
                        wc4.metric(
                            "Solar Radiation",
                            f"{wx['mean_solar_radiation']:.1f} MJ/m\u00b2/day",
                        )

                    # Agricultural Inputs with source badges
                    irr_val = user_inputs.get(IRRIGATION_FEATURE, state_irr_avg)
                    irr_cat, irr_icon = _category_icon(irr_val)
                    irr_badge = (
                        _source_badge_user()
                        if irrigation_source == "user"
                        else _source_badge_state()
                    )

                    npk_val = user_inputs.get(FERTILIZER_FEATURE, state_npk_avg)
                    npk_cat, npk_icon = _fert_cat(npk_val)
                    npk_badge = (
                        _source_badge_user()
                        if npk_source == "user"
                        else _source_badge_state()
                    )

                    st.markdown("#### Agricultural Inputs")
                    ic1, ic2, ic3, ic4 = st.columns(4)
                    ic1.metric("Irrigation", f"{irr_val:.0f}%")
                    ic2.metric(
                        "Irrigation Level",
                        f"{irr_icon} {irr_cat}",
                        help=irr_badge,
                    )
                    ic3.metric("NPK Fertilizer", f"{npk_val:.0f} kg/ha")
                    ic4.metric(
                        "NPK Level",
                        f"{npk_icon} {npk_cat}",
                        help=npk_badge,
                    )

                    st.caption(
                        f"\U0001f4a7 Irrigation: {irr_badge} "
                        f"\u00b7 \U0001f331 NPK: {npk_badge}"
                    )

                except Exception:
                    pass

                # Risk alerts (historical only)
                if not use_forecast:
                    try:
                        risk = assess_state_risk(str(selected_state))
                        st.markdown("#### Weather Risk Assessment")
                        st.markdown(
                            _risk_badge_html(
                                risk.overall_risk,
                                f"{risk.overall_icon} Overall: {risk.overall_risk}",
                            ),
                            unsafe_allow_html=True,
                        )
                        if risk.alerts:
                            for alert in risk.alerts:
                                color_map = {
                                    "CRITICAL": "error",
                                    "HIGH": "error",
                                    "MODERATE": "warning",
                                }
                                fn = getattr(
                                    st, color_map.get(alert.level, "info")
                                )
                                fn(
                                    f"{alert.icon} **{alert.title}** "
                                    f"({alert.level})\n\n{alert.message}"
                                )
                        else:
                            st.success(
                                f"{risk.overall_icon} All indicators within normal range."
                            )
                    except Exception:
                        pass

            # Input summary with Source column
            st.markdown("#### Input Summary")
            summary_rows = []
            for col_name in feature_cols:
                val = user_inputs.get(col_name, "")
                display_val = friendly_name(col_name)

                # Determine source label
                if col_name == IRRIGATION_FEATURE:
                    src = (
                        _source_badge_user()
                        if irrigation_source == "user"
                        else _source_badge_state()
                    )
                elif col_name == FERTILIZER_FEATURE:
                    src = (
                        _source_badge_user()
                        if npk_source == "user"
                        else _source_badge_state()
                    )
                elif col_name == "area":
                    src = _source_badge_user()
                elif col_name in WEATHER_FEATURES:
                    src = (
                        "\U0001f52e Forecast" if use_forecast else "\U0001f570\ufe0f Historical"
                    )
                elif col_name in ("crop", "state", "season", "crop_year"):
                    src = _source_badge_user()
                else:
                    src = "\u2014"

                # Format value consistently
                if isinstance(val, float):
                    if abs(val - round(val)) < 0.05:
                        fmt_val = f"{int(round(val))}"
                    else:
                        fmt_val = f"{val:.1f}"
                else:
                    fmt_val = str(val)

                summary_rows.append(
                    {"Feature": display_val, "Value": fmt_val, "Source": src}
                )

            st.dataframe(
                pd.DataFrame(summary_rows), width="stretch", hide_index=True
            )
        else:
            st.error("Prediction failed.")
    else:
        st.info(
            "Configure your farm details in the sidebar and "
            "click **\U0001f52e Predict Yield**."
        )
        st.markdown(
            """
            **How it works:**
            1. Select **weather source**: Historical (NASA POWER) or Forecast (Open-Meteo)
            2. Choose your **crop**, **state**, **season**, and **year**
            3. Enter your **farm area** in hectares
            4. Optionally **override** irrigation and NPK with your own values
            5. Click **Predict Yield** to see your personalized forecast
            """
        )


# =====================  TAB 2 — SHAP  ====================================
with tab_shap:
    st.markdown("#### Model Explainability (SHAP)")
    st.caption("Feature contributions to the prediction.")

    if not SHAP_OK:
        st.warning("SHAP unavailable.")
    elif not predict_btn:
        st.info("Run a prediction first.")
    else:
        input_df = pd.DataFrame([user_inputs], columns=feature_cols)
        _, X_arr = predict_yield(
            booster, preprocessor, input_df, output_feature_names
        )
        if X_arr is not None:
            try:
                if isinstance(explainer, shap.KernelExplainer):
                    shap_values = explainer.shap_values(X_arr, nsamples=200)
                else:
                    try:
                        dmat = xgb.DMatrix(
                            X_arr, feature_names=output_feature_names or None
                        )
                        shap_values = explainer.shap_values(dmat)
                    except Exception:
                        shap_values = explainer.shap_values(X_arr)
                shap_vals = np.asarray(shap_values)
                shap_vals_single = (
                    shap_vals[0] if shap_vals.ndim == 2 else shap_vals
                )
                expected = explainer.expected_value
                expected = (
                    float(expected[0])
                    if isinstance(expected, np.ndarray)
                    else float(expected)
                )

                col_w, col_b = st.columns(2)
                with col_w:
                    st.markdown("##### Waterfall")
                    try:
                        explanation = shap.Explanation(
                            values=shap_vals_single,
                            base_values=expected,
                            data=X_arr[0],
                            feature_names=output_feature_names,
                        )
                        fig_wf, _ = plt.subplots(
                            figsize=(
                                8,
                                max(4, len(output_feature_names) * 0.35),
                            )
                        )
                        shap.plots.waterfall(
                            explanation,
                            max_display=min(25, len(output_feature_names)),
                            show=False,
                        )
                        st.pyplot(fig_wf)
                        plt.close("all")
                    except Exception as exc:
                        st.warning(f"Waterfall unavailable: {exc}")

                with col_b:
                    st.markdown("##### Feature Importance (|SHAP|)")
                    try:
                        importance = (
                            pd.DataFrame(
                                {
                                    "feature": output_feature_names,
                                    "importance": np.abs(shap_vals_single),
                                }
                            )
                            .sort_values("importance", ascending=True)
                            .tail(25)
                        )
                        fig_bar = px.bar(
                            importance,
                            x="importance",
                            y="feature",
                            orientation="h",
                            color="importance",
                            color_continuous_scale=[
                                "#1b5e20",
                                "#66bb6a",
                                "#f9a825",
                            ],
                        )
                        fig_bar.update_layout(
                            height=max(300, len(importance) * 28),
                            margin=dict(l=10, r=10, t=10, b=10),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font={
                                "family": "Source Serif 4, serif",
                                "color": "#c8e6c9",
                            },
                            coloraxis_showscale=False,
                            xaxis_title="|SHAP value|",
                            yaxis_title="",
                        )
                        fig_bar.update_xaxes(gridcolor="#1e3a1e")
                        fig_bar.update_yaxes(gridcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig_bar, width="stretch")
                    except Exception as exc:
                        st.warning(f"Feature importance unavailable: {exc}")

                st.markdown("##### Prediction Breakdown")
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Baseline", f"{expected:.2f} q/ha")
                mc2.metric(
                    "SHAP contribution", f"{shap_vals_single.sum():+.2f} q/ha"
                )
                mc3.metric(
                    "Final Prediction",
                    f"{expected + shap_vals_single.sum():.2f} q/ha",
                )

                with st.expander("Raw SHAP values (top 25)"):
                    shap_df = pd.DataFrame(
                        {
                            "Feature": output_feature_names,
                            "SHAP Value": shap_vals_single,
                            "|SHAP|": np.abs(shap_vals_single),
                            "Direction": [
                                "Increases" if v > 0 else "Decreases"
                                for v in shap_vals_single
                            ],
                        }
                    ).sort_values("|SHAP|", ascending=False).head(25)
                    st.dataframe(
                        shap_df, width="stretch", hide_index=True
                    )
            except Exception as exc:
                st.error(f"SHAP failed: {exc}")


# =====================  TAB 3 — DATA INSIGHTS  ===========================
with tab_data:
    st.markdown("#### Dataset Insights (50K rows)")

    if not DATA_OK:
        st.warning("Dataset not found.")
    else:
        df = dataset.copy()

        if crop_col and crop_col in df.columns:
            all_crops = sorted(df[crop_col].dropna().unique().tolist())
            crop_filter = st.multiselect(
                "Filter Crops", options=all_crops, default=all_crops[:5]
            )
            if crop_filter:
                df = df[df[crop_col].isin(crop_filter)]

        col1, col2 = st.columns(2)
        with col1:
            if (
                crop_col
                and target_col
                and crop_col in df.columns
                and target_col in df.columns
            ):
                st.markdown("##### Average Yield by Crop")
                crop_yield = (
                    df.groupby(crop_col)[target_col]
                    .mean()
                    .sort_values(ascending=True)
                    .reset_index()
                )
                fig_crop = px.bar(
                    crop_yield,
                    x=target_col,
                    y=crop_col,
                    orientation="h",
                    color=target_col,
                    color_continuous_scale=[
                        "#1b5e20",
                        "#66bb6a",
                        "#f9a825",
                    ],
                )
                fig_crop.update_layout(
                    height=380,
                    margin=dict(l=10, r=10, t=30, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={
                        "family": "Source Serif 4, serif",
                        "color": "#c8e6c9",
                    },
                    coloraxis_showscale=False,
                    xaxis=dict(gridcolor="#1e3a1e", title="Yield (q/ha)"),
                    yaxis=dict(gridcolor="rgba(0,0,0,0)", title=""),
                )
                st.plotly_chart(fig_crop, width="stretch")

        with col2:
            if (
                state_col
                and target_col
                and state_col in df.columns
                and target_col in df.columns
            ):
                st.markdown("##### Top States by Average Yield")
                state_yield = (
                    df.groupby(state_col)[target_col]
                    .mean()
                    .nlargest(15)
                    .sort_values(ascending=True)
                    .reset_index()
                )
                fig_state = px.bar(
                    state_yield,
                    x=target_col,
                    y=state_col,
                    orientation="h",
                    color=target_col,
                    color_discrete_sequence=[
                        "#1b5e20",
                        "#66bb6a",
                        "#f9a825",
                    ],
                )
                fig_state.update_layout(
                    height=380,
                    margin=dict(l=10, r=10, t=30, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={
                        "family": "Source Serif 4, serif",
                        "color": "#c8e6c9",
                    },
                    coloraxis_showscale=False,
                    xaxis=dict(gridcolor="#1e3a1e", title="Yield (q/ha)"),
                    yaxis=dict(gridcolor="rgba(0,0,0,0)", title=""),
                )
                st.plotly_chart(fig_state, width="stretch")

        if "crop_year" in df.columns and target_col in df.columns:
            with st.expander("Yield Trend Over Years"):
                yearly = df.groupby("crop_year")[target_col].mean().reset_index()
                fig_trend = px.line(
                    yearly,
                    x="crop_year",
                    y=target_col,
                    markers=True,
                    color_discrete_sequence=["#f9a825"],
                )
                fig_trend.update_layout(
                    height=300,
                    margin=dict(l=10, r=10, t=30, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={
                        "family": "Source Serif 4, serif",
                        "color": "#c8e6c9",
                    },
                    xaxis=dict(gridcolor="#1e3a1e", title="Year"),
                    yaxis=dict(gridcolor="#1e3a1e", title="Mean Yield (q/ha)"),
                )
                st.plotly_chart(fig_trend, width="stretch")

        if target_col and target_col in df.columns:
            with st.expander("Yield Distribution"):
                fig_hist = px.histogram(
                    df,
                    x=target_col,
                    nbins=50,
                    color=(
                        crop_col
                        if crop_col and crop_col in df.columns
                        else None
                    ),
                    opacity=0.7,
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig_hist.update_layout(
                    height=350,
                    margin=dict(l=10, r=10, t=30, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={
                        "family": "Source Serif 4, serif",
                        "color": "#c8e6c9",
                    },
                    xaxis=dict(gridcolor="#1e3a1e", title="Yield (q/ha)"),
                    yaxis=dict(gridcolor="#1e3a1e", title="Count"),
                    bargap=0.05,
                )
                st.plotly_chart(fig_hist, width="stretch")

        if all(c in df.columns for c in WEATHER_FEATURES):
            with st.expander("Weather vs Yield (NASA POWER)"):
                wx_labels = {
                    "mean_temperature": "Avg Temp (\u00b0C)",
                    "total_precipitation": "Rainfall (mm/yr)",
                    "mean_relative_humidity": "Humidity (%)",
                    "mean_solar_radiation": "Solar Rad (MJ/m\u00b2/day)",
                }
                available_wx = [
                    f for f in WEATHER_FEATURES if df[f].notna().any()
                ]
                if available_wx and target_col in df.columns:
                    wx_col = st.selectbox(
                        "Weather Feature",
                        options=available_wx,
                        format_func=lambda x: wx_labels.get(x, x),
                    )
                    fig_wx = px.scatter(
                        df,
                        x=wx_col,
                        y=target_col,
                        color=(
                            crop_col if crop_col in df.columns else None
                        ),
                        opacity=0.5,
                        color_discrete_sequence=px.colors.qualitative.Set2,
                        labels={
                            wx_col: wx_labels.get(wx_col, wx_col),
                            target_col: "Yield (q/ha)",
                        },
                    )
                    fig_wx.update_layout(
                        height=400,
                        margin=dict(l=10, r=10, t=30, b=10),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font={
                            "family": "Source Serif 4, serif",
                            "color": "#c8e6c9",
                        },
                        xaxis=dict(gridcolor="#1e3a1e"),
                        yaxis=dict(gridcolor="#1e3a1e"),
                        legend=dict(bgcolor="rgba(0,0,0,0)"),
                    )
                    st.plotly_chart(fig_wx, width="stretch")

        if IRRIGATION_FEATURE in df.columns:
            with st.expander("Irrigation Coverage"):
                if state_col in df.columns:
                    irr_by = (
                        df.groupby(state_col)[IRRIGATION_FEATURE]
                        .first()
                        .sort_values(ascending=True)
                        .reset_index()
                    )
                    fig_irr = px.bar(
                        irr_by,
                        x=IRRIGATION_FEATURE,
                        y=state_col,
                        orientation="h",
                        color=IRRIGATION_FEATURE,
                        color_continuous_scale=[
                            "#f44336",
                            "#ffc107",
                            "#4caf50",
                        ],
                    )
                    fig_irr.update_layout(
                        height=max(300, len(irr_by) * 22),
                        margin=dict(l=10, r=10, t=30, b=10),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font={
                            "family": "Source Serif 4, serif",
                            "color": "#c8e6c9",
                        },
                        coloraxis_showscale=False,
                        xaxis=dict(
                            gridcolor="#1e3a1e",
                            title="Irrigation (%)",
                            range=[0, 100],
                        ),
                        yaxis=dict(gridcolor="rgba(0,0,0,0)", title=""),
                    )
                    st.plotly_chart(fig_irr, width="stretch")

        if FERTILIZER_FEATURE in df.columns:
            with st.expander("NPK Fertilizer Consumption"):
                if state_col in df.columns:
                    fert_by = (
                        df.groupby(state_col)[FERTILIZER_FEATURE]
                        .first()
                        .sort_values(ascending=True)
                        .reset_index()
                    )
                    fig_fert = px.bar(
                        fert_by,
                        x=FERTILIZER_FEATURE,
                        y=state_col,
                        orientation="h",
                        color=FERTILIZER_FEATURE,
                        color_continuous_scale=[
                            "#f44336",
                            "#ffc107",
                            "#4caf50",
                        ],
                    )
                    fig_fert.update_layout(
                        height=max(300, len(fert_by) * 22),
                        margin=dict(l=10, r=10, t=30, b=10),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font={
                            "family": "Source Serif 4, serif",
                            "color": "#c8e6c9",
                        },
                        coloraxis_showscale=False,
                        xaxis=dict(gridcolor="#1e3a1e", title="NPK (kg/ha)"),
                        yaxis=dict(gridcolor="rgba(0,0,0,0)", title=""),
                    )
                    st.plotly_chart(fig_fert, width="stretch")

        with st.expander(
            "Weather Risk Alerts \u2014 All States", expanded=False
        ):
            try:
                all_assessments = assess_all_states()
                risk_counts: Dict[str, int] = {}
                for a in all_assessments:
                    risk_counts[a.overall_risk] = (
                        risk_counts.get(a.overall_risk, 0) + 1
                    )
                rc1, rc2, rc3, rc4 = st.columns(4)
                rc1.metric(
                    "Critical/High",
                    risk_counts.get("CRITICAL", 0) + risk_counts.get("HIGH", 0),
                )
                rc2.metric("Moderate", risk_counts.get("MODERATE", 0))
                rc3.metric(
                    "Low/Optimal",
                    risk_counts.get("LOW", 0) + risk_counts.get("OPTIMAL", 0),
                )
                rc4.metric("States", len(all_assessments))
                risk_rows = []
                for a in all_assessments:
                    alerts_str = (
                        " | ".join(
                            f"{al.icon} {al.title}" for al in a.alerts
                        )
                        if a.alerts
                        else "\U0001f7e2 All clear"
                    )
                    risk_rows.append(
                        {
                            "State": a.state,
                            "Risk": f"{a.overall_icon} {a.overall_risk}",
                            "Temp": f"{a.weather.get('mean_temperature', 0):.1f}",
                            "Rain": f"{a.weather.get('total_precipitation', 0):.0f}",
                            "Alerts": alerts_str,
                        }
                    )
                st.dataframe(
                    pd.DataFrame(risk_rows), width="stretch", hide_index=True
                )
            except Exception as exc:
                st.warning(f"Risk assessment unavailable: {exc}")

        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] >= 2:
            with st.expander("Feature Correlation Heatmap"):
                corr = numeric_df.corr()
                fig_corr, ax_corr = plt.subplots(figsize=(12, 8))
                sns.heatmap(
                    corr,
                    annot=True,
                    fmt=".2f",
                    cmap="YlGn",
                    ax=ax_corr,
                    linewidths=0.5,
                    cbar_kws={"shrink": 0.8},
                )
                ax_corr.set_title(
                    "Feature Correlations", fontsize=14, pad=12
                )
                plt.tight_layout()
                st.pyplot(fig_corr)
                plt.close("all")

        with st.expander("Raw Data Preview"):
            st.dataframe(df.head(200), width="stretch", hide_index=True)
            st.caption(f"Showing {min(200, len(df))} of {len(df)} rows")


# =====================  TAB 4 — MODEL INFO  ==============================
with tab_info:
    st.markdown("#### Model Performance & Metadata")

    if METRICS_OK and metrics:
        st.markdown(
            f"##### Evaluation Metrics "
            f"*(on {metrics['data_source']}, n={metrics['n_samples']})*"
        )
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("RMSE", f"{metrics['rmse']:.2f} q/ha")
        mc2.metric("R\u00b2 Score", f"{metrics['r2']:.4f}")
        mc3.metric("MAE", f"{metrics['mae']:.2f} q/ha")
        mc4.metric("MAPE", f"{metrics['mape']:.1f}%")

        per_crop_data = metrics.get("per_crop", {})
        if per_crop_data:
            st.markdown("##### Per-Crop Metrics")
            crop_rows = [
                {
                    "Crop": cn,
                    "N": cm["n"],
                    "RMSE": f"{cm['rmse']:.2f}" if cm["rmse"] else "\u2014",
                    "MAE": f"{cm['mae']:.2f}" if cm["mae"] else "\u2014",
                    "R\u00b2": f"{cm['r2']:.4f}" if cm["r2"] else "\u2014",
                }
                for cn, cm in per_crop_data.items()
            ]
            st.dataframe(
                pd.DataFrame(crop_rows), width="stretch", hide_index=True
            )

        col_sc, col_re = st.columns(2)
        with col_sc:
            st.markdown("##### Predicted vs Actual")
            fig_pvsa = go.Figure()
            fig_pvsa.add_trace(
                go.Scatter(
                    x=metrics["y_true"],
                    y=metrics["y_pred"],
                    mode="markers",
                    marker=dict(
                        color=(
                            np.array(metrics["y_pred"])
                            - np.array(metrics["y_true"])
                        ),
                        colorscale=["#f44336", "#ffeb3b", "#4caf50"],
                        size=5,
                        opacity=0.6,
                        colorbar=dict(title="Residual"),
                    ),
                    name="Predictions",
                )
            )
            all_vals = metrics["y_true"] + metrics["y_pred"]
            v_min, v_max = min(all_vals), max(all_vals)
            margin = (v_max - v_min) * 0.05
            fig_pvsa.add_trace(
                go.Scatter(
                    x=[v_min - margin, v_max + margin],
                    y=[v_min - margin, v_max + margin],
                    mode="lines",
                    line=dict(
                        color="#f9a825", dash="dash", width=1.5
                    ),
                    name="Perfect fit",
                )
            )
            fig_pvsa.update_layout(
                height=380,
                margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={
                    "family": "Source Serif 4, serif",
                    "color": "#c8e6c9",
                },
                xaxis=dict(gridcolor="#1e3a1e", title="Actual (q/ha)"),
                yaxis=dict(
                    gridcolor="#1e3a1e", title="Predicted (q/ha)"
                ),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_pvsa, width="stretch")

        with col_re:
            st.markdown("##### Residual Distribution")
            residuals = np.array(metrics["y_pred"]) - np.array(
                metrics["y_true"]
            )
            fig_resid = px.histogram(
                x=residuals,
                nbins=50,
                color_discrete_sequence=["#66bb6a"],
                opacity=0.8,
            )
            fig_resid.add_vline(
                x=0,
                line_dash="dash",
                line_color="#f9a825",
                line_width=1.5,
            )
            fig_resid.update_layout(
                height=380,
                margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={
                    "family": "Source Serif 4, serif",
                    "color": "#c8e6c9",
                },
                xaxis=dict(
                    gridcolor="#1e3a1e", title="Residual (q/ha)"
                ),
                yaxis=dict(gridcolor="#1e3a1e", title="Count"),
                bargap=0.05,
                showlegend=False,
            )
            st.plotly_chart(fig_resid, width="stretch")
    else:
        st.info("Metrics not available.")

    st.divider()
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Model", "XGBRegressor v3.2")
    with col_b:
        if HAS_XGB:
            st.metric("XGBoost", xgb.__version__)
    with col_c:
        if HAS_SHAP:
            st.metric("SHAP", shap.__version__)

    with st.expander("Data Sources"):
        st.markdown(
            """
            | Feature | Source | License |
            |---------|--------|---------|
            | Crop yields | Kaggle "Crop Production in India" (246K, sampled 50K) | CC0 |
            | Historical weather | NASA POWER API (year-specific, 538 cache) | Public Domain |
            | Seasonal forecast | Open-Meteo Seasonal Forecast (ECMWF SEAS5) | CC-BY 4.0 |
            | Irrigation | Agriculture Census 2015-16 | Govt. of India |
            | NPK Fertilizer | Dept. of Fertilizers, MoCF | Govt. of India |

            **Attribution:** Forecast data by
            [Open-Meteo.com](https://open-meteo.com/) (CC-BY 4.0)
            """
        )

    with st.expander("Feature Schema"):
        if schema:
            rows = []
            for col in feature_cols:
                is_wx = col in WEATHER_FEATURES
                is_irr = col == IRRIGATION_FEATURE
                is_fert = col == FERTILIZER_FEATURE
                rows.append(
                    {
                        "Feature": col,
                        "Display": friendly_name(col),
                        "Type": (
                            "categorical"
                            if col in cat_cols
                            else "weather"
                            if is_wx
                            else "irrigation"
                            if is_irr
                            else "fertilizer"
                            if is_fert
                            else "numerical"
                        ),
                        "Source": (
                            "NASA POWER / Open-Meteo"
                            if is_wx
                            else "Agri Census / user"
                            if is_irr
                            else "Dept. Fertilizers / user"
                            if is_fert
                            else "Dataset"
                        ),
                    }
                )
            st.dataframe(
                pd.DataFrame(rows), width="stretch", hide_index=True
            )

    with st.expander("Model File Inventory"):
        if MODELS_DIR.exists():
            file_data = [
                {
                    "File": f.name,
                    "Size (KB)": f"{f.stat().st_size / 1024:,.1f}",
                    "Format": f.suffix or "(none)",
                }
                for f in sorted(MODELS_DIR.iterdir())
                if f.is_file() and f.name != ".gitkeep"
            ]
            if file_data:
                st.dataframe(
                    pd.DataFrame(file_data), width="stretch", hide_index=True
                )

    with st.expander("Model Configuration"):
        if booster is not None:
            try:
                st.json(json_lib.loads(booster.save_config()))
            except Exception:
                st.info("Could not extract config.")


# =========================================================================
#  SECTION 8 — Footer
# =========================================================================

st.divider()
st.caption(
    "Crop Yield Prediction v3.2.1 \u00b7 50K district-level records "
    "\u00b7 Historical: NASA POWER \u00b7 Forecast: Open-Meteo (CC-BY 4.0) "
    "\u00b7 Inputs: Irrigation + NPK (overridable) "
    "\u00b7 Built with Streamlit, XGBoost, SHAP"
)