# streamlit_app_custom_with_editor_openweather.py

import os
from datetime import date, datetime, timedelta
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import requests
import streamlit as st

# --------------------------
# App Config and Constants
# --------------------------

APP_TITLE = "Site Weather Check (OpenWeatherMap)"
SITES_CSV = "sites.csv"

# OpenWeatherMap API configuration
OPENWEATHERMAP_API_KEY = (
    st.secrets.get("OPENWEATHERMAP_API_KEY")
    or os.environ.get("OPENWEATHERMAP_API_KEY")
    or ""
)
FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
ONECALL_URL = "https://api.openweathermap.org/data/3.0/onecall"

# Forecast horizon in days for OpenWeatherMap (for the 3-hour forecast path)
FORECAST_MAX_DAYS = 5

# If you want to prefer hourly data, flip this to True
USE_ONECALL_HOURLY = True
# Maximum days for hourly data (2 days -> 48 hours)
HOURLY_MAX_DAYS = 2

REQUESTS_TEMPLATE_CSV = """location,start_date,end_date
Site A,2025-01-01,2025-01-03
Site B,2025-01-05,2025-01-06
"""

SITES_TEMPLATE_CSV = """location,latitude,longitude
Kuala Lumpur,3.1390,101.6869
Singapore,1.3521,103.8198
Jakarta,-6.2088,106.8456
"""

USER_AGENT = "site-weather-check/1.0 (+https://github.com/your-org/your-repo)"


# --------------------------
# Utility Functions
# --------------------------

def ensure_sites_csv_exists(path: str = SITES_CSV) -> None:
    """Ensure a default sites.csv exists so the app is usable on first launch."""
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(SITES_TEMPLATE_CSV)


@st.cache_data(show_spinner=False)
def load_sites_csv(path: str = SITES_CSV) -> pd.DataFrame:
    ensure_sites_csv_exists(path)
    df = pd.read_csv(path, dtype={"location": str}, encoding="utf-8").dropna(how="all")
    required = {"location", "latitude", "longitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"sites.csv missing required columns: {', '.join(sorted(missing))}")
    df["location"] = df["location"].astype(str).str.strip()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["location", "latitude", "longitude"])
    return df.reset_index(drop=True)


def save_sites_csv(df: pd.DataFrame, path: str = SITES_CSV) -> None:
    cols_required = ["location", "latitude", "longitude"]
    missing = [c for c in cols_required if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot save sites.csv, missing columns: {missing}")
    df = df.copy()
    df["location"] = df["location"].astype(str).str.strip()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["location", "latitude", "longitude"])
    df.to_csv(path, index=False, encoding="utf-8")


def normalize_sites_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Accept flexible input headers and normalize to: location, latitude, longitude."""
    notes: List[str] = []
    orig_cols = list(df.columns)
    clean_cols = [str(c).replace("\ufeff", "").strip() for c in orig_cols]
    lower_cols = [c.lower() for c in clean_cols]

    alias_map = {
        "location": {"location", "site", "name", "place"},
        "latitude": {"latitude", "lat"},
        "longitude": {"longitude", "lon", "lng", "long"},
    }

    rename_dict = {}
    used_targets = set()
    for i, lc in enumerate(lower_cols):
        target = None
        for key, aliases in alias_map.items():
            if lc in aliases and key not in used_targets:
                target = key
                used_targets.add(key)
                break
        rename_dict[orig_cols[i]] = target if target else lc

    df = df.rename(columns=rename_dict)
    required = {"location", "latitude", "longitude"}
    if not required.issubset(df.columns):
        found = ", ".join(df.columns)
        raise ValueError(
            "Could not find required columns after normalization. "
            f"Found columns: {found}. Expected: location, latitude, longitude."
        )

    df["location"] = df["location"].astype(str).str.strip()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["location", "latitude", "longitude"])
    dropped = before - len(df)
    if dropped:
        notes.append(f"Dropped {dropped} rows with missing/invalid location/lat/lon.")

    return df, notes


def to_date(obj) -> date:
    if isinstance(obj, date) and not isinstance(obj, datetime):
        return obj
    if isinstance(obj, datetime):
        return obj.date()
    if isinstance(obj, str):
        return datetime.fromisoformat(obj).date()
    raise ValueError(f"Unsupported date-like: {obj}")


def validate_date_range(start: date, end: date) -> Tuple[date, date]:
    if end < start:
        raise ValueError("End date must be on or after start date.")
    earliest = date.today() - timedelta(days=5)  # OpenWeatherMap has limited historical data
    latest = date.today() + timedelta(days=FORECAST_MAX_DAYS)
    s = max(start, earliest)
    e = min(end, latest)
    return s, e


def owm_weather_to_text(weather_main: str, weather_description: str) -> str:
    """Convert OpenWeatherMap weather info to simplified text."""
    if not weather_main:
        return "unknown"
    main_lower = weather_main.lower()
    if main_lower == "clear":
        return "sunny"
    if main_lower == "clouds":
        return "cloudy"
    if main_lower == "rain":
        return "rainy"
    if main_lower == "drizzle":
        return "light rain"
    if main_lower == "thunderstorm":
        return "thunderstorm"
    if main_lower == "snow":
        return "snowy"
    if main_lower in {"mist", "fog", "haze"}:
        return "foggy"
    return weather_description.lower() if weather_description else main_lower


# --------------------------
# OpenWeatherMap Fetching
# --------------------------

def _request_openweathermap(url: str, params: dict) -> dict:
    try:
        params["appid"] = OPENWEATHERMAP_API_KEY
        params["units"] = "metric"
        r = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=30)
        r.raise_for_status()
        data = r.json()
        cod = data.get("cod", 200)
        cod_str = str(cod)
        if cod_str != "200":
            # OWM returns error info in 'message'
            msg = data.get("message", f"HTTP cod {cod_str}")
            raise ValueError(f"OpenWeatherMap error: {msg}")
        return data
    except requests.exceptions.HTTPError as e:
        # Scrub API key from any logged URL
        safe_url = ""
        try:
            req_url = r.request.url or ""
            if OPENWEATHERMAP_API_KEY:
                safe_url = req_url.replace(OPENWEATHERMAP_API_KEY, "****")
            else:
                safe_url = req_url
        except Exception:
            pass
        body = ""
        try:
            body = r.text[:500]
        except Exception:
            pass
        status = getattr(e.response, "status_code", "HTTP")
        reason = getattr(e.response, "reason", "Error")
        raise RuntimeError(f"OpenWeatherMap HTTP error: {status} {reason} for url: {safe_url} | response: {body}") from e
    except Exception as e:
        raise RuntimeError(f"OpenWeatherMap fetch failed: {e}") from e


@st.cache_data(show_spinner=False)
def fetch_openweathermap_forecast(lat: float, lon: float, start_d: date, end_d: date) -> pd.DataFrame:
    """Fallback 3-hour forecast data (for longer ranges)."""
    today = date.today()
    if start_d > today + timedelta(days=FORECAST_MAX_DAYS):
        return pd.DataFrame()

    params = {
        "lat": lat,
        "lon": lon,
    }

    data = _request_openweathermap(FORECAST_URL, params)

    forecast_list = data.get("list", [])
    if not forecast_list:
        return pd.DataFrame()

    rows = []
    for item in forecast_list:
        dt = datetime.fromtimestamp(item["dt"])
        if start_d <= dt.date() <= end_d:
            weather = item.get("weather", [{}])[0]
            main_weather = weather.get("main", "")
            description = weather.get("description", "")
            precip = item.get("rain", {}).get("3h", 0) + item.get("snow", {}).get("3h", 0)
            rows.append({
                "time": dt,
                "date": dt.date(),
                "hour_label": dt.strftime("%H:00"),
                "weather_main": main_weather,
                "weather_description": description,
                "weather_text": owm_weather_to_text(main_weather, description),
                "precipitation": precip,
                "cloudcover": item.get("clouds", {}).get("all", 0),
                "temperature": item.get("main", {}).get("temp", np.nan),
                "humidity": item.get("main", {}).get("humidity", np.nan),
            })

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def fetch_openweathermap_hourly(lat: float, lon: float, start_d: date, end_d: date) -> pd.DataFrame:
    """Hourly data via OpenWeatherMap One Call API (1-hour steps)."""
    today = date.today()
    if start_d > today + timedelta(days=HOURLY_MAX_DAYS):
        return pd.DataFrame()

    data = _request_openweathermap(
        ONECALL_URL,
        {
            "lat": lat,
            "lon": lon,
            "exclude": "current,minutely,daily,alerts",
            "units": "metric",
        }
    )

    hourly_list = data.get("hourly", [])
    if not hourly_list:
        return pd.DataFrame()

    rows = []
    for item in hourly_list:
        dt = datetime.fromtimestamp(item["dt"])
        if start_d <= dt.date() <= end_d:
            weather = item.get("weather", [{}])[0]
            main_weather = weather.get("main", "")
            description = weather.get("description", "")
            precip = float(item.get("rain", {}).get("1h", 0)) + float(item.get("snow", {}).get("1h", 0))
            rows.append({
                "time": dt,
                "date": dt.date(),
                "hour_label": dt.strftime("%H:00"),
                "weather_main": main_weather,
                "weather_description": description,
                "weather_text": owm_weather_to_text(main_weather, description),
                "precipitation": precip,
                "cloudcover": item.get("clouds", 0),
                "temperature": item.get("temp", np.nan),
                "humidity": item.get("humidity", np.nan),
            })
    return pd.DataFrame(rows)


def fetch_openweathermap_range(lat: float, lon: float, start_d: date, end_d: date) -> pd.DataFrame:
    """Choose hourly data if within hourly window, otherwise fallback to 3-hour forecast."""
    start_d, end_d = validate_date_range(start_d, end_d)
    if USE_ONECALL_HOURLY:
        days_span = (end_d - start_d).days + 1
        if days_span <= HOURLY_MAX_DAYS:
            df = fetch_openweathermap_hourly(lat, lon, start_d, end_d)
            if not df.empty:
                return df
    return fetch_openweathermap_forecast(lat, lon, start_d, end_d)


# --------------------------
# App Logic
# --------------------------
# (The rest of your app logic remains the same, using fetch_openweathermap_range
#  to obtain data and build pivots as before.)
# --------------------------
