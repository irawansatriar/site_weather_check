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

# Forecast horizon in days for OpenWeatherMap (3-hour forecast path)
FORECAST_MAX_DAYS = 5

# Hourly data path options
USE_ONECALL_HOURLY = True
HOURLY_MAX_DAYS = 2  # up to 48 hours

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
# Calibration (sunny / rain thresholds)
# --------------------------

DEFAULT_CALIBRATION = {
    "sunny_max_clouds": 20,      # percent
    "sunny_max_pop": 0.10,       # 0-1
    "partly_max_clouds": 60,     # percent

    # Rain intensity by mm/h ( NOAA-style bands ):
    # light < 2.5, moderate 2.5–7.6, heavy > 7.6
    "light_min_mmph": 0.10,
    "moderate_min_mmph": 2.5,
    "heavy_min_mmph": 7.6,

    # Optional: if mm=0 but POP high and clouds high, treat as light rain
    "light_pop_fallback": 0.50,  # 0-1
}

def get_calibration() -> dict:
    cfg = {**DEFAULT_CALIBRATION, **st.session_state.get("calibration", {})}
    return cfg

def classify_weather(
    weather_id: Optional[int],
    weather_main: str,
    weather_description: str,
    precip_mm_per_h: float,
    clouds_pct: float,
    pop: Optional[float],
    calib: dict,
) -> str:
    pop = pop if pop is not None else 0.0

    # Thunderstorms / snow / fog first (by code or by category)
    if isinstance(weather_id, (int, float)):
        wid = int(weather_id)
        if 200 <= wid <= 232:
            return "thunderstorm"
        if 600 <= wid <= 622:
            # Snow intensities handled like rain thresholds
            if precip_mm_per_h >= calib["heavy_min_mmph"]:
                return "heavy snow"
            if precip_mm_per_h >= calib["moderate_min_mmph"]:
                return "moderate snow"
            if precip_mm_per_h >= calib["light_min_mmph"] or pop >= calib["light_pop_fallback"]:
                return "light snow"
            return "snow"
        if 701 <= wid <= 781:
            return "foggy"

    # Rain by mm/h
    if precip_mm_per_h >= calib["heavy_min_mmph"]:
        return "heavy rain"
    if precip_mm_per_h >= calib["moderate_min_mmph"]:
        return "moderate rain"
    if precip_mm_per_h >= calib["light_min_mmph"]:
        return "light rain"

    # Fallback to POP + clouds if mm=0
    if pop >= calib["light_pop_fallback"] and clouds_pct >= max(calib["partly_max_clouds"], 60):
        return "light rain"

    # Clouds → sky condition
    if clouds_pct <= calib["sunny_max_clouds"] and pop <= calib["sunny_max_pop"]:
        return "sunny"
    if clouds_pct <= calib["partly_max_clouds"]:
        return "partly cloudy"
    return "cloudy"

def classify_weather_row(row: pd.Series, calib: dict) -> str:
    return classify_weather(
        weather_id=row.get("weather_id"),
        weather_main=str(row.get("weather_main", "")),
        weather_description=str(row.get("weather_description", "")),
        precip_mm_per_h=float(row.get("precip_mm_per_h", row.get("precipitation", 0) or 0)),
        clouds_pct=float(row.get("cloudcover", 0) or 0),
        pop=float(row.get("pop", 0) or 0),
        calib=calib,
    )


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
            msg = data.get("message", f"HTTP cod {cod_str}")
            raise ValueError(f"OpenWeatherMap error: {msg}")
        return data
    except requests.exceptions.HTTPError as e:
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

    params = {"lat": lat, "lon": lon}
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
            weather_id = weather.get("id", None)

            rain_3h = float(item.get("rain", {}).get("3h", 0) or 0.0)
            snow_3h = float(item.get("snow", {}).get("3h", 0) or 0.0)
            precip_3h = rain_3h + snow_3h
            precip_mm_per_h = precip_3h / 3.0

            rows.append(
                {
                    "time": dt,
                    "date": dt.date(),
                    "hour_label": dt.strftime("%H:00"),
                    "weather_id": weather_id,
                    "weather_main": main_weather,
                    "weather_description": description,
                    "pop": item.get("pop", np.nan),                     # 0–1
                    "precipitation": precip_3h,                          # mm per 3h
                    "precip_mm_per_h": precip_mm_per_h,                  # mm/h (for classification)
                    "cloudcover": item.get("clouds", {}).get("all", 0),  # %
                    "temperature": item.get("main", {}).get("temp", np.nan),
                    "humidity": item.get("main", {}).get("humidity", np.nan),
                }
            )

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
            weather_id = weather.get("id", None)

            rain_1h = float(item.get("rain", {}).get("1h", 0) or 0.0)
            snow_1h = float(item.get("snow", {}).get("1h", 0) or 0.0)
            precip_1h = rain_1h + snow_1h

            rows.append(
                {
                    "time": dt,
                    "date": dt.date(),
                    "hour_label": dt.strftime("%H:00"),
                    "weather_id": weather_id,
                    "weather_main": main_weather,
                    "weather_description": description,
                    "pop": item.get("pop", np.nan),     # 0–1
                    "precipitation": precip_1h,         # mm per 1h
                    "precip_mm_per_h": precip_1h,       # mm/h
                    "cloudcover": item.get("clouds", 0),
                    "temperature": item.get("temp", np.nan),
                    "humidity": item.get("humidity", np.nan),
                }
            )
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

def build_single_site_pivot(site_row: pd.Series, start_d: date, end_d: date) -> Tuple[pd.DataFrame, pd.DataFrame]:
    lat = float(site_row["latitude"])
    lon = float(site_row["longitude"])

    df = fetch_openweathermap_range(lat, lon, start_d, end_d)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    calib = get_calibration()
    df = df.copy()
    # Classify each hour into a human-friendly label
    df["weather_text"] = df.apply(lambda r: classify_weather_row(r, calib), axis=1)

    raw_cols = [
        "time",
        "date",
        "hour_label",
        "weather_main",
        "weather_description",
        "weather_text",
        "precipitation",
        "precip_mm_per_h",
        "pop",
        "cloudcover",
        "temperature",
        "humidity",
    ]
    hourly_df = df[raw_cols].copy()

    pivot = hourly_df.pivot_table(
        index="hour_label",
        columns="date",
        values="weather_text",
        aggfunc="first",
        dropna=False,
    ).sort_index()

    return pivot, hourly_df


def process_batch_requests(requests_df: pd.DataFrame, sites_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    req = requests_df.copy()
    req.columns = [c.strip().lower() for c in req.columns]
    required = {"location", "start_date", "end_date"}
    missing = required - set(req.columns)
    if missing:
        raise ValueError(f"Requests CSV missing required columns: {', '.join(sorted(missing))}")

    req["start_date"] = pd.to_datetime(req["start_date"], errors="coerce").dt.date
    req["end_date"] = pd.to_datetime(req["end_date"], errors="coerce").dt.date

    sites_sub = sites_df[["location", "latitude", "longitude"]].copy()
    sites_sub["location_key"] = sites_sub["location"].str.strip().str.lower()
    req["location_key"] = req["location"].astype(str).str.strip().str.lower()
    merged = req.merge(sites_sub, how="left", on="location_key", suffixes=("", "_site"))
    merged["location"] = merged["location"].fillna(merged["location_site"]).fillna(merged["location_key"])
    merged = merged.drop(columns=["location_site"])

    results = []
    failures = []

    for _, row in merged.iterrows():
        loc = str(row["location"]).strip()
        lat = row.get("latitude", np.nan)
        lon = row.get("longitude", np.nan)
        sd = row.get("start_date", None)
        ed = row.get("end_date", None)

        if pd.isna(lat) or pd.isna(lon):
            failures.append(
                {"location": loc, "start_date": sd, "end_date": ed, "error": "Location not found in sites.csv"}
            )
            continue
        if pd.isna(sd) or pd.isna(ed):
            failures.append(
                {"location": loc, "start_date": sd, "end_date": ed, "error": "Invalid start/end date"}
            )
            continue

        try:
            sd_v, ed_v = validate_date_range(to_date(sd), to_date(ed))
            df = fetch_openweathermap_range(float(lat), float(lon), sd_v, ed_v)
            if df.empty:
                failures.append(
                    {"location": loc, "start_date": sd_v, "end_date": ed_v, "error": "No data returned for range"}
                )
                continue

            df_out = df.copy()
            df_out.insert(0, "location", loc)
            df_out = df_out[
                [
                    "location",
                    "time",
                    "date",
                    "hour_label",
                    "weather_main",
                    "weather_description",
                    "weather_text",
                    "precipitation",
                    "precip_mm_per_h",
                    "pop",
                    "cloudcover",
                    "temperature",
                    "humidity",
                ]
            ]
            # Reclassify for batch results as well
            calib = get_calibration()
            df_out["weather_text"] = df_out.apply(lambda r: classify_weather_row(r, calib), axis=1)

            results.append(df_out)

        except Exception as e:
            failures.append({"location": loc, "start_date": sd, "end_date": ed, "error": str(e)})

    res_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    fail_df = pd.DataFrame(failures) if failures else pd.DataFrame()
    return res_df, fail_df


def download_csv_button(df: pd.DataFrame, label: str, file_name: str, help_text: Optional[str] = None):
    if df is None or df.empty:
        st.info("No data to download.")
        return
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=file_name,
        mime="text/csv",
        help=help_text,
    )


# --------------------------
# UI: Streamlit App
# --------------------------

def render_calibration_ui():
    st.subheader("Calibration")
    cfg = get_calibration()
    c1, c2 = st.columns(2)
    with c1:
        sunny_max_clouds = st.slider("Sunny: max clouds (%)", 0, 100, cfg["sunny_max_clouds"])
        partly_max_clouds = st.slider("Partly: max clouds (%)", 0, 100, cfg["partly_max_clouds"])
        sunny_max_pop = st.slider("Sunny: max POP", 0.0, 1.0, cfg["sunny_max_pop"], 0.01)
    with c2:
        light_min_mmph = st.slider("Light rain min (mm/h)", 0.0, 5.0, cfg["light_min_mmph"], 0.05)
        moderate_min_mmph = st.slider("Moderate rain min (mm/h)", 0.0, 10.0, cfg["moderate_min_mmph"], 0.05)
        heavy_min_mmph = st.slider("Heavy rain min (mm/h)", 0.0, 30.0, cfg["heavy_min_mmph"], 0.05)
        light_pop_fallback = st.slider("If POP ≥, treat as light rain (mm=0)", 0.0, 1.0, cfg["light_pop_fallback"], 0.05)

    if st.button("Apply calibration"):
        st.session_state["calibration"] = {
            "sunny_max_clouds": int(sunny_max_clouds),
            "sunny_max_pop": float(sunny_max_pop),
            "partly_max_clouds": int(partly_max_clouds),
            "light_min_mmph": float(light_min_mmph),
            "moderate_min_mmph": float(moderate_min_mmph),
            "heavy_min_mmph": float(heavy_min_mmph),
            "light_pop_fallback": float(light_pop_fallback),
        }
        st.success("Calibration saved. Rebuild to apply.")


def render_sites_editor():
    st.subheader("Sites master (sites.csv)")
    st.caption(
        "Manage your sites list. Required columns: location, latitude, longitude. "
        "Uploader accepts common aliases: name/site/place → location; lat → latitude; lon/lng/long → longitude."
    )

    uploaded = st.file_uploader(
        "Optionally upload a sites.csv (it will replace current file for this session)",
        type=["csv"],
        accept_multiple_files=False,
        key="sites_uploader",
    )
    if uploaded is not None:
        try:
            df_new = pd.read_csv(uploaded, sep=None, engine="python")
            df_norm, notes = normalize_sites_columns(df_new)
            save_sites_csv(df_norm, SITES_CSV)
            st.success("sites.csv replaced from upload.")
            if notes:
                for n in notes:
                    st.info(n)
            st.cache_data.clear()
        except Exception as e:
            st.error(f"Failed to ingest uploaded sites.csv: {e}")

    try:
        sites_df = load_sites_csv(SITES_CSV)
    except Exception as e:
        st.error(f"Failed to load sites.csv: {e}")
        st.stop()

    st.write("Edit your sites below, then click Save.")
    edited_df = st.data_editor(
        sites_df,
        num_rows="dynamic",
        use_container_width=True,
        key="sites_editor",
        column_config={
            "location": st.column_config.TextColumn("location"),
            "latitude": st.column_config.NumberColumn("latitude", format="%.6f"),
            "longitude": st.column_config.NumberColumn("longitude", format="%.6f"),
        },
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Save sites.csv", type="primary"):
            try:
                save_sites_csv(edited_df, SITES_CSV)
                st.success("sites.csv saved.")
                st.cache_data.clear()
            except Exception as e:
                st.error(f"Failed to save sites.csv: {e}")
    with c2:
        st.download_button(
            "Download current sites.csv",
            data=edited_df.to_csv(index=False).encode("utf-8"),
            file_name="sites.csv",
            mime="text/csv",
        )


def render_single_site_ui(sites_df: pd.DataFrame):
    st.header("B. Single site selection")

    site_names = sites_df["location"].tolist()
    site_sel = st.selectbox("Select a site (from sites.csv)", site_names)

    today_d = date.today()
    default_end = min(today_d + timedelta(days=3), today_d + timedelta(days=FORECAST_MAX_DAYS))
    start_d = st.date_input("Start date", value=today_d, format="YYYY-MM-DD")
    end_d = st.date_input("End date", value=default_end, format="YYYY-MM-DD")

    submit = st.button("Build pivot", type="primary")

    if submit:
        if OPENWEATHERMAP_API_KEY == "" or OPENWEATHERMAP_API_KEY == "your_api_key_here":
            st.error("Please set your OpenWeatherMap API key in Streamlit secrets or environment.")
            st.info("Get a free API key at: https://openweathermap.org/api")
            return

        site_row = sites_df.loc[sites_df["location"] == site_sel].iloc[0]
        try:
            sd, ed = validate_date_range(to_date(start_d), to_date(end_d))
            with st.spinner("Fetching data..."):
                pivot_df, hourly_df = build_single_site_pivot(site_row, sd, ed)

            if hourly_df.empty:
                st.warning("No data returned for the requested range.")
                return

            st.subheader("Pivot (rows: hour 00:00–23:00, columns: date, values: weather)")
            st.dataframe(pivot_df, use_container_width=True)
            st.caption("Note: values show simplified weather descriptions per hour.")

            c1, c2 = st.columns(2)
            with c1:
                pivot_flat = pivot_df.copy()
                pivot_flat.index.name = "hour"
                pivot_flat.reset_index(inplace=True)
                download_csv_button(
                    pivot_flat,
                    "Download pivot CSV",
                    f"{site_sel}_pivot_{sd}_to_{ed}.csv",
                )
            with c2:
                download_csv_button(
                    hourly_df,
                    "Download raw hourly CSV",
                    f"{site_sel}_hourly_{sd}_to_{ed}.csv",
                )

        except Exception as e:
            st.error(f"Failed to build pivot: {e}")


def render_batch_ui(sites_df: pd.DataFrame):
    st.header("A. Batch: upload requests CSV")
    st.caption(
        "Upload a CSV with columns: location,start_date,end_date. "
        "Locations will be matched to sites.csv to retrieve coordinates."
    )

    with st.expander("Download template", expanded=False):
        st.code(REQUESTS_TEMPLATE_CSV, language="csv")
        st.download_button(
            "Download requests template CSV",
            data=REQUESTS_TEMPLATE_CSV.encode("utf-8"),
            file_name="requests_template.csv",
            mime="text/csv",
        )

    up = st.file_uploader("Upload requests CSV", type=["csv"], accept_multiple_files=False, key="req_csv")
    if up is not None:
        try:
            req_df = pd.read_csv(up)
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")
            return

        if st.button("Run batch", type="primary"):
            try:
                with st.spinner("Processing batch..."):
                    res_df, fail_df = process_batch_requests(req_df, sites_df)

                if not res_df.empty:
                    st.subheader("Combined results (hourly)")
                    st.dataframe(res_df.head(200), use_container_width=True)
                    st.caption("Showing first 200 rows. Use download for full data.")
                    download_csv_button(res_df, "Download all results CSV", "batch_hourly_results.csv")

                if not fail_df.empty:
                    st.subheader("Failed rows")
                    st.dataframe(fail_df, use_container_width=True)
                    download_csv_button(fail_df, "Download failures CSV", "batch_failures.csv")

                if res_df.empty and fail_df.empty:
                    st.info("No output produced.")

            except Exception as e:
                st.error(f"Batch processing failed: {e}")


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    # Calibration UI (in sidebar)
    with st.sidebar:
        render_calibration_ui()
        st.header("Sites master")
        render_sites_editor()

    # Quick API key sanity
    if not OPENWEATHERMAP_API_KEY:
        st.warning("⚠️ OpenWeatherMap API key not configured. Please set it in Streamlit secrets.")
        st.info("Get a free API key at: https://openweathermap.org/api")

    st.write(
        "- Time format: 24-hour (HH:00)\n"
        "- Mode A: upload requests CSV with location,start_date,end_date (dates in YYYY-MM-DD)\n"
        "- Mode B: select a single site and date range to see a pivot (hour x date → weather)\n"
        "- Uses OpenWeatherMap API (requires free API key)\n"
        "- Edit sites.csv in-app or upload your own"
    )

    try:
        sites_df = load_sites_csv(SITES_CSV)
    except Exception as e:
        st.error(f"Cannot proceed: {e}")
        st.stop()

    mode = st.radio("Choose mode", ["B. Single site selection", "A. Upload batch CSV"], index=0, horizontal=True)

    if mode.startswith("B"):
        render_single_site_ui(sites_df)
    else:
        render_batch_ui(sites_df)

    st.divider()
    with st.expander("Help / Tips"):
        st.markdown(
            """
            - API Key: Set OPENWEATHERMAP_API_KEY in Streamlit secrets or environment.
            - Rate Limits: OpenWeatherMap free tier allows ~1,000 calls/day.
            - Date Range: With One Call hourly data, you can query up to ~48 hours. Otherwise, uses 3-hour forecast for longer ranges.
            - File Uploads: Accept aliases for column names (name/site → location, lat → latitude, lon/lng/long → longitude).
            """
        )


if __name__ == "__main__":
    main()
