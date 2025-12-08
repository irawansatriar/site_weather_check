# streamlit_app_custom_with_editor.py

import io
import os
from datetime import date, datetime, timedelta
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import pytz
import requests
import streamlit as st

# --------------------------
# App Config and Constants
# --------------------------

APP_TITLE = "Site Weather Check (Custom)"
SITES_CSV = "sites.csv"

# Open-Meteo endpoints
FORECAST_API = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"

# Variables requested from Open-Meteo (use ones supported by both endpoints)
HOURLY_VARS = [
    "weathercode",
    "precipitation",
    "cloudcover",
]

# Forecast horizon in days for the standard forecast endpoint
FORECAST_MAX_DAYS = 16

REQUESTS_TEMPLATE_CSV = """location,start_date,end_date
Site A,2025-01-01,2025-01-03
Site B,2025-01-05,2025-01-06
"""

SITES_TEMPLATE_CSV = """location,latitude,longitude,timezone
Kuala Lumpur,3.1390,101.6869,Asia/Kuala_Lumpur
Singapore,1.3521,103.8198,Asia/Singapore
Jakarta,-6.2088,106.8456,Asia/Jakarta
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
    # timezone is optional now; default to 'auto' if missing
    required = {"location", "latitude", "longitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"sites.csv missing required columns: {', '.join(sorted(missing))}")
    if "timezone" not in df.columns:
        df["timezone"] = "auto"

    # Normalize column types
    df["location"] = df["location"].astype(str).str.strip()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["timezone"] = df["timezone"].astype(str).str.strip()
    df = df.dropna(subset=["location", "latitude", "longitude"])
    return df.reset_index(drop=True)


def save_sites_csv(df: pd.DataFrame, path: str = SITES_CSV) -> None:
    cols_required = ["location", "latitude", "longitude"]
    missing = [c for c in cols_required if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot save sites.csv, missing columns: {missing}")
    df = df.copy()
    if "timezone" not in df.columns:
        df["timezone"] = "auto"
    df["location"] = df["location"].astype(str).str.strip()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["timezone"] = df["timezone"].astype(str).str.strip()
    # Drop invalid rows
    df = df.dropna(subset=["location", "latitude", "longitude"])
    df.to_csv(path, index=False, encoding="utf-8")


def coerce_tz_or_auto(tz: Optional[str]) -> str:
    """Return a valid timezone string for Open-Meteo. If invalid, return 'auto'."""
    if tz and tz in pytz.all_timezones:
        return tz
    return "auto"


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
    # Keep within a reasonable hard maximum (2 years back and 1 year forward)
    earliest = date.today() - timedelta(days=365 * 2)
    latest = date.today() + timedelta(days=365)
    s = max(start, earliest)
    e = min(end, latest)
    return s, e


def weathercode_to_text(code: int) -> str:
    mapping = {
        0: "sunny",
        1: "mainly clear",
        2: "partly cloudy",
        3: "overcast",
        45: "fog",
        48: "depositing rime fog",
        51: "light drizzle",
        53: "moderate drizzle",
        55: "dense drizzle",
        56: "light freezing drizzle",
        57: "dense freezing drizzle",
        61: "light rain",
        63: "moderate rain",
        65: "heavy rain",
        66: "light freezing rain",
        67: "heavy freezing rain",
        71: "light snow",
        73: "moderate snow",
        75: "heavy snow",
        77: "snow grains",
        80: "light rain showers",
        81: "moderate rain showers",
        82: "violent rain showers",
        85: "light snow showers",
        86: "heavy snow showers",
        95: "thunderstorm",
        96: "thunderstorm with light hail",
        99: "thunderstorm with heavy hail",
    }
    return mapping.get(int(code), f"code {code}")


# --------------------------
# CSV Normalization Helpers
# --------------------------

def normalize_sites_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Accept flexible input headers and normalize to: location, latitude, longitude, timezone.
    - Handles common aliases (name/site/place -> location; lat -> latitude; lon/lng/long -> longitude; tz/time_zone -> timezone)
    - Strips whitespace and BOM
    - Auto-fills timezone='auto' if missing
    """
    notes: List[str] = []
    # Remove BOM in headers if present and strip whitespace
    orig_cols = list(df.columns)
    clean_cols = [str(c).replace("\ufeff", "").strip() for c in orig_cols]
    lower_cols = [c.lower() for c in clean_cols]

    alias_map = {
        "location": {"location", "site", "name", "place"},
        "latitude": {"latitude", "lat"},
        "longitude": {"longitude", "lon", "lng", "long"},
        "timezone": {"timezone", "tz", "time_zone", "time zone"},
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
            f"Found columns: {found}. Expected at least: location, latitude, longitude. "
            "Accepted aliases: location(name/site/place), latitude(lat), longitude(lon/lng/long), timezone(tz/time_zone)."
        )

    if "timezone" not in df.columns:
        df["timezone"] = "auto"
        notes.append("timezone defaulted to 'auto' for all rows.")

    # Coerce types and clean
    df["location"] = df["location"].astype(str).str.strip()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["timezone"] = df["timezone"].astype(str).str.strip()

    before = len(df)
    df = df.dropna(subset=["location", "latitude", "longitude"])
    dropped = before - len(df)
    if dropped:
        notes.append(f"dropped {dropped} rows with missing/invalid location/lat/lon.")

    return df, notes


# --------------------------
# Open-Meteo Fetching
# --------------------------

def _request_open_meteo(url: str, params: dict) -> dict:
    try:
        r = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=30)
        r.raise_for_status()
        data = r.json()
        if "hourly" not in data or "time" not in data["hourly"]:
            raise ValueError("No hourly data returned by API.")
        return data
    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = f" | response: {r.text[:500]}"
        except Exception:
            pass
        raise RuntimeError(f"Open-Meteo HTTP error: {e}{detail}") from e
    except Exception as e:
        raise RuntimeError(f"Open-Meteo fetch failed: {e}") from e


@st.cache_data(show_spinner=False)
def fetch_open_meteo_forecast(lat: float, lon: float, start_d: date, end_d: date, tz: str) -> pd.DataFrame:
    today_utc = date.today()
    max_end = today_utc + timedelta(days=FORECAST_MAX_DAYS)
    if start_d > max_end:
        return pd.DataFrame()
    end_clamped = min(end_d, max_end)

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_VARS),
        "start_date": start_d.isoformat(),
        "end_date": end_clamped.isoformat(),
        "timezone": coerce_tz_or_auto(tz),
    }
    data = _request_open_meteo(FORECAST_API, params)
    return hourly_json_to_df(data)


@st.cache_data(show_spinner=False)
def fetch_open_meteo_archive(lat: float, lon: float, start_d: date, end_d: date, tz: str) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_VARS),
        "start_date": start_d.isoformat(),
        "end_date": end_d.isoformat(),
        "timezone": coerce_tz_or_auto(tz),
    }
    data = _request_open_meteo(ARCHIVE_API, params)
    return hourly_json_to_df(data)


def hourly_json_to_df(data: dict) -> pd.DataFrame:
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    df = pd.DataFrame({"time": pd.to_datetime(times, utc=False, errors="coerce")})
    for var in HOURLY_VARS:
        df[var] = hourly.get(var, [np.nan] * len(df))
    df["date"] = df["time"].dt.date
    df["hour_label"] = df["time"].dt.strftime("%H:00")
    df["weather_text"] = df["weathercode"].apply(lambda x: weathercode_to_text(int(x)) if pd.notna(x) else "")
    return df


def fetch_open_meteo_range(lat: float, lon: float, start_d: date, end_d: date, tz: str) -> pd.DataFrame:
    start_d, end_d = validate_date_range(start_d, end_d)
    today_d = date.today()

    parts: List[pd.DataFrame] = []

    if start_d < today_d:
        arch_end = min(end_d, today_d - timedelta(days=1))
        if start_d <= arch_end:
            df_arch = fetch_open_meteo_archive(lat, lon, start_d, arch_end, tz)
            if not df_arch.empty:
                parts.append(df_arch)

    if end_d >= today_d:
        fc_start = max(start_d, today_d)
        df_fc = fetch_open_meteo_forecast(lat, lon, fc_start, end_d, tz)
        if not df_fc.empty:
            parts.append(df_fc)

    if not parts:
        return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True)
    mask = (df["date"] >= start_d) & (df["date"] <= end_d)
    df = df.loc[mask].copy()
    df = df.sort_values("time").reset_index(drop=True)
    return df


# --------------------------
# App Logic
# --------------------------

def build_single_site_pivot(site_row: pd.Series, start_d: date, end_d: date) -> Tuple[pd.DataFrame, pd.DataFrame]:
    lat = float(site_row["latitude"])
    lon = float(site_row["longitude"])
    tz_name = str(site_row.get("timezone", "")) or "auto"

    df = fetch_open_meteo_range(lat, lon, start_d, end_d, tz_name)

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    raw_cols = [
        "time",
        "date",
        "hour_label",
        "weathercode",
        "weather_text",
        "precipitation",
        "cloudcover",
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

    sites_sub = sites_df[["location", "latitude", "longitude", "timezone"]].copy()
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
        tz = str(row.get("timezone", "auto"))
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
            df = fetch_open_meteo_range(float(lat), float(lon), sd_v, ed_v, tz)
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
                    "weathercode",
                    "weather_text",
                    "precipitation",
                    "cloudcover",
                ]
            ]
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

def render_sites_editor():
    st.subheader("Sites master (sites.csv)")
    st.caption(
        "Manage your sites list. Required columns: location, latitude, longitude. "
        "Optional: timezone (IANA, e.g., Asia/Kuala_Lumpur). Uploader accepts common aliases: "
        "name/site/place → location; lat → latitude; lon/lng/long → longitude; tz/time_zone → timezone."
    )

    uploaded = st.file_uploader(
        "Optionally upload a sites.csv (it will replace current file for this session)",
        type=["csv"],
        accept_multiple_files=False,
        key="sites_uploader",
    )
    if uploaded is not None:
        try:
            # Auto-detect delimiter and handle BOM
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

    # Load current sites
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
            "timezone": st.column_config.TextColumn("timezone"),
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
        "Locations will be matched to sites.csv to retrieve coordinates and timezone."
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
    st.write(
        "- Time format: 24-hour (HH:00)\n"
        "- Mode A: upload requests CSV with location,start_date,end_date (dates in YYYY-MM-DD)\n"
        "- Mode B: select a single site and date range to see a pivot (hour x date -> weather)\n"
        "- Edit sites.csv in-app (IANA timezones recommended, e.g., Asia/Kuala_Lumpur)."
    )

    with st.sidebar:
        st.header("Sites master")
        render_sites_editor()

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
            - Uploader accepts aliases: name/site/place → location; lat → latitude; lon/lng/long → longitude; tz/time_zone → timezone.
            - If you encounter an API error, verify that your timezone in `sites.csv` is a valid IANA zone (e.g., `Asia/Singapore`). Invalid timezones automatically fall back to `auto`.
            - Forecast endpoint typically supports up to 16 days ahead. Longer future ranges are auto-clamped.
            - Past dates are fetched from the Archive API; future from the Forecast API.
            - On Streamlit Cloud, saving files persists only for the current session/build.
            """
        )


if __name__ == "__main__":
    main()
