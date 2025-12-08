#!/usr/bin/env python3
# streamlit_app_custom_with_editor.py
"""
Streamlit app with in-app editor for sites.csv.

- Fixed local `sites.csv` containing site metadata: location,name (or site), lat, lon, timezone (optional)
- Editor: edit/save/reset/download sites.csv from the web UI
- Two modes:
  * Batch upload: upload CSV with columns (location,start,end) -> daily.csv & hourly.csv
  * Single site: select site & date range -> pivot table (hours x dates) of hourly category
Time format: 24-hour (HH:MM)
"""
import streamlit as st
import pandas as pd
import requests
import csv
import os
import time
from datetime import datetime, timedelta, date
from io import StringIO, BytesIO
from dateutil import parser

MM_TO_INCH = 0.03937007874015748  # $1\\ \\text{mm} \\approx 0.03937\\ \\text{in}$

# Minimal Windows -> IANA timezone mapping (extend if needed)
WINDOWS_TO_IANA = {
    "singapore standard time": "Asia/Singapore",
    "china standard time": "Asia/Shanghai",
    "indian standard time": "Asia/Kolkata",
    "eastern standard time": "America/New_York",
    "pacific standard time": "America/Los_Angeles",
}

# ---------- Helpers ----------
def normalize_timezone(tz_str):
    if not tz_str or not isinstance(tz_str, str):
        return None
    return WINDOWS_TO_IANA.get(tz_str.strip().lower(), None)

def excel_serial_to_date(serial):
    try:
        s = float(serial)
    except Exception:
        raise ValueError(f"Invalid excel serial: {serial}")
    base = datetime(1899, 12, 30)
    return (base + timedelta(days=int(round(s)))).date()

def parse_date_value(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        raise ValueError("Empty date")
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    s = str(v).strip()
    # try excel serial heuristic
    try:
        if s != "" and all(ch.isdigit() for ch in s) and len(s) <= 6:
            return excel_serial_to_date(s)
        try:
            f = float(s)
            if abs(f) > 365 and f < 60000:
                return excel_serial_to_date(f)
        except Exception:
            pass
    except Exception:
        pass
    try:
        dt = parser.parse(s)
        return dt.date()
    except Exception as e:
        raise ValueError(f"Unrecognized date format: {s}") from e

def detect_delimiter_from_text(text):
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",\t;")
        return dialect.delimiter
    except Exception:
        return "\t" if "\t" in text else ","

# ---------- sites.csv read/write and editor state ----------
SITES_PATH = "sites.csv"

def read_sites_local(path=SITES_PATH):
    """
    Read local `sites.csv`. Expected minimal columns:
      location (or name or site), lat, lon, timezone (optional).
    Returns DataFrame normalized to columns: 'location','lat','lon','timezone' (timezone optional).
    """
    if not os.path.exists(path):
        # Return empty DataFrame with columns so UI can create rows
        return pd.DataFrame(columns=["location", "lat", "lon", "timezone"])
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        sample = fh.read(8192)
        fh.seek(0)
        delim = detect_delimiter_from_text(sample)
        df = pd.read_csv(fh, delimiter=delim, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}
    colmap = {}
    for k in ("location", "site", "name"):
        if k in lower:
            colmap[lower[k]] = "location"
            break
    for k in ("lat", "latitude"):
        if k in lower:
            colmap[lower[k]] = "lat"
            break
    for k in ("lon", "longitude"):
        if k in lower:
            colmap[lower[k]] = "lon"
            break
    for k in ("timezone", "tz"):
        if k in lower:
            colmap[lower[k]] = "timezone"
            break
    df = df.rename(columns=colmap)
    # ensure required cols present (if missing, create empty)
    for c in ("location", "lat", "lon", "timezone"):
        if c not in df.columns:
            df[c] = ""
    # keep ordering
    df = df[["location", "lat", "lon", "timezone"]]
    return df.fillna("")

def write_sites_local(df, path=SITES_PATH):
    # Validate required columns
    for c in ("location", "lat", "lon"):
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    # Save as CSV (comma)
    df_to_save = df[["location", "lat", "lon", "timezone"]]
    df_to_save.to_csv(path, index=False)
    return True

# Helper to store edited dataframe in session state
def get_edited_sites_state():
    if "edited_sites_df" not in st.session_state:
        st.session_state["edited_sites_df"] = read_sites_local(SITES_PATH)
    return st.session_state["edited_sites_df"]

def reset_sites_state():
    st.session_state["edited_sites_df"] = read_sites_local(SITES_PATH)

# ---------- Open-Meteo fetch (cached) ----------
@st.cache_data(show_spinner=False)
def fetch_open_meteo(lat, lon, start_date_iso, end_date_iso, tz_iana=None):
    base = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": float(lat),
        "longitude": float(lon),
        "start_date": start_date_iso,
        "end_date": end_date_iso,
        "hourly": "precipitation",
        "daily": "precipitation_sum",
        "timezone": (tz_iana if tz_iana else "auto"),
    }
    r = requests.get(base, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    parsed = {"daily": [], "hourly": []}
    if "daily" in data and "time" in data["daily"]:
        times = data["daily"].get("time", [])
        sums = data["daily"].get("precipitation_sum", [])
        for t, s in zip(times, sums):
            parsed["daily"].append({"date": t, "precipitation_sum": (s if s is not None else None)})
    if "hourly" in data and "time" in data["hourly"]:
        times = data["hourly"].get("time", [])
        prec = data["hourly"].get("precipitation", [])
        for t, p in zip(times, prec):
            parsed["hourly"].append({"time": t, "precipitation": (p if p is not None else 0.0)})
    return parsed

# ---------- Classification ----------
def classify_hourly_precip(p_mm, any_thunder_flag=False, cloud_zero=False):
    p = float(p_mm) if p_mm is not None else 0.0
    if p == 0:
        return "cloudy" if cloud_zero else "sunny"
    if any_thunder_flag or p >= 30.0:
        return "thunderstorm"
    if p >= 7.6:
        return "heavy rain"
    if p < 2.5:
        return "small rain"
    return "heavy rain"

def classify_precipitation_window(hourly_precip_mm, any_thunder_flag=False, cloud_zero=False):
    hrs = list(hourly_precip_mm)
    if not hrs:
        return "cloudy" if cloud_zero else "sunny"
    total = sum(hrs)
    peak = max(hrs) if hrs else 0.0
    SMALL_TOTAL_THRESHOLD = 5.0
    HEAVY_TOTAL_THRESHOLD = 20.0
    LIGHT_HOURLY = 2.5
    HEAVY_HOURLY = 7.6
    THUNDER_HOURLY = 30.0
    if any_thunder_flag or peak >= THUNDER_HOURLY:
        return "thunderstorm"
    if total == 0:
        return "cloudy" if cloud_zero else "sunny"
    if peak >= HEAVY_HOURLY or total >= HEAVY_TOTAL_THRESHOLD:
        return "heavy rain"
    if peak < LIGHT_HOURLY and total < SMALL_TOTAL_THRESHOLD:
        return "small rain"
    return "heavy rain"

# ---------- Batch builder (upload) ----------
def process_batch_upload(uploaded_file, sites_df, cloud_zero=False, include_tz=False, include_inches=False, delay=1.0):
    raw = uploaded_file.read()
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="replace")
    else:
        text = str(raw)
    delim = detect_delimiter_from_text(text)
    df = pd.read_csv(StringIO(text), delimiter=delim, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}
    if "location" not in lower or ("start" not in lower and "date_start" not in lower) or ("end" not in lower and "date_end" not in lower):
        raise ValueError("Uploaded CSV must include columns: location, start, end (names case-insensitive)")
    loc_col = lower["location"]
    start_col = lower.get("start", lower.get("date_start"))
    end_col = lower.get("end", lower.get("date_end"))
    rows = []
    for _, r in df.iterrows():
        loc = str(r[loc_col]).strip()
        try:
            sd = parse_date_value(r[start_col])
            ed = parse_date_value(r[end_col])
        except Exception as e:
            st.warning(f"Skipping row for {loc}: invalid dates ({e})")
            continue
        rows.append({"location": loc, "start": sd, "end": ed})
    if not rows:
        raise ValueError("No valid rows to process in uploaded CSV")

    daily_rows = []
    hourly_rows = []
    total = len(rows)
    progress = st.progress(0)
    processed = 0
    for rr in rows:
        processed += 1
        loc = rr["location"]
        sd = rr["start"]
        ed = rr["end"]
        match = sites_df[sites_df["location"].str.lower() == loc.lower()]
        if match.empty:
            st.warning(f"Location '{loc}' not found in sites.csv; skipping")
            progress.progress(processed / total)
            continue
        site = match.iloc[0]
        lat = site["lat"]
        lon = site["lon"]
        tz_raw = site.get("timezone", "")
        tz_param = normalize_timezone(tz_raw)
        try:
            api = fetch_open_meteo(lat, lon, sd.isoformat(), ed.isoformat(), tz_param)
        except Exception as e:
            st.warning(f"Open-Meteo error for {loc}: {e}")
            progress.progress(processed / total)
            time.sleep(delay)
            continue

        daily_map = {d["date"]: d.get("precipitation_sum") for d in api.get("daily", [])}
        hourly_by_date = {}
        for h in api.get("hourly", []):
            tstr = h.get("time")
            p = h.get("precipitation", 0.0)
            if not tstr:
                continue
            date_part, _, time_part = tstr.partition("T")
            hourly_by_date.setdefault(date_part, []).append({"time": tstr, "precipitation": p})

        cur = sd
        while cur <= ed:
            date_iso = cur.isoformat()
            daily_precip = daily_map.get(date_iso, None)
            hourly_list = hourly_by_date.get(date_iso, [])
            hourly_vals = [float(h.get("precipitation", 0.0)) for h in hourly_list]
            if not hourly_vals and daily_precip not in (None, ""):
                hourly_vals = [float(daily_precip)]
            day_cat = classify_precipitation_window(hourly_vals, any_thunder_flag=False, cloud_zero=cloud_zero)
            hourly_sum = round(sum(hourly_vals), 3) if hourly_vals else ""
            peak_hour = round(max(hourly_vals), 3) if hourly_vals else ""
            daily_row = {
                "site": loc, "lat": lat, "lon": lon, "date": date_iso,
                "precipitation_sum_mm": ("" if daily_precip is None else daily_precip),
                "hourly_sum_mm": hourly_sum, "peak_hour_mm": peak_hour, "category": day_cat
            }
            if include_tz:
                daily_row["timezone"] = (tz_param if tz_param else (tz_raw or "auto"))
            if include_inches:
                daily_row["precipitation_sum_in"] = (round(float(daily_row["precipitation_sum_mm"]) * MM_TO_INCH, 3)
                                                     if daily_row["precipitation_sum_mm"] not in ("", None) else "")
                daily_row["hourly_sum_in"] = (round(float(hourly_sum) * MM_TO_INCH, 3)
                                              if hourly_sum not in ("", None, "") else "")
                daily_row["peak_hour_in"] = (round(float(peak_hour) * MM_TO_INCH, 3)
                                             if peak_hour not in ("", None, "") else "")
            daily_rows.append(daily_row)

            if hourly_list:
                for h in hourly_list:
                    t_iso = h.get("time")
                    p = float(h.get("precipitation", 0.0))
                    hour_cat = classify_hourly_precip(p, any_thunder_flag=False, cloud_zero=cloud_zero)
                    hourly_row = {
                        "site": loc, "lat": lat, "lon": lon, "date": date_iso,
                        "time": t_iso, "precipitation_mm": round(p, 3),
                        "day_category": day_cat, "hour_category": hour_cat
                    }
                    if include_tz:
                        hourly_row["timezone"] = (tz_param if tz_param else (tz_raw or "auto"))
                    if include_inches:
                        hourly_row["precipitation_in"] = round(p * MM_TO_INCH, 3)
                    hourly_rows.append(hourly_row)
            cur += timedelta(days=1)
        time.sleep(delay)
        progress.progress(processed / total)
    progress.empty()
    daily_df = pd.DataFrame(daily_rows)
    hourly_df = pd.DataFrame(hourly_rows)
    return daily_df, hourly_df

# ---------- Single site pivot builder ----------
def build_single_site_pivot(site_row, start_date, end_date, cloud_zero=False):
    name = site_row["location"]
    lat = site_row["lat"]
    lon = site_row["lon"]
    tz_param = normalize_timezone(site_row.get("timezone", ""))
    api = fetch_open_meteo(lat, lon, start_date.isoformat(), end_date.isoformat(), tz_param)
    hourly_records = api.get("hourly", [])
    rec_map = {}
    dates = []
    for h in hourly_records:
        tstr = h.get("time")
        p = float(h.get("precipitation", 0.0))
        if not tstr:
            continue
        dt = pd.to_datetime(tstr)
        date_iso = dt.date().isoformat()
        hour_label = dt.strftime("%H:%M")
        cat = classify_hourly_precip(p, any_thunder_flag=False, cloud_zero=cloud_zero)
        rec_map[(hour_label, date_iso)] = cat
        if date_iso not in dates:
            dates.append(date_iso)
    dates = sorted(list(set(dates)))
    hours = [f"{h:02d}:00" for h in range(24)]
    pivot = pd.DataFrame(index=hours, columns=dates)
    for d in dates:
        for hr in hours:
            val = rec_map.get((hr, d), "")
            pivot.at[hr, d] = val
    hourly_rows = []
    for (hr, d), cat in rec_map.items():
        hourly_rows.append({"site": name, "date": d, "hour": hr, "category": cat})
    hourly_df = pd.DataFrame(hourly_rows)
    return pivot, hourly_df

# ---------- Streamlit UI ----------
st.set_page_config(layout="wide", page_title="Weather App (editor)")
st.title("Weather App — sites.csv editor + Batch/Single-site processing")

# Left column: sites editor + controls
editor_col, app_col = st.columns([1, 3])

with editor_col:
    st.header("sites.csv Editor")
    st.markdown("Edit `sites.csv` (local file). Required columns: `location`, `lat`, `lon`. Optional: `timezone`.")
    # Initialize state
    edited_df = get_edited_sites_state()
    # Provide buttons
    btn_col1, btn_col2, btn_col3 = st.columns([1,1,1])
    with btn_col1:
        if st.button("Reload from file"):
            reset_sites_state()
            st.experimental_rerun()
    with btn_col2:
        if st.button("Add empty row"):
            # append an empty row
            new_row = {"location": "", "lat": "", "lon": "", "timezone": ""}
            st.session_state["edited_sites_df"] = pd.concat([st.session_state["edited_sites_df"], pd.DataFrame([new_row])], ignore_index=True)
            st.experimental_rerun()
    with btn_col3:
        if st.button("Save sites.csv"):
            try:
                write_sites_local(st.session_state["edited_sites_df"])
                st.success(f"Saved to {SITES_PATH}")
            except Exception as e:
                st.error(f"Failed to save: {e}")

    # Editable table area
    st.markdown("Edit cells directly below. Add rows using the button above. Delete rows by clearing the 'location' value and saving.")
    # Use newer st.data_editor if available, otherwise fall back to experimental
    try:
        # st.data_editor returns the edited DataFrame
        edited = st.data_editor(edited_df, num_rows="dynamic")
    except Exception:
        edited = st.experimental_data_editor(edited_df, num_rows="dynamic")

    # store edited copy back in session state
    st.session_state["edited_sites_df"] = edited

    # Download current edited dataset
    csv_sites = edited.to_csv(index=False).encode("utf-8")
    st.download_button("Download sites CSV (current)", data=csv_sites, file_name="sites_current.csv", mime="text/csv")

# Right/main column: app functionality
with app_col:
    st.header("Processing Modes")
    sites_df = get_edited_sites_state()  # use edited dataset for processing
    st.info(f"{len(sites_df)} sites loaded (live from editor).")

    mode = st.radio("Mode", ["Batch upload (CSV)", "Single site pivot"], horizontal=True)
    # Common options
    st.markdown("Options")
    cloudy = st.checkbox("Treat zeros as 'cloudy' instead of 'sunny'", value=False)
    include_tz = st.checkbox("Include timezone columns in outputs (batch mode)", value=True)
    include_inches = st.checkbox("Add inches columns (mm → in)", value=False)
    delay = st.slider("Delay between site API calls (seconds)", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
    max_sites = st.number_input("Max sites for batch (0 = all)", min_value=0, value=0, step=1)

    if mode == "Batch upload (CSV)":
        st.markdown("Upload CSV with columns: `location, start, end` (start/end can be Excel serial or date string).")
        uploaded = st.file_uploader("Upload CSV for batch processing", type=["csv", "txt"])
        if uploaded is not None:
            try:
                daily_df, hourly_df = process_batch_upload(uploaded, sites_df,
                                                           cloud_zero=cloudy,
                                                           include_tz=include_tz,
                                                           include_inches=include_inches,
                                                           delay=delay)
            except Exception as e:
                st.error(f"Error processing uploaded CSV: {e}")
            else:
                st.success("Batch processing complete")
                st.subheader("Daily")
                st.dataframe(daily_df)
                st.download_button("Download daily.csv", data=daily_df.to_csv(index=False).encode("utf-8"),
                                   file_name="daily.csv", mime="text/csv")
                st.subheader("Hourly")
                st.dataframe(hourly_df)
                st.download_button("Download hourly.csv", data=hourly_df.to_csv(index=False).encode("utf-8"),
                                   file_name="hourly.csv", mime="text/csv")
        else:
            st.info("Please upload a CSV to start batch processing.")
    else:
        st.markdown("Select a site (from `sites.csv`) and choose start/end dates. Pivot rows = hours ($00:00$..$23:00$), columns = dates.")
        if sites_df.empty:
            st.warning("No sites available. Use the editor to add sites and save.")
        else:
            site_options = sites_df["location"].tolist()
            selected = st.selectbox("Choose site", options=site_options)
            site_row = sites_df[sites_df["location"] == selected].iloc[0]
            today = datetime.utcnow().date()
            default_end = today - timedelta(days=1)
            default_start = default_end - timedelta(days=6)
            start_date = st.date_input("Start date", value=default_start, max_value=default_end)
            end_date = st.date_input("End date", value=default_end, min_value=start_date)
            run = st.button("Fetch & Build pivot")
            if run:
                if start_date > end_date:
                    st.error("Start date must be on or before end date.")
                else:
                    with st.spinner("Fetching data and building pivot..."):
                        pivot_df, pivot_hourly_df = build_single_site_pivot(site_row, start_date, end_date, cloud_zero=cloudy)
                    st.success("Pivot ready")
                    st.subheader("Hourly category pivot (rows=hour 24hr, columns=dates)")
                    st.dataframe(pivot_df)
                    st.download_button("Download pivot (CSV)", data=pivot_df.to_csv(index=True).encode("utf-8"),
                                       file_name=f"pivot_{selected}.csv", mime="text/csv")
                    if not pivot_hourly_df.empty:
                        st.download_button("Download hourly categories (flat)", data=pivot_hourly_df.to_csv(index=False).encode("utf-8"),
                                           file_name=f"hourly_categories_{selected}.csv", mime="text/csv")