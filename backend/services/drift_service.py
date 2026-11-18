"""Real-time ocean weather & 72-hour Lagrangian drift trajectory service.

Connects to Open-Meteo Marine API for live surface current velocity (u, v in m/s)
and 10-metre wind vectors, then integrates a Lagrangian drift model:

    V_drift = V_current + (0.025 * V_wind) + V_stokes

Stokes drift is approximated from the wave parameters when available, or
estimated as 1.2% of the wind speed (empirical deep-water rule).

Returns projected centroid positions at +6h, +12h, +24h, +48h, +72h intervals.
All coordinates come from the live detected centroid: nothing is hard-coded.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger("aquaflex.drift_service")

# ── Physical constants ─────────────────────────────────────────────────────────
METERS_PER_DEG_LAT = 111_320.0
WIND_LEEWAY_FACTOR = 0.025          # 2.5 % Lagrangian wind coupling
STOKES_WIND_FACTOR  = 0.012          # Stokes drift ≈ 1.2 % of wind speed (fallback)

# ── Forecast steps ─────────────────────────────────────────────────────────────
RECORD_HOURS = [6, 12, 24, 48, 72]
TIMESTEP_H   = 1                     # 1-hour integration step

# ── Reference Coastal River Outlets & Estuaries (South Asia focus) ─────────────
RIVER_OUTLETS: List[Dict[str, Any]] = [
    {"name": "Periyar River Estuary",    "lat":  9.9617, "lon": 76.2699, "type": "RIVER_RUNOFF", "region": "Kerala"},
    {"name": "Pamba River Mouth",        "lat":  9.2396, "lon": 76.3961, "type": "RIVER_RUNOFF", "region": "Kerala"},
    {"name": "Chaliyar River Mouth",     "lat": 11.2548, "lon": 75.7781, "type": "RIVER_RUNOFF", "region": "Kerala"},
    {"name": "Bharathapuzha Estuary",    "lat": 10.8530, "lon": 75.9640, "type": "RIVER_RUNOFF", "region": "Kerala"},
    {"name": "Kallada River Mouth",      "lat":  8.9036, "lon": 76.5570, "type": "RIVER_RUNOFF", "region": "Kerala"},
    {"name": "Tamiraparani River Mouth", "lat":  8.5614, "lon": 77.9999, "type": "RIVER_RUNOFF", "region": "Tamil Nadu"},
    {"name": "Vaigai River Mouth",       "lat":  9.2696, "lon": 79.1234, "type": "RIVER_RUNOFF", "region": "Tamil Nadu"},
    {"name": "Cauvery Delta Outlet",     "lat": 11.0640, "lon": 79.8588, "type": "RIVER_RUNOFF", "region": "Tamil Nadu"},
    {"name": "Godavari Delta Outflow",   "lat": 16.5600, "lon": 82.2800, "type": "RIVER_RUNOFF", "region": "Andhra Pradesh"},
    {"name": "Krishna Estuary",          "lat": 15.7200, "lon": 80.9700, "type": "RIVER_RUNOFF", "region": "Andhra Pradesh"},
    {"name": "Mahanadi Delta",           "lat": 20.2700, "lon": 86.5400, "type": "RIVER_RUNOFF", "region": "Odisha"},
    {"name": "Hooghly River Estuary",    "lat": 21.6500, "lon": 88.0500, "type": "RIVER_RUNOFF", "region": "West Bengal"},
    {"name": "Zuari River Outflow",      "lat": 15.4167, "lon": 73.8000, "type": "RIVER_RUNOFF", "region": "Goa"},
    {"name": "Netravati River Mouth",    "lat": 12.8333, "lon": 74.8333, "type": "RIVER_RUNOFF", "region": "Karnataka"},
    {"name": "Ulhas River Creek",        "lat": 19.2800, "lon": 72.8800, "type": "COASTAL_OUTFLOW", "region": "Maharashtra"},
    {"name": "Ennore Creek Outflow",     "lat": 13.2323, "lon": 80.3242, "type": "COASTAL_OUTFLOW", "region": "Tamil Nadu"},
    {"name": "Adyar River Mouth",        "lat": 13.0100, "lon": 80.2750, "type": "COASTAL_OUTFLOW", "region": "Tamil Nadu"},
    {"name": "Cochin Backwaters Channel","lat":  9.9700, "lon": 76.2400, "type": "COASTAL_OUTFLOW", "region": "Kerala"},
]

# ── Reference Major Shipping Corridors & Port Entrances ───────────────────────
SHIPPING_CORRIDORS: List[Dict[str, Any]] = [
    {"name": "Kochi Port Approach Corridor",          "lat":  9.9650, "lon": 76.2200, "type": "VESSEL_DISCHARGE", "traffic": "High Container & Tanker Traffic"},
    {"name": "Colombo International Shipping Lane",   "lat":  6.9500, "lon": 79.8000, "type": "VESSEL_DISCHARGE", "traffic": "Major East-West Transshipment Lane"},
    {"name": "Mumbai Deepwater Offshore Channel",     "lat": 18.9200, "lon": 72.8200, "type": "VESSEL_DISCHARGE", "traffic": "Commercial & Chemical Cargo Flow"},
    {"name": "Visakhapatnam Port Entrance Lane",      "lat": 17.6800, "lon": 83.2900, "type": "VESSEL_DISCHARGE", "traffic": "Bulk Mineral & Naval Traffic"},
    {"name": "Chennai Port Commercial Approach",      "lat": 13.0900, "lon": 80.3000, "type": "VESSEL_DISCHARGE", "traffic": "Major Bay of Bengal Maritime Hub"},
    {"name": "JNPT / Nhava Sheva Approach Lane",      "lat": 18.9500, "lon": 72.9500, "type": "VESSEL_DISCHARGE", "traffic": "Primary Indian Container Gateway"},
    {"name": "Mormugao Harbor Shipping Lane",         "lat": 15.4100, "lon": 73.7900, "type": "VESSEL_DISCHARGE", "traffic": "Iron Ore & Coastal Shipping Corridor"},
    {"name": "Mangalore Port Entrance Route",         "lat": 12.9200, "lon": 74.8000, "type": "VESSEL_DISCHARGE", "traffic": "Petrochemical & Bulk Cargo Route"},
    {"name": "Tuticorin / VOC Port Channel",          "lat":  8.7500, "lon": 78.1800, "type": "VESSEL_DISCHARGE", "traffic": "Gulf of Mannar Coastal Gateway"},
    {"name": "Cape Comorin Maritime Chokepoint",      "lat":  7.9500, "lon": 77.5500, "type": "VESSEL_DISCHARGE", "traffic": "Southern Peninsular Tanker Route"},
    {"name": "Paradip Deep Sea Approach Channel",     "lat": 20.2600, "lon": 86.6800, "type": "VESSEL_DISCHARGE", "traffic": "East Coast Crude & Coal Port"},
    {"name": "Galle Southern International Corridor", "lat":  5.9500, "lon": 80.2500, "type": "VESSEL_DISCHARGE", "traffic": "Global Mega-Container Transit Lane"},
]


# ── Open-Meteo Marine API ──────────────────────────────────────────────────────
def _fetch_open_meteo_marine(lat: float, lon: float) -> Dict[str, Any]:
    """Fetch live hourly ocean current + wave data from Open-Meteo Marine API."""
    url = "https://marine-api.open-meteo.com/v1/marine"
    params = {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "hourly": "ocean_current_velocity,ocean_current_direction,wave_height,swell_wave_direction",
        "forecast_days": 4,
        "timezone": "UTC",
    }
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Open-Meteo Marine fetch failed: %s", exc)
        return {}


def _fetch_open_meteo_wind(lat: float, lon: float) -> Dict[str, Any]:
    """Fetch live 10 m wind speed + direction (GFS/ERA5) from Open-Meteo."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "hourly": "windspeed_10m,winddirection_10m",
        "forecast_days": 4,
        "timezone": "UTC",
        "windspeed_unit": "ms",
    }
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Open-Meteo Wind fetch failed: %s", exc)
        return {}


def _dir_to_uv(speed: float, direction_deg: float) -> Tuple[float, float]:
    """Convert oceanographic convention (direction the water/wind flows TOWARD) to (u, v)."""
    rad = math.radians(direction_deg)
    u = speed * math.sin(rad)
    v = speed * math.cos(rad)
    return u, v


def _extract_hourly_vectors(
    marine_data: Dict[str, Any],
    wind_data: Dict[str, Any],
    n_hours: int,
) -> List[Dict[str, float]]:
    """Build a list of per-hour ocean + wind vectors (u/v in m/s)."""
    marine_hourly = marine_data.get("hourly", {})
    wind_hourly   = wind_data.get("hourly", {})

    cur_vel = marine_hourly.get("ocean_current_velocity", [])
    cur_dir = marine_hourly.get("ocean_current_direction", [])
    win_spd = wind_hourly.get("windspeed_10m", [])
    win_dir = wind_hourly.get("winddirection_10m", [])
    wav_ht  = marine_hourly.get("wave_height", [])

    vectors: List[Dict[str, float]] = []
    for i in range(min(n_hours, max(len(cur_vel), len(win_spd), 1))):
        cv  = float(cur_vel[i]) if i < len(cur_vel) and cur_vel[i] is not None else 0.05
        cd  = float(cur_dir[i]) if i < len(cur_dir) and cur_dir[i] is not None else 180.0
        ws  = float(win_spd[i]) if i < len(win_spd) and win_spd[i] is not None else 4.0
        wd  = float(win_dir[i]) if i < len(win_dir) and win_dir[i] is not None else 225.0
        wh  = float(wav_ht[i])  if i < len(wav_ht)  and wav_ht[i]  is not None else None

        u_cur, v_cur = _dir_to_uv(cv, cd)
        u_wind, v_wind = _dir_to_uv(ws, wd)

        # Stokes drift: if wave height available use Hs-based approximation,
        # else fall back to 1.2 % of wind speed in the wind direction.
        if wh is not None and wh > 0:
            stokes_speed = 0.016 * ws * wh        # Holthuijsen approximation
        else:
            stokes_speed = STOKES_WIND_FACTOR * ws
        u_stokes, v_stokes = _dir_to_uv(stokes_speed, wd)

        vectors.append({
            "u_current":  round(u_cur,    4),
            "v_current":  round(v_cur,    4),
            "u_wind":     round(u_wind,   4),
            "v_wind":     round(v_wind,   4),
            "u_stokes":   round(u_stokes, 4),
            "v_stokes":   round(v_stokes, 4),
            "wind_speed": round(ws, 2),
            "wave_height": round(wh, 2) if wh is not None else None,
        })

    # Pad with last value if forecast is shorter than requested.
    while len(vectors) < n_hours:
        vectors.append(vectors[-1] if vectors else {
            "u_current": 0.05, "v_current": 0.0,
            "u_wind": 2.0, "v_wind": 0.0,
            "u_stokes": 0.024, "v_stokes": 0.0,
            "wind_speed": 4.0, "wave_height": None,
        })
    return vectors


def _to_cardinal(deg: float) -> str:
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = int((deg + 11.25) / 22.5) % 16
    return dirs[idx]


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return round(2 * r * math.asin(math.sqrt(a)), 2)


def compute_lagrangian_drift(
    lat: float,
    lon: float,
    vectors: List[Dict[str, float]],
) -> List[Dict[str, Any]]:
    """Step-wise Lagrangian integration over ``vectors`` (1-hour timesteps).

    Drift model:
        V_drift = V_current + (0.025 * V_wind) + V_stokes

    Returns a list of dicts for each RECORD_HOURS step with heading and distance.
    """
    cur_lat, cur_lon = lat, lon
    results: List[Dict[str, Any]] = []

    for hour in range(1, max(RECORD_HOURS) + 1):
        idx = min(hour - 1, len(vectors) - 1)
        v = vectors[idx]

        u_drift = v["u_current"] + WIND_LEEWAY_FACTOR * v["u_wind"] + v["u_stokes"]
        v_drift = v["v_current"] + WIND_LEEWAY_FACTOR * v["v_wind"] + v["v_stokes"]

        dt = 3600.0  # seconds per hour
        dlat = (v_drift * dt) / METERS_PER_DEG_LAT
        dlon = (u_drift * dt) / (
            METERS_PER_DEG_LAT * max(math.cos(math.radians(cur_lat)), 1e-6)
        )
        cur_lat += dlat
        cur_lon += dlon

        if hour in RECORD_HOURS:
            heading_deg = round((math.degrees(math.atan2(u_drift, v_drift)) + 360.0) % 360.0, 1)
            dist_km = _haversine_km(lat, lon, cur_lat, cur_lon)
            results.append({
                "hour":                   hour,
                "latitude":               round(cur_lat, 5),
                "longitude":              round(cur_lon, 5),
                "heading_deg":            heading_deg,
                "cardinal_heading":       _to_cardinal(heading_deg),
                "distance_from_origin_km": dist_km,
                "u_drift":                round(u_drift, 4),
                "v_drift":                round(v_drift, 4),
                "u_current":              v["u_current"],
                "v_current":              v["v_current"],
                "u_wind":                 v["u_wind"],
                "v_wind":                 v["v_wind"],
                "u_stokes":               v["u_stokes"],
                "v_stokes":               v["v_stokes"],
                "drift_speed_ms":         round(math.hypot(u_drift, v_drift), 3),
                "wind_speed_ms":          v["wind_speed"],
                "wave_height_m":          v["wave_height"],
            })

    return results


def run_drift_forecast(
    centroid_lat: float,
    centroid_lon: float,
) -> Dict[str, Any]:
    """Full 72-hour Lagrangian drift forecast from the detected debris centroid.

    1. Fetches live current + wind data from Open-Meteo.
    2. Integrates the Lagrangian particle model.
    3. Returns projected positions at +6/+12/+24/+48/+72 h.
    """
    logger.info(
        "Running drift forecast from centroid (%.4f, %.4f)", centroid_lat, centroid_lon
    )

    marine_data = _fetch_open_meteo_marine(centroid_lat, centroid_lon)
    wind_data   = _fetch_open_meteo_wind(centroid_lat, centroid_lon)

    # Assess data availability
    data_available = bool(marine_data.get("hourly") or wind_data.get("hourly"))

    vectors = _extract_hourly_vectors(marine_data, wind_data, max(RECORD_HOURS))
    positions = compute_lagrangian_drift(centroid_lat, centroid_lon, vectors)

    # Summarise the surface conditions at hour 0 (current fetch)
    h0 = vectors[0] if vectors else {}
    surface_conditions = {
        "u_current_ms":   h0.get("u_current", 0.0),
        "v_current_ms":   h0.get("v_current", 0.0),
        "u_wind_ms":      h0.get("u_wind", 0.0),
        "v_wind_ms":      h0.get("v_wind", 0.0),
        "wind_speed_ms":  h0.get("wind_speed", 0.0),
        "wave_height_m":  h0.get("wave_height"),
    }

    return {
        "origin_latitude":    centroid_lat,
        "origin_longitude":   centroid_lon,
        "forecast_horizon_h": max(RECORD_HOURS),
        "record_hours":       RECORD_HOURS,
        "data_source":        "open-meteo-marine + open-meteo-forecast",
        "data_available":     data_available,
        "surface_conditions": surface_conditions,
        "drift_equation":     "V_drift = V_current + (0.025 * V_wind) + V_stokes",
        "positions":          positions,
    }


# ── Reverse Lagrangian Backtracking & Source Attribution Engine ──────────────
def _fetch_open_meteo_historical(
    lat: float,
    lon: float,
    past_days: int = 3,
) -> Tuple[List[Dict[str, float]], bool]:
    """Fetch past hourly ocean current and 10m wind vectors from Open-Meteo for backtracking.

    Returns list of hourly vectors ordered backwards from current hour (t-1h, t-2h, ...).
    """
    marine_url = "https://marine-api.open-meteo.com/v1/marine"
    marine_params = {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "hourly": "ocean_current_velocity,ocean_current_direction,wave_height",
        "past_days": past_days,
        "forecast_days": 1,
        "timezone": "UTC",
    }
    wind_url = "https://api.open-meteo.com/v1/forecast"
    wind_params = {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "hourly": "windspeed_10m,winddirection_10m",
        "past_days": past_days,
        "forecast_days": 1,
        "timezone": "UTC",
        "windspeed_unit": "ms",
    }
    try:
        m_resp = requests.get(marine_url, params=marine_params, timeout=12)
        w_resp = requests.get(wind_url, params=wind_params, timeout=12)
        m_data = m_resp.json() if m_resp.status_code == 200 else {}
        w_data = w_resp.json() if w_resp.status_code == 200 else {}
        m_hourly = m_data.get("hourly", {})
        w_hourly = w_data.get("hourly", {})

        cur_vel = m_hourly.get("ocean_current_velocity", [])
        cur_dir = m_hourly.get("ocean_current_direction", [])
        win_spd = w_hourly.get("windspeed_10m", [])
        win_dir = w_hourly.get("winddirection_10m", [])
        wav_ht  = m_hourly.get("wave_height", [])

        total_pts = min(len(cur_vel), len(win_spd)) if (cur_vel and win_spd) else max(len(cur_vel), len(win_spd))
        if total_pts > 0:
            target_hours = past_days * 24
            vectors: List[Dict[str, float]] = []
            for step in range(1, min(target_hours + 1, total_pts + 1)):
                idx = total_pts - step
                cv = float(cur_vel[idx]) if idx < len(cur_vel) and cur_vel[idx] is not None else 0.05
                cd = float(cur_dir[idx]) if idx < len(cur_dir) and cur_dir[idx] is not None else 180.0
                ws = float(win_spd[idx]) if idx < len(win_spd) and win_spd[idx] is not None else 3.5
                wd = float(win_dir[idx]) if idx < len(win_dir) and win_dir[idx] is not None else 225.0
                wh = float(wav_ht[idx])  if idx < len(wav_ht)  and wav_ht[idx]  is not None else None

                u_cur, v_cur = _dir_to_uv(cv, cd)
                u_wind, v_wind = _dir_to_uv(ws, wd)
                stk_spd = (0.016 * ws * wh) if (wh is not None and wh > 0) else (STOKES_WIND_FACTOR * ws)
                u_stk, v_stk = _dir_to_uv(stk_spd, wd)

                vectors.append({
                    "u_current": round(u_cur, 4),
                    "v_current": round(v_cur, 4),
                    "u_wind":    round(u_wind, 4),
                    "v_wind":    round(v_wind, 4),
                    "u_stokes":  round(u_stk, 4),
                    "v_stokes":  round(v_stk, 4),
                    "wind_speed": round(ws, 2),
                    "wave_height": round(wh, 2) if wh is not None else None,
                })
            if vectors:
                return vectors, True
    except Exception as exc:
        logger.warning("Historical Open-Meteo vector query failed (%s); using fallback climatological data.", exc)

    fallback: List[Dict[str, float]] = []
    for _ in range(past_days * 24):
        fallback.append({
            "u_current": 0.045,
            "v_current": 0.022,
            "u_wind":    2.2,
            "v_wind":    1.1,
            "u_stokes":  0.026,
            "v_stokes":  0.013,
            "wind_speed": 3.5,
            "wave_height": 1.1,
        })
    return fallback, False


def calculate_reverse_attribution(
    lat: float,
    lon: float,
    hours: int = 72,
) -> Dict[str, Any]:
    """Reverse Lagrangian Backtracking & Source Attribution Module.

    1. Fetches historical surface current (-u, -v) and wind vectors for past 72 hours from Open-Meteo.
    2. Integrates backward in time step-by-step:
           lat_prev = lat_current - (v_current * dt_hours)
           lon_prev = lon_current - (u_current * dt_hours)
    3. Checks spatial intersection at each step against reference point dictionaries:
           - River Outlets (Periyar River Estuary, Godavari Delta Outflow, Krishna Estuary, etc.)
           - Shipping Corridors / Port Entrances (Kochi, Colombo, Mumbai, Vizag, Chennai, etc.)
    4. Returns structured attribution result with:
           - origin_type: "RIVER_RUNOFF" | "VESSEL_DISCHARGE" | "OFFSHORE_SPILL" | "COASTAL_OUTFLOW"
           - origin_coordinates: { "latitude": ..., "longitude": ... }
           - backtrack_path: list of { hour, latitude, longitude }
           - nearest_source: { "name": ..., "type": ..., "distance_km": ... }
           - confidence_score: float (0.0 to 1.0)
           - discharge_timestamp_estimate: ISO string
    """
    logger.info("Running reverse Lagrangian attribution for (%.4f, %.4f) over %d hours", lat, lon, hours)

    past_days = max(1, math.ceil(hours / 24))
    hist_vectors, data_available = _fetch_open_meteo_historical(lat, lon, past_days=past_days)

    cur_lat, cur_lon = lat, lon
    dt_sec = 3600.0  # 1 hour steps
    backtrack_path: List[Dict[str, Any]] = []
    intersection_candidates: List[Dict[str, Any]] = []

    # Initial detected location at T=0
    backtrack_path.append({
        "hour": 0,
        "latitude": round(cur_lat, 5),
        "longitude": round(cur_lon, 5),
        "u_drift": 0.0,
        "v_drift": 0.0,
        "distance_from_detection_km": 0.0,
    })

    for step in range(1, hours + 1):
        idx = min(step - 1, len(hist_vectors) - 1)
        v = hist_vectors[idx] if hist_vectors else {
            "u_current": 0.045, "v_current": 0.022,
            "u_wind": 2.2, "v_wind": 1.1,
            "u_stokes": 0.026, "v_stokes": 0.013,
        }

        u_drift = v["u_current"] + WIND_LEEWAY_FACTOR * v["u_wind"] + v.get("u_stokes", 0.0)
        v_drift = v["v_current"] + WIND_LEEWAY_FACTOR * v["v_wind"] + v.get("v_stokes", 0.0)

        # Backward integration: reverse drift vector
        dlat = (v_drift * dt_sec) / METERS_PER_DEG_LAT
        dlon = (u_drift * dt_sec) / (METERS_PER_DEG_LAT * max(math.cos(math.radians(cur_lat)), 1e-6))

        cur_lat -= dlat
        cur_lon -= dlon

        dist_from_detection = _haversine_km(lat, lon, cur_lat, cur_lon)

        # Spatial intersection check at this step against River Outlets
        for river in RIVER_OUTLETS:
            d_riv = _haversine_km(cur_lat, cur_lon, river["lat"], river["lon"])
            if d_riv <= 25.0:
                intersection_candidates.append({
                    "step_hour": step,
                    "target": river,
                    "type": river.get("type", "RIVER_RUNOFF"),
                    "distance_km": round(d_riv, 2),
                    "latitude": cur_lat,
                    "longitude": cur_lon,
                })

        # Spatial intersection check at this step against Shipping Corridors
        for corr in SHIPPING_CORRIDORS:
            d_corr = _haversine_km(cur_lat, cur_lon, corr["lat"], corr["lon"])
            if d_corr <= 35.0:
                intersection_candidates.append({
                    "step_hour": step,
                    "target": corr,
                    "type": corr.get("type", "VESSEL_DISCHARGE"),
                    "distance_km": round(d_corr, 2),
                    "latitude": cur_lat,
                    "longitude": cur_lon,
                })

        # Record interval nodes (every 6h or key milestones)
        if step % 6 == 0 or step == 1 or step == hours:
            backtrack_path.append({
                "hour": -step,
                "latitude": round(cur_lat, 5),
                "longitude": round(cur_lon, 5),
                "u_drift": round(u_drift, 4),
                "v_drift": round(v_drift, 4),
                "distance_from_detection_km": round(dist_from_detection, 2),
            })

    origin_lat = round(cur_lat, 5)
    origin_lon = round(cur_lon, 5)

    # Find closest reference river outlet from backtrack origin
    best_river = None
    min_river_dist = float("inf")
    for r in RIVER_OUTLETS:
        d = _haversine_km(origin_lat, origin_lon, r["lat"], r["lon"])
        if d < min_river_dist:
            min_river_dist = d
            best_river = r

    # Find closest reference shipping corridor from backtrack origin
    best_corridor = None
    min_corridor_dist = float("inf")
    for c in SHIPPING_CORRIDORS:
        d = _haversine_km(origin_lat, origin_lon, c["lat"], c["lon"])
        if d < min_corridor_dist:
            min_corridor_dist = d
            best_corridor = c

    # Sort intersection candidates along trajectory
    closest_candidate = None
    if intersection_candidates:
        intersection_candidates.sort(key=lambda x: x["distance_km"])
        closest_candidate = intersection_candidates[0]

    # Source Attribution Classification
    if min_river_dist <= 15.0 or (closest_candidate and closest_candidate["type"] in ("RIVER_RUNOFF", "COASTAL_OUTFLOW") and closest_candidate["distance_km"] <= 8.0):
        matched = best_river if min_river_dist <= 15.0 else closest_candidate["target"]
        dist = min_river_dist if min_river_dist <= 15.0 else closest_candidate["distance_km"]
        step_h = hours if min_river_dist <= 15.0 else closest_candidate["step_hour"]
        origin_type = matched.get("type", "RIVER_RUNOFF")
        conf = max(0.70, min(0.96, 1.0 - (dist / 30.0)))
        nearest_source = {
            "name": matched["name"],
            "type": origin_type,
            "region": matched.get("region", "Coastal"),
            "distance_km": round(dist, 2),
            "latitude": matched["lat"],
            "longitude": matched["lon"],
        }
        description = (
            f"Reverse Lagrangian backtracking traces debris origin to within {dist:.1f} km of {matched['name']} "
            f"({origin_type}). High correlation with urban riverine runoff and monsoon drainage outflow."
        )
    elif min_corridor_dist <= 25.0 or (closest_candidate and closest_candidate["type"] == "VESSEL_DISCHARGE" and closest_candidate["distance_km"] <= 12.0):
        matched = best_corridor if min_corridor_dist <= 25.0 else closest_candidate["target"]
        dist = min_corridor_dist if min_corridor_dist <= 25.0 else closest_candidate["distance_km"]
        step_h = hours if min_corridor_dist <= 25.0 else closest_candidate["step_hour"]
        origin_type = "VESSEL_DISCHARGE"
        conf = max(0.68, min(0.94, 1.0 - (dist / 40.0)))
        nearest_source = {
            "name": matched["name"],
            "type": "VESSEL_DISCHARGE",
            "traffic": matched.get("traffic", "Commercial Shipping"),
            "distance_km": round(dist, 2),
            "latitude": matched["lat"],
            "longitude": matched["lon"],
        }
        description = (
            f"Backtrack trajectory intersects {matched['name']} at {dist:.1f} km approach. "
            f"Classified as commercial vessel discharge / maritime container loss. AIS cross-referencing recommended."
        )
    elif min_river_dist <= 25.0:
        origin_type = "COASTAL_OUTFLOW"
        conf = 0.74
        step_h = hours
        nearest_source = {
            "name": best_river["name"] if best_river else "Coastal Outflow",
            "type": "COASTAL_OUTFLOW",
            "distance_km": round(min_river_dist, 2),
            "latitude": best_river["lat"] if best_river else origin_lat,
            "longitude": best_river["lon"] if best_river else origin_lon,
        }
        description = (
            f"Backtracked origin point is located on coastal shelf {min_river_dist:.1f} km from "
            f"{best_river['name'] if best_river else 'shoreline'}. Classified as coastal outflow / nearshore discharge."
        )
    else:
        origin_type = "OFFSHORE_SPILL"
        conf = 0.85
        step_h = hours
        closer = best_corridor if min_corridor_dist < min_river_dist else best_river
        closer_dist = min(min_corridor_dist, min_river_dist)
        nearest_source = {
            "name": closer["name"] if closer else "Open Ocean Deepwater",
            "type": "OFFSHORE_SPILL",
            "distance_km": round(closer_dist, 2) if closer else 50.0,
            "latitude": closer["lat"] if closer else origin_lat,
            "longitude": closer["lon"] if closer else origin_lon,
        }
        description = (
            f"Backtracked origin is in open deepwater ({closer_dist:.1f} km from {closer['name'] if closer else 'corridor'}). "
            f"Classified as offshore marine spill or mid-voyage maritime discharge."
        )

    discharge_dt = datetime.now(timezone.utc) - timedelta(hours=step_h)
    discharge_ts = discharge_dt.isoformat()
    total_backtrack_dist = _haversine_km(lat, lon, origin_lat, origin_lon)

    return {
        "status": "success",
        "origin_type": origin_type,
        "source_type": origin_type,
        "confidence_score": round(conf, 3),
        "origin_coordinates": {
            "latitude": origin_lat,
            "longitude": origin_lon,
        },
        "primary_origin": {
            "latitude": origin_lat,
            "longitude": origin_lon,
        },
        "backtrack_path": backtrack_path,
        "nearest_source": nearest_source,
        "nearest_river_mouth": best_river["name"] if best_river else None,
        "river_mouth_dist_km": round(min_river_dist, 2) if best_river else None,
        "discharge_timestamp_estimate": discharge_ts,
        "backtrack_hours": hours,
        "total_backtrack_distance_km": round(total_backtrack_dist, 2),
        "data_source": "open-meteo-historical-marine" if data_available else "open-meteo-fallback",
        "description": description,
    }

