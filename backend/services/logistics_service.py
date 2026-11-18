"""Autonomous Cleanup Fleet Intercept & Route Logistics Engine.

Models major coastal response bases, computes optimal dynamic intercept waypoints
along the 72-hour Lagrangian drift path, and estimates vessel fleet requirements,
transit times, and operational fuel consumption.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("aquaflex.logistics")

KM_PER_NAUTICAL_MILE = 1.852
STANDARD_SKIMMER_SPEED_KNOTS = 12.0
FUEL_LITERS_PER_NM_PER_VESSEL = 45.0
SKIMMER_CAPACITY_KM2 = 0.25

# Key coastal base ports along the Indian coastline
COASTAL_BASE_PORTS: List[Dict[str, Any]] = [
    {
        "name": "Kochi Port",
        "lat": 9.9600,
        "lon": 76.2200,
        "region": "Arabian Sea / Kerala",
        "fleet_available": 4,
    },
    {
        "name": "Visakhapatnam Port",
        "lat": 17.6800,
        "lon": 83.2100,
        "region": "Bay of Bengal / Andhra Pradesh",
        "fleet_available": 5,
    },
    {
        "name": "Chennai Port",
        "lat": 13.0800,
        "lon": 80.2900,
        "region": "Coromandel Coast / Tamil Nadu",
        "fleet_available": 3,
    },
    {
        "name": "Tuticorin Port",
        "lat": 8.7500,
        "lon": 78.1800,
        "region": "Gulf of Mannar / Tamil Nadu",
        "fleet_available": 3,
    },
    {
        "name": "Mangalore Port",
        "lat": 12.9200,
        "lon": 74.8100,
        "region": "Karnataka Coast",
        "fleet_available": 2,
    },
]


def _haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers."""
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def find_nearest_port(center_lat: float, center_lon: float) -> Tuple[Dict[str, Any], float]:
    """Find the closest home port to the given coordinates. Returns (port_dict, distance_km)."""
    closest_port = COASTAL_BASE_PORTS[0]
    min_dist_km = float("inf")

    for port in COASTAL_BASE_PORTS:
        d_km = _haversine_distance_km(center_lat, center_lon, port["lat"], port["lon"])
        if d_km < min_dist_km:
            min_dist_km = d_km
            closest_port = port

    return closest_port, min_dist_km


def plan_fleet_intercept(
    center_lat: float,
    center_lon: float,
    debris_area_km2: float = 0.05,
    drift_positions: Optional[List[Dict[str, Any]]] = None,
    skimmer_speed_knots: float = STANDARD_SKIMMER_SPEED_KNOTS,
) -> Dict[str, Any]:
    """Compute optimal vessel intercept waypoint, fleet allocation, and fuel logistics.

    Parameters
    ----------
    center_lat : float
        Current debris centroid latitude (WGS84).
    center_lon : float
        Current debris centroid longitude (WGS84).
    debris_area_km2 : float
        Estimated debris surface area in square kilometers.
    drift_positions : Optional[List[Dict[str, Any]]]
        Projected 72h drift trajectory positions (e.g. from drift_service).
    skimmer_speed_knots : float
        Vessel cruising speed in knots (default: 12.0).

    Returns
    -------
    Dict[str, Any]
        Structured intercept logistics plan.
    """
    nearest_port, initial_dist_km = find_nearest_port(center_lat, center_lon)
    port_lat, port_lon = nearest_port["lat"], nearest_port["lon"]

    # Trajectory positions fallback if none provided
    if not drift_positions:
        drift_positions = [
            {"hour": 6, "latitude": round(center_lat + 0.03, 4), "longitude": round(center_lon + 0.03, 4)},
            {"hour": 12, "latitude": round(center_lat + 0.06, 4), "longitude": round(center_lon + 0.06, 4)},
            {"hour": 24, "latitude": round(center_lat + 0.12, 4), "longitude": round(center_lon + 0.12, 4)},
            {"hour": 48, "latitude": round(center_lat + 0.25, 4), "longitude": round(center_lon + 0.25, 4)},
            {"hour": 72, "latitude": round(center_lat + 0.38, 4), "longitude": round(center_lon + 0.38, 4)},
        ]

    # Evaluate each waypoint to find the earliest interceptable coordinate
    best_waypoint: Optional[Dict[str, Any]] = None
    best_dist_nm: float = initial_dist_km / KM_PER_NAUTICAL_MILE
    best_transit_hours: float = best_dist_nm / skimmer_speed_knots

    for pos in drift_positions:
        hour = pos.get("hour", 12)
        wp_lat = float(pos.get("latitude", center_lat))
        wp_lon = float(pos.get("longitude", center_lon))
        d_km = _haversine_distance_km(port_lat, port_lon, wp_lat, wp_lon)
        d_nm = d_km / KM_PER_NAUTICAL_MILE
        transit_hrs = d_nm / max(1.0, skimmer_speed_knots)

        # Vessel can intercept if transit time is less than or equal to forecast arrival hour
        if transit_hrs <= hour:
            best_waypoint = {"hour": hour, "lat": round(wp_lat, 4), "lon": round(wp_lon, 4)}
            best_dist_nm = round(d_nm, 1)
            best_transit_hours = round(transit_hrs, 2)
            break

    # If no waypoint meets the condition, select the T+12h or closest waypoint
    if best_waypoint is None:
        target_pos = next((p for p in drift_positions if p.get("hour") == 12), drift_positions[0])
        wp_lat = float(target_pos.get("latitude", center_lat))
        wp_lon = float(target_pos.get("longitude", center_lon))
        d_km = _haversine_distance_km(port_lat, port_lon, wp_lat, wp_lon)
        best_dist_nm = round(d_km / KM_PER_NAUTICAL_MILE, 1)
        best_transit_hours = round(best_dist_nm / max(1.0, skimmer_speed_knots), 2)
        best_waypoint = {
            "hour": target_pos.get("hour", 12),
            "lat": round(wp_lat, 4),
            "lon": round(wp_lon, 4),
        }

    # Fleet sizing: 1 skimmer per 0.25 sq km of debris, minimum 1
    required_skimmers = max(1, math.ceil(debris_area_km2 / SKIMMER_CAPACITY_KM2))
    # Fuel estimate: distance (NM) * 45 L/NM * number of vessels
    estimated_fuel_liters = round(best_dist_nm * FUEL_LITERS_PER_NM_PER_VESSEL * required_skimmers, 1)

    return {
        "status": "success",
        "nearest_port": nearest_port["name"],
        "port_coords": [port_lat, port_lon],
        "port_distance_nm": best_dist_nm,
        "intercept_waypoint": best_waypoint,
        "eta_hours": best_transit_hours,
        "transit_speed_knots": skimmer_speed_knots,
        "required_skimmers": required_skimmers,
        "estimated_fuel_liters": estimated_fuel_liters,
        "region": nearest_port.get("region", "Coastal Waters"),
        "nearest_base_port": {
            "name": nearest_port["name"],
            "state": nearest_port.get("region", "Coastal Waters"),
            "latitude": port_lat,
            "longitude": port_lon,
            "agency": "Indian Coast Guard",
        },
        "geodesic_distance": {
            "nautical_miles": best_dist_nm,
            "kilometers": round(best_dist_nm * KM_PER_NAUTICAL_MILE, 1),
        },
        "intercept_solution": {
            "target_hour": best_waypoint["hour"],
            "intercept_latitude": best_waypoint["lat"],
            "intercept_longitude": best_waypoint["lon"],
            "vessel_transit_hours": best_transit_hours,
            "cruise_speed_knots": skimmer_speed_knots,
        },
        "fleet_recommendation": {
            "skimmer_vessels_count": required_skimmers,
            "coverage_capacity_km2": round(required_skimmers * SKIMMER_CAPACITY_KM2, 2),
            "fuel_consumption_liters": estimated_fuel_liters,
            "estimated_fuel_cost_inr": round(estimated_fuel_liters * 90.0),
        },
    }
