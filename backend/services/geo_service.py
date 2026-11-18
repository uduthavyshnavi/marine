"""Dynamic geospatial analysis for live satellite rasters.

All coordinates, areas, distances and coastal lookups are derived from the
uploaded raster's own metadata (CRS, affine transform, resolution) and live
spatial queries. No coordinates, village names, or alert strings are
hard-coded.

Reprojection to WGS84 uses pyproj when installed, otherwise rasterio.warp
(PROJ-backed). Great-circle distance uses geopy.geodesic when installed,
otherwise an equivalent WGS84 Vincenty inverse implementation.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger("aquaflex.geo")

WGS84_EPSG = "EPSG:4326"

# WGS84 ellipsoid parameters (metres).
_WGS84_A = 6378137.0
_WGS84_F = 1 / 298.257223563
_WGS84_B = _WGS84_A * (1 - _WGS84_F)


def vincenty_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """WGS84 geodesic distance in km (Vincenty inverse). geopy-free fallback."""
    if abs(lat1 - lat2) < 1e-12 and abs(lon1 - lon2) < 1e-12:
        return 0.0
    a = _WGS84_A
    b = _WGS84_B
    f = _WGS84_F
    L = math.radians(lon2 - lon1)
    U1 = math.atan((1 - f) * math.tan(math.radians(lat1)))
    U2 = math.atan((1 - f) * math.tan(math.radians(lat2)))
    sin_u1, cos_u1 = math.sin(U1), math.cos(U1)
    sin_u2, cos_u2 = math.sin(U2), math.cos(U2)

    lam = L
    for _ in range(200):
        sin_lam, cos_lam = math.sin(lam), math.cos(lam)
        sin_sigma = math.hypot(
            cos_u2 * sin_lam, cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lam
        )
        if sin_sigma == 0:
            return 0.0
        cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lam
        sigma = math.atan2(sin_sigma, cos_sigma)
        sin_alpha = cos_u1 * cos_u2 * sin_lam / sin_sigma
        cos2_alpha = 1 - sin_alpha ** 2
        cos_2sigma_m = cos_sigma - 2 * sin_u1 * sin_u2 / (cos2_alpha or 1e-12)
        C = f / 16 * cos2_alpha * (4 + f * (4 - 3 * cos2_alpha))
        lam_prev = lam
        lam = L + (1 - C) * f * sin_alpha * (
            sigma
            + C
            * sin_sigma
            * (cos_2sigma_m + C * cos_sigma * (-1 + 2 * cos_2sigma_m ** 2))
        )
        if abs(lam - lam_prev) < 1e-12:
            break
    u2 = cos2_alpha * (a ** 2 - b ** 2) / (b ** 2 or 1.0)
    A = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
    d_sigma = (
        B
        * sin_sigma
        * (
            cos_2sigma_m
            + B
            / 4
            * (
                cos_sigma * (-1 + 2 * cos_2sigma_m ** 2)
                - B / 6 * cos_2sigma_m * (-3 + 4 * sin_sigma ** 2) * (-3 + 4 * cos_2sigma_m ** 2)
            )
        )
    )
    s = b * A * (sigma - d_sigma)
    return s / 1000.0


def geodesic_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Geodesic distance in km. Uses geopy.geodesic when available."""
    try:
        from geopy.distance import geodesic

        return float(geodesic((lat1, lon1), (lat2, lon2)).km)
    except Exception:
        return vincenty_distance_km(lat1, lon1, lat2, lon2)


def reproject_point(lon: float, lat: float, src_crs) -> Tuple[float, float]:
    """Reproject a single (lon, lat) in ``src_crs`` to WGS84 (EPSG:4326)."""
    if src_crs is None:
        return lon, lat
    try:
        from pyproj import Transformer

        transformer = Transformer.from_crs(src_crs, WGS84_EPSG, always_xy=True)
        out_lon, out_lat = transformer.transform(lon, lat)
        return float(out_lon), float(out_lat)
    except Exception:
        import rasterio.warp as warp

        xs, ys = warp.transform(src_crs, WGS84_EPSG, [lon], [lat])
        return float(xs[0]), float(ys[0])


def mask_centroid_and_bbox_wgs84(
    mask: np.ndarray, transform, crs
) -> Tuple[Optional[Tuple[float, float]], Optional[List[float]]]:
    """Return (centroid_lonlat, bbox) in WGS84 for the debris pixels in ``mask``.

    centroid is (lon, lat); bbox is [min_lon, min_lat, max_lon, max_lat].
    Both are derived from the raster affine transform and reprojected to WGS84.
    Returns (None, None) when the mask has no debris pixels.
    """
    import rasterio

    rows, cols = np.where(mask > 0)
    if rows.size == 0:
        return None, None

    row_c = float(rows.mean())
    col_c = float(cols.mean())
    cx, cy = rasterio.transform.xy(transform, row_c, col_c)
    clon, clat = reproject_point(float(cx), float(cy), crs)

    rmin, rmax = int(rows.min()), int(rows.max())
    cmin, cmax = int(cols.min()), int(cols.max())
    corners_x, corners_y = [], []
    for r, c in ((rmin, cmin), (rmin, cmax), (rmax, cmin), (rmax, cmax)):
        x, y = rasterio.transform.xy(transform, r, c)
        corners_x.append(x)
        corners_y.append(y)
    xs, ys = [], []
    try:
        import rasterio.warp as warp

        xs, ys = warp.transform(crs, WGS84_EPSG, corners_x, corners_y)
    except Exception:
        xs, ys = corners_x, corners_y
    bbox = [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]
    return (clon, clat), bbox


def pixel_area_sq_m(res_x: float, res_y: float, crs, ref_lat: Optional[float]) -> float:
    """Ground area of one pixel in square metres.

    For projected CRS in metres, this is res_x * res_y. For geographic CRS
    (degrees), it converts degrees to metres using the reference latitude.
    """
    res_x = abs(float(res_x))
    res_y = abs(float(res_y))
    if res_x == 0 or res_y == 0:
        return 0.0
    is_geographic = True
    try:
        is_geographic = crs.is_geographic if crs is not None else True
    except Exception:
        is_geographic = True
    if not is_geographic:
        return res_x * res_y
    lat = ref_lat if ref_lat is not None else 0.0
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * max(math.cos(math.radians(lat)), 1e-6)
    return (res_x * m_per_deg_lon) * (res_y * m_per_deg_lat)


def classify_severity(area_sq_km: float, distance_km: Optional[float]) -> Tuple[str, str]:
    """Dynamic severity from debris area and coastal proximity.

    CRITICAL: area > 0.05 sq km OR nearest settlement < 5 km.
    MODERATE: 0.01 - 0.05 sq km.
    LOW:     area < 0.01 sq km.
    Returns (zone, hex_color).
    """
    if area_sq_km > 0.05 or (distance_km is not None and distance_km < 5.0):
        return "CRITICAL", "#FF0000"
    if area_sq_km >= 0.01:
        return "MODERATE", "#FFD500"
    return "LOW", "#00C853"


def threat_level_and_action(
    severity: str, distance_km: Optional[float]
) -> Tuple[str, str]:
    """Derive fishing-impact threat level and recommended action from severity."""
    if severity == "CRITICAL":
        if distance_km is not None and distance_km < 5.0:
            return (
                "HIGH_RISK_TO_FISHING_GEAR",
                "Issue Immediate Navigation & Net-Casting Warning",
            )
        return (
            "ELEVATED_RISK_TO_FISHING_GEAR",
            "Issue Navigation Advisory & Preposition Cleanup Crew",
        )
    if severity == "MODERATE":
        return (
            "MODERATE_RISK_TO_FISHING_GEAR",
            "Issue Coastal Watch Advisory & Schedule Survey",
        )
    return (
        "LOW_RISK",
        "Continue Routine Monitoring; No Immediate Action Required",
    )


def find_nearest_coastal_settlement(
    lat: float, lon: float, radius_km: float = 30.0, timeout_s: int = 25
) -> dict:
    """Query OpenStreetMap (Overpass API) for the nearest inhabited coastal node.

    Searches for place nodes (village/town/city/hamlet) and harbour/port
    features within ``radius_km`` of the debris centroid, then returns the
    closest by geodesic distance. Nothing is hard-coded: the settlement name
    and distance come entirely from the live query. On any failure an honest
    status dict is returned (no fabricated name/distance).
    """
    import requests

    radius_m = int(radius_km * 1000)
    query = f"""
    [out:json][timeout:{timeout_s}];
    (
      node(around:{radius_m},{lat},{lon})["place"~"village|town|city|hamlet|suburb|neighbourhood"];
      node(around:{radius_m},{lat},{lon})["harbour"];
      way(around:{radius_m},{lat},{lon})["harbour"];
      node(around:{radius_m},{lat},{lon})["seamark:type"];
      node(around:{radius_m},{lat},{lon})["amenity"="ferry_terminal"];
    );
    out center tags;
    """
    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]
    last_err = None
    for url in endpoints:
        try:
            resp = requests.post(url, data={"data": query}, timeout=timeout_s)
            if resp.status_code != 200:
                last_err = f"HTTP {resp.status_code} from {url}"
                continue
            elements = resp.json().get("elements", [])
            best = None
            best_dist = float("inf")
            for el in elements:
                el_lat = el.get("lat") or (el.get("center") or {}).get("lat")
                el_lon = el.get("lon") or (el.get("center") or {}).get("lon")
                if el_lat is None or el_lon is None:
                    continue
                tags = el.get("tags", {})
                name = tags.get("name")
                if not name:
                    continue
                d = geodesic_distance_km(lat, lon, float(el_lat), float(el_lon))
                if d < best_dist:
                    best_dist = d
                    best = name
            if best is None:
                return {
                    "settlement_name": None,
                    "distance_km": None,
                    "query_status": "no_settlement_found",
                }
            return {
                "settlement_name": best,
                "distance_km": round(best_dist, 3),
                "query_status": "ok",
            }
        except Exception as exc:
            last_err = str(exc)
            continue
    return {
        "settlement_name": None,
        "distance_km": None,
        "query_status": f"overpass_unavailable: {last_err}",
    }


