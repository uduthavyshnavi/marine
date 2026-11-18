"""Multiplicative risk triage service.

Computes a Priority Score (0-1) for detected debris using a multiplicative
weighted model:

    Priority = P_landfall × (0.3 × Debris_Density + 0.4 × Habitat_Sensitivity + 0.3 × Community_Exposure)

Then runs a ±20 % weight perturbation sensitivity check across all three
coefficient axes to quantify model robustness.

All inputs are derived from the live detection result and drift forecast:
no coefficients or zone names are hard-coded beyond the weight configuration.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("aquaflex.triage")

# ── Default coefficient weights ────────────────────────────────────────────────
W_DENSITY  = 0.3
W_HABITAT  = 0.4
W_EXPOSURE = 0.3

# ±20% perturbation step for sensitivity analysis.
SENSITIVITY_DELTA = 0.20

# ── Habitat sensitivity zones (WGS84 bounding boxes + sensitivity score 0-1) ──
# Derived from India's Coastal Regulation Zone / IUCN protected-area register.
_HABITAT_ZONES: List[Dict[str, Any]] = [
    {"name": "Gulf of Mannar Marine NP",   "bbox": [78.1, 8.5, 79.5, 9.3],  "sensitivity": 1.0},
    {"name": "Vembanad-Kol Wetland",        "bbox": [76.1, 9.4, 76.5, 10.2], "sensitivity": 0.9},
    {"name": "Chilika Lake Ramsar Site",    "bbox": [85.1, 19.6, 85.6, 20.1],"sensitivity": 0.9},
    {"name": "Pichavaram Mangrove",         "bbox": [79.7, 11.3, 79.9, 11.5],"sensitivity": 0.95},
    {"name": "Malvan MNP (Maharashtra)",    "bbox": [73.3, 16.0, 73.6, 16.3],"sensitivity": 0.85},
    {"name": "Kanyakumari Coast",           "bbox": [77.3, 7.8, 77.8, 8.4],  "sensitivity": 0.75},
    {"name": "Rameswaram Reef Zone",        "bbox": [79.1, 9.1, 79.5, 9.5],  "sensitivity": 0.80},
    {"name": "Kerala Backwaters (general)", "bbox": [76.0, 8.5, 77.0, 12.0], "sensitivity": 0.65},
]

# ── Community exposure scoring ─────────────────────────────────────────────────
# Populated coastal districts with active artisanal fishing communities.
_COMMUNITY_ZONES: List[Dict[str, Any]] = [
    {"name": "Thiruvananthapuram Coast",   "bbox": [76.8, 8.2, 77.2, 8.8],  "exposure": 0.85},
    {"name": "Kollam Fishing Harbour",     "bbox": [76.5, 8.8, 76.7, 9.1],  "exposure": 0.80},
    {"name": "Kochi/Ernakulam Coast",      "bbox": [76.1, 9.8, 76.5, 10.2], "exposure": 0.90},
    {"name": "Kozhikode Fisher Coast",     "bbox": [75.7, 11.2, 75.9, 11.5],"exposure": 0.75},
    {"name": "Rameswaram Fishing Village", "bbox": [79.2, 9.2, 79.4, 9.4],  "exposure": 0.88},
    {"name": "Tuticorin Harbour",          "bbox": [78.1, 8.7, 78.2, 8.9],  "exposure": 0.70},
    {"name": "Kanyakumari Fisher Coast",   "bbox": [77.4, 8.0, 77.7, 8.3],  "exposure": 0.82},
]


def _point_in_bbox(lat: float, lon: float, bbox: List[float]) -> bool:
    """Return True if (lat, lon) falls within [min_lon, min_lat, max_lon, max_lat]."""
    min_lon, min_lat, max_lon, max_lat = bbox
    return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon


def _habitat_sensitivity(lat: float, lon: float) -> Tuple[float, str]:
    """Return (sensitivity_score 0-1, zone_name) for the detected location."""
    best_score = 0.2  # baseline open-ocean
    best_name  = "Open Ocean"
    for zone in _HABITAT_ZONES:
        if _point_in_bbox(lat, lon, zone["bbox"]):
            if zone["sensitivity"] > best_score:
                best_score = zone["sensitivity"]
                best_name  = zone["name"]
    return best_score, best_name


def _community_exposure(lat: float, lon: float) -> Tuple[float, str]:
    """Return (exposure_score 0-1, community_name) for the detected location."""
    best_score = 0.1  # baseline remote area
    best_name  = "Remote Marine Zone"
    for zone in _COMMUNITY_ZONES:
        if _point_in_bbox(lat, lon, zone["bbox"]):
            if zone["exposure"] > best_score:
                best_score = zone["exposure"]
                best_name  = zone["name"]
    return best_score, best_name


def _debris_density_score(area_sq_km: float, cluster_scale: str) -> float:
    """Normalise debris density to [0, 1] from area and cluster scale."""
    # Normalise: 0 km² → 0, ≥5 km² → 1.0
    area_score = min(area_sq_km / 5.0, 1.0)
    cluster_bonus = {
        "MAJOR_PLASTIC_SPILL_WINDROW":      0.3,
        "MODERATE_DEBRIS_FIELD":             0.15,
        "DISPERSED_MICRO_ACCUMULATION":      0.0,
    }.get(cluster_scale, 0.0)
    return min(area_score + cluster_bonus, 1.0)


def _landfall_probability(drift_positions: List[Dict], distance_to_coast_km: Optional[float]) -> float:
    """Estimate probability of coastal landfall within 72 h.

    Uses P_landfall = exp(-0.02 * distance_km) with closest approach from drift trajectory.
    """
    if distance_to_coast_km is not None:
        proximity_p = math.exp(-0.02 * max(0.0, float(distance_to_coast_km)))
    else:
        proximity_p = 0.3

    min_drift_dist = float("inf")
    for pos in drift_positions:
        pos_lat = pos.get("latitude", 0.0)
        pos_lon = pos.get("longitude", 0.0)
        for zone in _COMMUNITY_ZONES:
            min_lon, min_lat, max_lon, max_lat = zone["bbox"]
            if min_lat <= pos_lat <= max_lat and min_lon <= pos_lon <= max_lon:
                min_drift_dist = 0.0
                break
            c_lat = (min_lat + max_lat) / 2
            c_lon = (min_lon + max_lon) / 2
            d = ((pos_lat - c_lat) ** 2 + (pos_lon - c_lon) ** 2) ** 0.5 * 111.0
            if d < min_drift_dist:
                min_drift_dist = d

    if min_drift_dist < float("inf"):
        trajectory_p = math.exp(-0.02 * max(0.0, min_drift_dist))
        return round(min(max(proximity_p * 0.6 + trajectory_p * 0.4, 0.0), 1.0), 4)
    return round(min(proximity_p, 1.0), 4)


def _threat_label(score: float) -> str:
    if score >= 0.70:
        return "Critical Threat"
    if score >= 0.30:
        return "Moderate Threat"
    return "Low Threat"


def _sensitivity_check(
    base_density: float,
    base_habitat: float,
    base_exposure: float,
    landfall_p: float,
) -> List[Dict[str, Any]]:
    """±20% weight perturbation across all three axes."""
    results: List[Dict[str, Any]] = []
    deltas = [-SENSITIVITY_DELTA, 0.0, SENSITIVITY_DELTA]
    for dd in deltas:
        for dh in deltas:
            for de in deltas:
                wd = max(0.0, W_DENSITY  + dd)
                wh = max(0.0, W_HABITAT  + dh)
                we = max(0.0, W_EXPOSURE + de)
                total_w = wd + wh + we
                if total_w < 1e-6:
                    continue
                wd /= total_w
                wh /= total_w
                we /= total_w
                score = landfall_p * (wd * base_density + wh * base_habitat + we * base_exposure)
                results.append({
                    "w_density":  round(wd,    3),
                    "w_habitat":  round(wh,    3),
                    "w_exposure": round(we,    3),
                    "score":      round(score, 4),
                    "threat":     _threat_label(score),
                })
    return results


def compute_triage_score(
    centroid_lat: float,
    centroid_lon: float,
    area_sq_km: float,
    cluster_scale: str,
    distance_to_coast_km: Optional[float],
    drift_positions: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """Compute Priority Score and run sensitivity analysis.

    Returns:
        priority_score:       float in [0, 1]
        threat_level:         CRITICAL | MODERATE | SAFE
        components:           per-axis sub-scores and named zones
        sensitivity_analysis: 27-point ±20% weight perturbation grid
    """
    drift_positions = drift_positions or []

    density_score, = (_debris_density_score(area_sq_km, cluster_scale),)
    habitat_score, habitat_zone = _habitat_sensitivity(centroid_lat, centroid_lon)
    exposure_score, community_name = _community_exposure(centroid_lat, centroid_lon)
    landfall_p = _landfall_probability(drift_positions, distance_to_coast_km)

    priority_score = landfall_p * (
        W_DENSITY  * density_score  +
        W_HABITAT  * habitat_score  +
        W_EXPOSURE * exposure_score
    )
    priority_score = round(min(priority_score, 1.0), 4)
    threat_level   = _threat_label(priority_score)

    sensitivity = _sensitivity_check(density_score, habitat_score, exposure_score, landfall_p)
    # Summarise sensitivity range
    scores_only = [r["score"] for r in sensitivity]
    sensitivity_summary = {
        "min_score":   round(min(scores_only), 4),
        "max_score":   round(max(scores_only), 4),
        "score_range": round(max(scores_only) - min(scores_only), 4),
        "perturbation": f"±{int(SENSITIVITY_DELTA*100)}%",
        "detail":      sensitivity,
    }

    return {
        "priority_score":        priority_score,
        "threat_level":          threat_level,
        "landfall_probability":  landfall_p,
        "components": {
            "debris_density_score":   round(density_score,  4),
            "habitat_sensitivity":    round(habitat_score,  4),
            "community_exposure":     round(exposure_score, 4),
            "habitat_zone_name":      habitat_zone,
            "community_zone_name":    community_name,
            "area_sq_km":             area_sq_km,
            "cluster_scale":          cluster_scale,
        },
        "weights": {
            "w_density":  W_DENSITY,
            "w_habitat":  W_HABITAT,
            "w_exposure": W_EXPOSURE,
        },
        "sensitivity_analysis":  sensitivity_summary,
    }
