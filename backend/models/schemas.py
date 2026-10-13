"""Pydantic schemas for AquaFlex AI request/response validation.

Defines the structured contracts exchanged between the FastAPI backend and the
GIS dashboard. Every detection carries an explicit confidence score, every
trajectory point carries a forecast hour and risk-severity status, and the full
simulation payload bundles case-study metadata, the initial detection centroid,
the predicted trajectory, and priority coastal cleanup zones.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class DebrisClass(str, Enum):
    """Supported floating-marine-plastic debris categories."""

    MACRO_PLASTIC = "Macro-Plastic"
    NURDLES = "Nurdles"
    MIXED = "Macro-Plastic / Nurdles"


class RiskSeverity(str, Enum):
    """Coastal impact risk severity ladder for trajectory points and zones."""

    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    CRITICAL = "Critical"


class Coordinate2D(BaseModel):
    """A geographic coordinate (longitude, latitude) in WGS84 (EPSG:4326)."""

    lon: float = Field(..., description="Longitude in decimal degrees.")
    lat: float = Field(..., description="Latitude in decimal degrees.")


class DebrisDetection(BaseModel):
    """A single AI/ML detection of potential floating marine plastic debris.

    bbox is the pixel-space box [xmin, ymin, xmax, ymax]; confidence is in
    [0, 1]; confidence_percent is confidence * 100; centroid is the optional
    geographic centroid; area_px is the bounding-box area in square pixels.
    """

    bbox: List[float] = Field(
        ..., min_length=4, max_length=4,
        description="Bounding box [xmin, ymin, xmax, ymax] in pixel coordinates.",
    )
    label: str = Field(..., description="Debris class ('Macro-Plastic' or 'Nurdles').")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Detection confidence in [0, 1]."
    )
    confidence_percent: float = Field(
        ..., ge=0.0, le=100.0, description="Detection confidence as a percentage (0-100)."
    )
    centroid: Optional[Coordinate2D] = Field(
        default=None, description="Geographic centroid (lon, lat), if geo-referenced."
    )
    area_px: Optional[float] = Field(
        default=None, ge=0.0, description="Bounding-box area in square pixels."
    )


class GeoJSONFeature(BaseModel):
    """A GeoJSON Feature carrying a detection's geometry and properties."""

    type: str = Field(default="Feature", description="GeoJSON geometry type.")
    geometry: dict = Field(
        ..., description="GeoJSON geometry (Point centroid, Polygon hotspot)."
    )
    properties: dict = Field(
        ..., description="Feature properties: label, confidence, confidence_percent."
    )


class GeoJSONFeatureCollection(BaseModel):
    """A GeoJSON FeatureCollection of debris detection features."""

    type: str = Field(default="FeatureCollection")
    features: List[GeoJSONFeature] = Field(default_factory=list)


class DetectionResponse(BaseModel):
    """Response payload for the debris detection endpoint."""

    source_image: str = Field(..., description="Path/identifier of the processed tile.")
    image_size: Optional[List[int]] = Field(
        default=None, min_length=2, max_length=2, description="[width, height] px."
    )
    detection_count: int = Field(..., ge=0, description="Number of retained detections.")
    mean_confidence: float = Field(
        ..., ge=0.0, le=100.0, description="Mean confidence across detections (%)."
    )
    detections: List[DebrisDetection] = Field(default_factory=list)
    geojson: GeoJSONFeatureCollection = Field(
        ..., description="GeoJSON FeatureCollection of the detections."
    )


class TrajectoryPoint(BaseModel):
    """A single forecast position of a drifting debris particle/seep."""

    lat: float = Field(..., description="Latitude in decimal degrees.")
    lon: float = Field(..., description="Longitude in decimal degrees.")
    hour: int = Field(..., description="Forecast lead time in hours (24/48/72).")
    u: Optional[float] = Field(default=None, description="Eastward current (m/s).")
    v: Optional[float] = Field(default=None, description="Northward current (m/s).")
    risk_status: RiskSeverity = Field(
        default=RiskSeverity.LOW, description="Coastal impact risk severity here."
    )


class Trajectory(BaseModel):
    """An ordered forecast trajectory for a single debris release/centroid."""

    particle_id: str = Field(..., description="Identifier for the tracked seep.")
    origin_lat: float = Field(..., description="Release latitude.")
    origin_lon: float = Field(..., description="Release longitude.")
    points: List[TrajectoryPoint] = Field(default_factory=list)


class PriorityRiskZone(BaseModel):
    """A coastal polygon flagged for targeted field verification / cleanup."""

    name: str = Field(..., description="Zone name (e.g. 'Kanyakumari').")
    risk_level: RiskSeverity = Field(..., description="Aggregated risk severity.")
    geometry: dict = Field(..., description="GeoJSON Polygon of the impact boundary.")
    eta_hours: Optional[float] = Field(
        default=None, ge=0.0, description="Estimated time of arrival at the zone (h)."
    )
    recommendation: str = Field(
        ..., description="Recommended field action (e.g. 'Deploy cleanup crew')."
    )


class Hotspot(BaseModel):
    """A geo-referenced concentration zone of detected debris."""

    id: str = Field(..., description="Unique hotspot identifier.")
    centroid: Coordinate2D = Field(..., description="Hotspot centroid (lon, lat).")
    radius_km: float = Field(..., ge=0.0, description="Approximate radius (km).")
    intensity: float = Field(
        ..., ge=0.0, le=100.0, description="Relative concentration intensity (%)."
    )
    detection_count: int = Field(
        ..., ge=0, description="Number of detections aggregated in the hotspot."
    )


class SimulationPayload(BaseModel):
    """Complete pre-configured dataset for the MSC ELSA 3 incident case study."""

    case_name: str = Field(default="MSC ELSA 3 Incident", description="Case study name.")
    incident_date: Optional[str] = Field(
        default="2025-05-25", description="Date the vessel sank (ISO date)."
    )
    generated_at: datetime = Field(
        default_factory=datetime.utcnow, description="UTC generation timestamp."
    )
    initial_centroid: Coordinate2D = Field(
        ..., description="Centroid of initial detections (~13 NM off Kerala)."
    )
    detections: List[DebrisDetection] = Field(
        default_factory=list, description="Initial AI detections with confidence scores."
    )
    hotspots: List[Hotspot] = Field(
        default_factory=list, description="Aggregated debris concentration zones."
    )
    trajectories: List[Trajectory] = Field(
        default_factory=list, description="Predicted drift trajectories (24/48/72h)."
    )
    priority_risk_zones: List[PriorityRiskZone] = Field(
        default_factory=list, description="Coastal polygons prioritised for cleanup."
    )
    overall_mean_confidence: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Mean detection confidence (%)."
    )


class HealthResponse(BaseModel):
    """Response payload for the health-check endpoint."""

    status: str = Field(..., description="Overall service status ('ok'/'degraded').")
    app_name: str = Field(..., description="Application name.")
    version: str = Field(..., description="Application version.")
    environment: str = Field(..., description="Deployment environment.")
    model_loaded: bool = Field(
        ..., description="Whether the YOLOv8 detector model is loaded."
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="UTC server time."
    )

