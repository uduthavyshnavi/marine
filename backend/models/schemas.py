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


class UNetDetectResponse(BaseModel):
    """Response payload for the U-Net segmentation endpoint.

    Reports the model identity, the configured inference threshold, and the
    debris pixel statistics derived from the segmentation mask. The mask is
    returned as a base64-encoded grayscale PNG (0 = background, 255 = debris)
    so it can be rendered directly in a browser <img> tag without external
    image libraries on the client.
    """

    model_name: str = Field(..., description="Human-readable model name from config.")
    framework: str = Field(default="PyTorch", description="Model framework.")
    threshold: float = Field(
        ..., ge=0.0, le=1.0, description="Probability threshold used for binarisation."
    )
    input_channels: int = Field(..., description="Number of spectral bands expected.")
    band_order: List[str] = Field(
        default_factory=list, description="Ordered spectral bands the model expects."
    )
    width: int = Field(..., ge=1, description="Mask width in pixels.")
    height: int = Field(..., ge=1, description="Mask height in pixels.")
    total_pixels: int = Field(..., ge=0, description="Total pixels in the mask.")
    debris_pixels: int = Field(..., ge=0, description="Pixels classified as debris.")
    debris_percent: float = Field(
        ..., ge=0.0, le=100.0, description="Debris pixels as a percentage of the scene."
    )
    mask_png_base64: str = Field(
        ..., description="Base64-encoded grayscale PNG of the binary debris mask."
    )


class LiveDetectionMetrics(BaseModel):
    """Dynamically computed debris metrics for a live raster scene."""

    total_pixels: int = Field(..., ge=0)
    debris_pixels: int = Field(..., ge=0)
    debris_coverage_percentage: float = Field(..., ge=0.0, le=100.0)
    estimated_debris_area_sq_meters: float = Field(..., ge=0.0)
    estimated_debris_area_sq_km: float = Field(..., ge=0.0)
    max_probability: float = Field(..., ge=0.0, le=1.0)
    severity_zone: str = Field(..., description="CRITICAL | MODERATE | LOW")
    severity_color_code: str = Field(..., description="Hex colour for the severity zone.")
    cluster_scale: str = Field(..., description="Dynamic cluster scale bucket.")
    cleaned_debris_pixels: int = Field(..., ge=0, description="Morphologically cleaned debris pixels.")


class LiveSpatialCoordinates(BaseModel):
    """WGS84 coordinates derived from the live raster transform."""

    centroid_latitude: Optional[float] = None
    centroid_longitude: Optional[float] = None
    bounding_box_wgs84: Optional[List[float]] = Field(
        default=None,
        min_length=4,
        max_length=4,
        description="[min_lon, min_lat, max_lon, max_lat] in WGS84.",
    )
    detection_resolution: dict = Field(
        default_factory=dict, description="Per-axis pixel resolution in metres."
    )


class CoastalCommunityImpact(BaseModel):
    """Nearest coastal settlement and its impact assessment (live query)."""

    nearest_settlement_name: Optional[str] = None
    distance_km: Optional[float] = Field(default=None, ge=0.0)
    impact_threat_level: str
    recommended_action: str
    lookup_status: str = Field(
        default="ok", description="Status of the live coastal query (never fabricated)."
    )


class LiveIntimation(BaseModel):
    """Live alerting dispatch record."""

    alert_status: str = Field(
        ..., description="TRIGGERED | PARTIAL | NOT_CONFIGURED"
    )
    alert_timestamp: str = Field(..., description="ISO-8601 UTC dispatch timestamp.")
    dispatch_channels: List[str] = Field(default_factory=list)
    dynamic_message_text: str


class LiveRasterMetadata(BaseModel):
    """Metadata read directly from the uploaded live raster."""

    filename: str
    crs: Optional[str] = None
    spatial_resolution_m: List[float] = Field(default_factory=list)
    image_dimensions: List[int] = Field(default_factory=list)
    acquisition_time: Optional[str] = None
    quality_gate: dict = Field(
        default_factory=dict,
        description="Image quality gate metrics (glint fraction, threshold, bands).",
    )
    detection_resolution_info: dict = Field(
        default_factory=dict, description="Per-axis resolution + pixel area in metres."
    )
    source_band_count: Optional[int] = None
    mode: str = Field(default="OPTICAL_DEEP_LEARNING_UNET")
    sar_fallback_active: bool = Field(default=False)


class LiveOutputs(BaseModel):
    """Downloadable artefacts produced by the live inference."""

    base64_preview_png: str = Field(..., description="data:image/png;base64,... preview.")
    geotiff_download_url: str


class UNetLiveResponse(BaseModel):
    """Complete live dashboard payload for POST /api/v1/detect/unet."""

    status: str = Field(default="success")
    mode: str = Field(
        default="OPTICAL_DEEP_LEARNING_UNET",
        description="OPTICAL_DEEP_LEARNING_UNET or ALL_WEATHER_SAR_RADAR_FALLBACK",
    )
    sar_fallback_active: bool = Field(
        default=False,
        description="True if optical quality failed and SAR radar fallback was engaged.",
    )
    metadata: LiveRasterMetadata
    live_detection_metrics: LiveDetectionMetrics
    spatial_coordinates: LiveSpatialCoordinates
    coastal_community_impact: CoastalCommunityImpact
    live_intimation: LiveIntimation
    outputs: LiveOutputs
    quality_gate: dict = Field(
        default_factory=dict,
        description="Pre-inference glint/cloud quality gate report.",
    )
    detection_resolution_info: dict = Field(
        default_factory=dict,
        description="Per-axis pixel resolution (m) and pixel area (m²).",
    )
    spatial_detection_metrics: dict = Field(
        default_factory=dict,
        description="Exact physical debris area (m², km²) and cluster scale.",
    )
    # ── Phase 3-5 extended payload fields ─────────────────────────────────────
    source_attribution: dict = Field(
        default_factory=dict,
        description="Reverse Lagrangian source attribution (URBAN_RIVERINE_RUNOFF | OFFSHORE_VESSEL_SPILL).",
    )
    drift_forecast: dict = Field(
        default_factory=dict,
        description="72-hour Lagrangian drift forecast positions (+6/+12/+24/+48/+72h).",
    )
    triage_score: dict = Field(
        default_factory=dict,
        description="Multiplicative risk triage: priority score, threat level, sensitivity analysis.",
    )
    triage: dict = Field(
        default_factory=dict,
        description="Dynamic triage data (threat_level, priority_score, landfall_probability).",
    )
    geospatial: dict = Field(
        default_factory=dict,
        description="Dynamic geospatial metrics (calculated_radius_km, debris_area_km2).",
    )
    detection: dict = Field(
        default_factory=dict,
        description="Dynamic detection statistics (confidence_intensity, debris_pixels).",
    )
    municipal_report_status: dict = Field(
        default_factory=dict,
        description="PDF incident report status and municipal smart-city webhook dispatch results.",
    )
    material_classification: dict = Field(
        default_factory=dict,
        description="Sub-pixel spectral unmixing and debris material composition (Ghost Nets, Macro-Plastics, Biomass).",
    )
    logistics_plan: dict = Field(
        default_factory=dict,
        description="Autonomous cleanup fleet intercept route, base port, ETA, and skimmer logistics.",
    )
    voice_advisories: dict = Field(
        default_factory=dict,
        description="Synthesized Telugu and English voice advisory static MP3 streaming URLs.",
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
    triage: Optional[dict] = Field(
        default_factory=dict, description="Dynamic triage summary for simulation."
    )
    geospatial: Optional[dict] = Field(
        default_factory=dict, description="Dynamic geospatial summary for simulation."
    )
    detection: Optional[dict] = Field(
        default_factory=dict, description="Dynamic detection metrics for simulation."
    )
    material_classification: Optional[dict] = Field(
        default_factory=dict, description="Dominant material composition and hazard."
    )
    logistics_plan: Optional[dict] = Field(
        default_factory=dict, description="Autonomous cleanup fleet intercept plan."
    )
    voice_advisories: Optional[dict] = Field(
        default_factory=dict, description="Synthesized voice advisory audio URLs."
    )
    source_attribution: Optional[dict] = Field(
        default_factory=dict, description="Reverse source attribution for simulation."
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


class CitizenReportRequest(BaseModel):
    """Payload for submitting a crowdsourced coastal citizen report."""

    latitude: Optional[float] = Field(default=None, ge=-90.0, le=90.0, description="WGS84 latitude")
    longitude: Optional[float] = Field(default=None, ge=-180.0, le=180.0, description="WGS84 longitude")
    location_name: Optional[str] = Field(default=None, description="Beach, harbor, or coastal landmark name")
    debris_type: str = Field(
        default="PLASTIC_NURDLES",
        description="Type of debris observed (PLASTIC_NURDLES | MACRO_PLASTICS | GHOST_NETS | OIL_SHEEN | MIXED_DEBRIS)",
    )
    observation_notes: Optional[str] = Field(default="", description="Observations, estimated density, or field notes")
    reporter_contact: Optional[str] = Field(default=None, description="Phone, email, or community group identifier")
    photo_base64: Optional[str] = Field(default=None, description="Optional base64-encoded image or EXIF GPS source")


class CitizenReportResponse(BaseModel):
    """Response returned upon submission of a citizen report."""

    report_id: str = Field(..., description="Unique UUID assigned to this incident report.")
    status: str = Field(default="received", description="Status (received | verified | matched)")
    timestamp: str = Field(..., description="ISO UTC timestamp of report registration.")
    latitude: Optional[float] = Field(default=None, description="Recorded or EXIF-extracted latitude")
    longitude: Optional[float] = Field(default=None, description="Recorded or EXIF-extracted longitude")
    location_name: Optional[str] = Field(default=None)
    debris_type: str
    cluster_matched: bool = Field(
        default=False,
        description="True if within spatial proximity buffer of a known satellite debris cluster.",
    )
    verified_proximity_km: Optional[float] = Field(
        default=None,
        description="Geodesic distance to nearest satellite cluster in km.",
    )
    linked_cluster_id: Optional[str] = Field(
        default=None,
        description="Correlated satellite detection hotspot ID if within spatial proximity buffer.",
    )
    verification_status: str = Field(
        default="PENDING_COMMUNITY_TRIAGE",
        description="Status: PENDING_COMMUNITY_TRIAGE | AUTO_VERIFIED_BY_SATELLITE | UNVERIFIED",
    )
    distance_to_nearest_satellite_hotspot_km: Optional[float] = None


