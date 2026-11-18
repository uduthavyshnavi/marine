"""Centralised configuration for AquaFlex AI.

Loads environment variables (via python-dotenv) and exposes a cached Settings
singleton plus the geographic bounding boxes for the Arabian Sea / Kerala /
Kanyakumari / Rameswaram maritime zone relevant to the MSC ELSA 3 incident.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv(*_args, **_kwargs):  # type: ignore
        """No-op fallback when python-dotenv is unavailable."""
        return False

load_dotenv()


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Env var {name!r} must be a float, got {raw!r}") from exc


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_list(name: str, default: List[str]) -> List[str]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class BoundingBox:
    """Geographic bounding box as (min_lon, min_lat, max_lon, max_lat)."""

    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)

    def as_geojson_polygon(self) -> Dict:
        ring = [
            [self.min_lon, self.min_lat],
            [self.max_lon, self.min_lat],
            [self.max_lon, self.max_lat],
            [self.min_lon, self.max_lat],
            [self.min_lon, self.min_lat],
        ]
        return {"type": "Polygon", "coordinates": [ring]}


@dataclass(frozen=True)
class Settings:
    """Immutable application settings resolved from the environment."""

    app_name: str = field(default_factory=lambda: os.getenv("APP_NAME", "AquaFlex AI"))
    app_version: str = field(default_factory=lambda: os.getenv("APP_VERSION", "1.0.0"))
    app_description: str = field(
        default_factory=lambda: os.getenv(
            "APP_DESCRIPTION",
            "Automated Marine Plastic Debris Detection & Predictive Drift "
            "Simulation Platform (MSC ELSA 3 Case Study)",
        )
    )
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))

    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    cors_origins: List[str] = field(
        default_factory=lambda: _get_list(
            "CORS_ORIGINS",
            [
                "http://localhost:8000",
                "http://127.0.0.1:8000",
                "http://localhost:5500",
                "http://127.0.0.1:5500",
                "http://localhost:3000",
                "http://127.0.0.1:3000",
                "null",
            ],
        )
    )

    use_mock_satellite: bool = field(
        default_factory=lambda: _get_bool("USE_MOCK_SATELLITE", True)
    )
    copernicus_username: str = field(default_factory=lambda: os.getenv("COPERNICUS_USERNAME", ""))
    copernicus_password: str = field(default_factory=lambda: os.getenv("COPERNICUS_PASSWORD", ""))
    sentinel2_bands: List[str] = field(
        default_factory=lambda: _get_list("SENTINEL2_BANDS", ["B04", "B03", "B02", "B08"])
    )

    incois_base_url: str = field(
        default_factory=lambda: os.getenv("INCOIS_BASE_URL", "https://incois.gov.in")
    )
    wind_leeway_factor: float = field(
        default_factory=lambda: _get_float("WIND_LEEWAY_FACTOR", 0.03)
    )

    yolo_model: str = field(default_factory=lambda: os.getenv("YOLO_MODEL", "yolov8n.pt"))
    detection_confidence_threshold: float = field(
        default_factory=lambda: _get_float("DETECTION_CONFIDENCE_THRESHOLD", 0.25)
    )

    # --- U-Net segmentation model (trained on MARIDA) ----------------------
    unet_model_path: str = field(
        default_factory=lambda: os.getenv(
            "UNET_MODEL_PATH",
            str(Path(__file__).resolve().parent.parent / "models" / "aquaflex_unet_final.pth"),
        )
    )
    unet_config_path: str = field(
        default_factory=lambda: os.getenv(
            "UNET_CONFIG_PATH",
            str(Path(__file__).resolve().parent.parent / "models" / "aquaflex_model_config.json"),
        )
    )
    # Directory for generated live inference outputs (e.g. mask GeoTIFFs).
    unet_output_dir: str = field(
        default_factory=lambda: os.getenv(
            "UNET_OUTPUT_DIR",
            str(Path(__file__).resolve().parent.parent / "outputs"),
        )
    )
    # Radius (km) for the live coastal-settlement proximity query.
    coastal_query_radius_km: float = field(
        default_factory=lambda: _get_float("COASTAL_QUERY_RADIUS_KM", 30.0)
    )

    forecast_horizon_hours: int = field(
        default_factory=lambda: int(os.getenv("FORECAST_HORIZON_HOURS", "72"))
    )
    drift_timestep_hours: float = field(
        default_factory=lambda: _get_float("DRIFT_TIMESTEP_HOURS", 1.0)
    )

    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    bbox_arabian_sea: BoundingBox = field(
        default_factory=lambda: BoundingBox(72.0, 6.0, 78.5, 13.5)
    )
    bbox_kerala_coast: BoundingBox = field(
        default_factory=lambda: BoundingBox(74.5, 8.0, 77.5, 12.8)
    )
    bbox_incident_site: BoundingBox = field(
        default_factory=lambda: BoundingBox(76.3, 8.0, 77.3, 9.0)
    )
    bbox_kanyakumari: BoundingBox = field(
        default_factory=lambda: BoundingBox(77.3, 7.9, 77.8, 8.3)
    )
    bbox_rameswaram: BoundingBox = field(
        default_factory=lambda: BoundingBox(79.0, 9.0, 79.6, 9.6)
    )

    @property
    def regions(self) -> Dict[str, BoundingBox]:
        return {
            "arabian_sea": self.bbox_arabian_sea,
            "kerala_coast": self.bbox_kerala_coast,
            "incident_site": self.bbox_incident_site,
            "kanyakumari": self.bbox_kanyakumari,
            "rameswaram": self.bbox_rameswaram,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance (process-wide singleton)."""
    return Settings()


settings = get_settings()
