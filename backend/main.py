"""AquaFlex AI FastAPI application.

Exposes the REST API that powers the GIS dashboard:
    GET  /api/v1/health
    POST /api/v1/detect
    GET  /api/v1/simulation/msc-elsa-3
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import os
import tempfile

from fastapi import File, HTTPException, UploadFile

from backend.services.unet_service import UNetDetectionService

from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.config import BoundingBox, settings
from backend.models.schemas import (
    Coordinate2D,
    DetectionResponse,
    HealthResponse,
    Hotspot,
    SimulationPayload,
)
from backend.services.detection_service import DetectionService, get_detection_service
from backend.services.drift_engine import DriftEngine, get_drift_engine
from backend.services.ingestion_service import IngestionService, get_ingestion_service

logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
logger = logging.getLogger("aquaflex.main")

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=settings.app_description,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    """Attach no-cache headers so the dashboard always loads fresh assets."""
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.endswith((".html", ".js", ".css")):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


# Serve the dashboard (index.html, css/, js/) directly from the backend so the
# whole app is reachable from a single origin (http://localhost:8000) with no
# CORS or stale-cache issues.
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.is_dir():
    if (FRONTEND_DIR / "css").is_dir():
        app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
    if (FRONTEND_DIR / "js").is_dir():
        app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")
    if (FRONTEND_DIR / "vendor").is_dir():
        app.mount("/vendor", StaticFiles(directory=str(FRONTEND_DIR / "vendor")), name="vendor")
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

_detection_service: Optional[DetectionService] = None
_drift_engine: Optional[DriftEngine] = None
_ingestion_service: Optional[IngestionService] = None


def get_services() -> tuple:
    """Lazily initialise and return the shared service singletons."""
    global _detection_service, _drift_engine, _ingestion_service
    if _detection_service is None:
        _detection_service = get_detection_service()
    if _drift_engine is None:
        _drift_engine = get_drift_engine()
    if _ingestion_service is None:
        _ingestion_service = get_ingestion_service()
    return _detection_service, _drift_engine, _ingestion_service


@app.get("/api/v1/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    detection_service, _, _ = get_services()
    status = "ok" if detection_service is not None else "degraded"
    return HealthResponse(
        status=status,
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        model_loaded=detection_service.model_loaded,
    )


@app.post("/api/v1/detect", response_model=DetectionResponse, tags=["detection"])
async def detect_debris(
    file: UploadFile = File(..., description="Satellite tile image to analyse."),
    bbox: Optional[str] = Form(
        default=None,
        description="Optional 'min_lon,min_lat,max_lon,max_lat' for geo-referencing.",
    ),
) -> DetectionResponse:
    detection_service, _, _ = get_services()

    suffix = os.path.splitext(file.filename or "tile.png")[1] or ".png"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        content = await file.read()
        tmp.write(content)
        tmp.flush()
        tmp.close()

        region = None
        if bbox:
            try:
                parts = [float(x) for x in bbox.split(",")]
                if len(parts) != 4:
                    raise ValueError
                region = BoundingBox(
                    min_lon=parts[0], min_lat=parts[1], max_lon=parts[2], max_lat=parts[3]
                )
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="bbox must be 'min_lon,min_lat,max_lon,max_lat'.",
                )

        detections, geojson = detection_service.detect_debris(tmp.name, bbox=region)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    mean_conf = (
        round(sum(d.confidence_percent for d in detections) / len(detections), 2)
        if detections
        else 0.0
    )

    return DetectionResponse(
        source_image=file.filename or "uploaded_tile",
        image_size=None,
        detection_count=len(detections),
        mean_confidence=mean_conf,
        detections=detections,
        geojson=geojson,
    )


@app.get(
    "/api/v1/simulation/msc-elsa-3",
    response_model=SimulationPayload,
    tags=["simulation"],
)
def msc_elsa_3_simulation() -> SimulationPayload:
    detection_service, drift_engine, ingestion = get_services()

    tile = ingestion.get_sentinel2_tile(
        bbox=settings.bbox_incident_site, width=640, height=640
    )
    tile_path = os.path.join(tempfile.gettempdir(), "aquaflex_elsa3_tile.png")
    tile.save_rgb(tile_path)

    detections, _ = detection_service.detect_debris(
        tile_path, bbox=settings.bbox_incident_site
    )

    if detections:
        lons = [d.centroid.lon for d in detections if d.centroid]
        lats = [d.centroid.lat for d in detections if d.centroid]
        if lons and lats:
            centroid = Coordinate2D(
                lon=round(sum(lons) / len(lons), 5),
                lat=round(sum(lats) / len(lats), 5),
            )
        else:
            sinking_lon, sinking_lat = drift_engine._sinking_site_offset()
            centroid = Coordinate2D(lon=sinking_lon, lat=sinking_lat)
    else:
        sinking_lon, sinking_lat = drift_engine._sinking_site_offset()
        centroid = Coordinate2D(lon=sinking_lon, lat=sinking_lat)

    trajectory, risk_zones = drift_engine.simulate_case_study()
    trajectory.origin_lat = centroid.lat
    trajectory.origin_lon = centroid.lon

    hotspots = _build_hotspots(detections, centroid)
    mean_conf = (
        round(sum(d.confidence_percent for d in detections) / len(detections), 2)
        if detections
        else 0.0
    )

    return SimulationPayload(
        case_name="MSC ELSA 3 Incident",
        incident_date="2025-05-25",
        initial_centroid=centroid,
        detections=detections,
        hotspots=hotspots,
        trajectories=[trajectory],
        priority_risk_zones=risk_zones,
        overall_mean_confidence=mean_conf,
    )


def _build_hotspots(detections, centroid: Coordinate2D) -> List[Hotspot]:
    """Cluster detections by class into geo-referenced concentration hotspots."""
    if not detections:
        return [
            Hotspot(
                id="HS-001",
                centroid=centroid,
                radius_km=15.0,
                intensity=50.0,
                detection_count=0,
            )
        ]

    clusters = {"Macro-Plastic": [], "Nurdles": []}
    for det in detections:
        key = "Nurdles" if det.label == "Nurdles" else "Macro-Plastic"
        clusters[key].append(det)

    hotspots: List[Hotspot] = []
    idx = 1
    for key, items in clusters.items():
        if not items:
            continue
        pts = [d.centroid for d in items if d.centroid]
        if pts:
            clon = round(sum(p.lon for p in pts) / len(pts), 5)
            clat = round(sum(p.lat for p in pts) / len(pts), 5)
        else:
            clon, clat = centroid.lon, centroid.lat
        intensity = round(sum(d.confidence_percent for d in items) / len(items), 2)
        hotspots.append(
            Hotspot(
                id=f"HS-{idx:03d}",
                centroid=Coordinate2D(lon=clon, lat=clat),
                radius_km=round(10.0 + len(items) * 0.5, 2),
                intensity=intensity,
                detection_count=len(items),
            )
        )
        idx += 1
    return hotspots


@app.get("/", tags=["system"])
def root():
    """Serve the dashboard at the root; fall back to JSON info if not found."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.is_file():
        from fastapi.responses import FileResponse

        return FileResponse(
            str(index_path),
            media_type="text/html",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "case_study": "MSC ELSA 3 Incident",
    }

_unet_service = None


@app.post("/api/v1/detect/unet")
async def detect_with_unet(file: UploadFile = File(...)):
    global _unet_service

    if not file.filename or not file.filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(
            status_code=400,
            detail="Upload a Sentinel-2 11-band GeoTIFF file."
        )

    temp_path = None

    try:
        if _unet_service is None:
            _unet_service = UNetDetectionService()

        with tempfile.NamedTemporaryFile(
            suffix=".tif", delete=False
        ) as temp_file:
            temp_file.write(await file.read())
            temp_path = temp_file.name

        result = _unet_service.predict_geotiff(temp_path)

        # The NumPy raster profile is not JSON serializable.
        result.pop("raster_profile", None)

        return {
            "model": "AquaFlex AI BCE-Dice U-Net",
            "status": "success",
            **result
        }

    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"U-Net inference failed: {str(exc)}"
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


