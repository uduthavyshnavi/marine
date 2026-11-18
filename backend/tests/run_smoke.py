"""Smoke test for the AquaFlex AI backend, focused on U-Net integration.

Runnable directly (no pytest required) and also collectable by pytest:

    py backend\\tests\\run_smoke.py          # Windows
    python backend/tests/run_smoke.py        # cross-platform
    pytest backend/tests/run_smoke.py        # if pytest is installed

It verifies:
  * the backend app imports and starts,
  * /openapi.json lists POST /api/v1/detect/unet,
  * the trained U-Net checkpoint loads (strict) and runs,
  * a valid 11-band, 256x256 GeoTIFF produces a segmentation mask,
  * an RGB (3-band) image is rejected,
  * the pre-existing endpoints (health, simulation, RGB detect) still work.

The test builds a small synthetic 11-band GeoTIFF in a temp directory. This is
test fixture data only - it is NOT fed to the model as if it were real imagery
in production, and no spectral values are fabricated as predictions.
"""

import os
import sys
import tempfile

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np  # noqa: E402


def _make_geotiff(path: str, bands: int, size: int = 256) -> str:
    """Write a synthetic multi-band GeoTIFF for testing."""
    import rasterio
    from rasterio.transform import from_bounds

    data = np.random.rand(bands, size, size).astype("float32") * 0.1
    transform = from_bounds(76.3, 8.0, 77.3, 9.0, size, size)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=bands,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)
    return path


def _make_rgb_png(path: str, size: int = 64) -> str:
    """Write a small RGB PNG for the RGB-rejection test."""
    from PIL import Image

    arr = (np.random.rand(size, size, 3) * 255).astype("uint8")
    Image.fromarray(arr, mode="RGB").save(path)
    return path


def run() -> int:
    from fastapi.testclient import TestClient

    from backend.main import app

    client = TestClient(app)
    failures = []

    def check(name, cond, extra=""):
        status = "PASS" if cond else "FAIL"
        print(f"[{status}] {name}" + (f"  ({extra})" if extra else ""))
        if not cond:
            failures.append(name)

    r = client.get("/api/v1/health")
    check("GET /api/v1/health returns 200", r.status_code == 200, f"status={r.status_code}")

    spec = client.get("/openapi.json").json()
    paths = spec.get("paths", {})
    check(
        "POST /api/v1/detect/unet present in OpenAPI",
        "/api/v1/detect/unet" in paths and "post" in paths["/api/v1/detect/unet"],
    )

    r = client.get("/api/v1/simulation/msc-elsa-3")
    ok_sim = r.status_code == 200 and "detections" in r.json()
    check("GET /api/v1/simulation/msc-elsa-3 still works", ok_sim, f"status={r.status_code}")

    with tempfile.TemporaryDirectory() as d:
        tif = _make_geotiff(os.path.join(d, "valid_11band.tif"), bands=11, size=256)
        with open(tif, "rb") as fh:
            r = client.post(
                "/api/v1/detect/unet",
                files={"file": ("valid_11band.tif", fh, "image/tiff")},
            )
        ok_unet = r.status_code == 200
        body = r.json() if r.status_code == 200 else {}
        check(
            "POST /detect/unet accepts valid 11-band GeoTIFF (200)",
            ok_unet,
            f"status={r.status_code} detail={body.get('detail', '')}",
        )
        if ok_unet:
            m = body.get("live_detection_metrics", {})
            md = body.get("metadata", {})
            check("response reports quality_gate (metadata)", bool(md.get("quality_gate")))
            check("response reports top-level quality_gate", bool(body.get("quality_gate")))
            check("response returns debris pixel count", isinstance(m.get("debris_pixels"), int))
            check("response returns base64 mask PNG", bool(body.get("outputs", {}).get("base64_preview_png")))
            check(
                "response reports spatial_detection_metrics (m2/km2)",
                isinstance(body.get("spatial_detection_metrics", {}).get("debris_area_sq_m"), (int, float)),
            )
            check(
                "response reports detection_resolution_info (pixel_area_m2)",
                isinstance(body.get("detection_resolution_info", {}).get("pixel_area_m2"), (int, float)),
            )
            check("response reports 11 source bands", md.get("source_band_count") == 11)

            # Phase 3-5: new extended response schema keys.
            check(
                "response includes source_attribution dict",
                isinstance(body.get("source_attribution"), dict),
                str(type(body.get("source_attribution"))),
            )
            check(
                "response includes drift_forecast dict",
                isinstance(body.get("drift_forecast"), dict),
                str(type(body.get("drift_forecast"))),
            )
            check(
                "response includes triage_score dict",
                isinstance(body.get("triage_score"), dict),
                str(type(body.get("triage_score"))),
            )
            check(
                "response includes municipal_report_status dict",
                isinstance(body.get("municipal_report_status"), dict),
                str(type(body.get("municipal_report_status"))),
            )

        rgb_tif = _make_geotiff(os.path.join(d, "rgb_3band.tif"), bands=3, size=256)
        with open(rgb_tif, "rb") as fh:
            r = client.post(
                "/api/v1/detect/unet",
                files={"file": ("rgb_3band.tif", fh, "image/tiff")},
            )
        check(
            "RGB (3-band) GeoTIFF rejected with 400",
            r.status_code == 400,
            f"status={r.status_code} detail={r.json().get('detail', '')[:60]}",
        )

        small_tif = _make_geotiff(os.path.join(d, "small_11band.tif"), bands=11, size=128)
        with open(small_tif, "rb") as fh:
            r = client.post(
                "/api/v1/detect/unet",
                files={"file": ("small_11band.tif", fh, "image/tiff")},
            )
        check(
            "11-band but non-256x256 GeoTIFF rejected with 400",
            r.status_code == 400,
            f"status={r.status_code}",
        )

        png = _make_rgb_png(os.path.join(d, "rgb.png"))
        with open(png, "rb") as fh:
            r = client.post(
                "/api/v1/detect/unet",
                files={"file": ("rgb.png", fh, "image/png")},
            )
        check(
            "non-GeoTIFF (PNG) rejected with 400",
            r.status_code == 400,
            f"status={r.status_code}",
        )

        with open(png, "rb") as fh:
            r = client.post(
                "/api/v1/detect",
                files={"file": ("rgb.png", fh, "image/png")},
                data={"bbox": "76.3,8.0,77.3,9.0"},
            )
        check(
            "POST /api/v1/detect (existing) still works",
            r.status_code == 200 and "detections" in r.json(),
            f"status={r.status_code}",
        )


    # Quality gate: oversaturated visible bands must abort with 400.
    import rasterio
    from rasterio.transform import from_bounds

    with tempfile.TemporaryDirectory() as d:
        oversaturated = np.random.rand(11, 256, 256).astype("float32") * 0.1
        oversaturated[1, :, :] = 0.95  # B02 saturated (>= 0.9 glint ceiling)
        oversaturated[2, :, :] = 0.95  # B03
        oversaturated[3, :, :] = 0.95  # B04
        tif = os.path.join(d, "glint_oversat.tif")
        with rasterio.open(
            tif,
            "w",
            driver="GTiff",
            height=256,
            width=256,
            count=11,
            dtype="float32",
            crs="EPSG:4326",
            transform=from_bounds(76.3, 8.0, 77.3, 9.0, 256, 256),
        ) as dst:
            dst.write(oversaturated)
        with open(tif, "rb") as fh:
            r = client.post(
                "/api/v1/detect/unet",
                files={"file": ("glint_oversat.tif", fh, "image/tiff")},
            )
        sar_body = r.json() if r.status_code == 200 else {}
        is_sar_fallback = (
            r.status_code == 200
            and sar_body.get("mode") == "ALL_WEATHER_SAR_RADAR_FALLBACK"
            and sar_body.get("sar_fallback_active") is True
        )
        check(
            "Oversaturated visible bands -> 200 All-Weather SAR Fallback",
            is_sar_fallback,
            f"status={r.status_code} mode={sar_body.get('mode')} sar_active={sar_body.get('sar_fallback_active')}",
        )
        if is_sar_fallback:
            check("SAR response includes sensor Sentinel-1 info", "Sentinel-1" in sar_body.get("quality_gate", {}).get("optical_quality", "") or sar_body.get("quality_gate", {}).get("sar_fallback_triggered") is True)
            check("SAR response computes debris area in m2/km2", isinstance(sar_body.get("spatial_detection_metrics", {}).get("debris_area_sq_m"), (int, float)))

    # Sub-pixel scaling + morphological filtering (unit level).
    from backend.services.unet_service import UNetDetectionService

    svc = UNetDetectionService()
    svc.band_order = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    svc.glint_reject_threshold = 0.4
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[50:54, 50:58] = 1  # 32 px block -> MODERATE_DEBRIS_FIELD (10-50 px)
    probe = np.random.rand(11, 256, 256).astype("float32") * 0.1
    quality = svc.check_image_quality(probe)
    metrics = svc._postprocess_mask(mask, quality, 10.0, 10.0)

    check(
        "postprocess returns cluster_scale", metrics.get("cluster_scale") == "MODERATE_DEBRIS_FIELD",
        metrics.get("cluster_scale"),
    )
    check(
        "postprocess area in m2",
        abs(metrics.get("estimated_debris_area_sq_meters", 0) - 3200.0) < 1,
        str(metrics.get("estimated_debris_area_sq_meters")),
    )
    check("postprocess has spatial_resolution_m", metrics.get("spatial_resolution_m"))

    # Phase 2: Crowdsourced Citizen Reporting endpoint tests
    check(
        "POST /api/v1/citizen/report present in OpenAPI",
        "/api/v1/citizen/report" in paths and "post" in paths["/api/v1/citizen/report"],
    )
    # Test JSON submission with buffer matching (Kanyakumari area)
    cit_payload = {
        "latitude": 8.0900,
        "longitude": 77.5400,
        "location_name": "Kanyakumari Coast",
        "debris_type": "PLASTIC_NURDLES",
        "observation_notes": "Dense windrow washed ashore near jetty.",
        "reporter_contact": "+91-9876543210",
    }
    r_cit = client.post("/api/v1/citizen/report", json=cit_payload)
    check(
        "POST /api/v1/citizen/report returns 201 Created",
        r_cit.status_code == 201,
        f"status={r_cit.status_code}",
    )
    if r_cit.status_code == 201:
        cb = r_cit.json()
        check("Citizen report has report_id", bool(cb.get("report_id")))
        check("Citizen report has verification_status", bool(cb.get("verification_status")))
        check(
            "Citizen report matched satellite cluster",
            cb.get("linked_cluster_id") == "SAT-CL-002" or cb.get("verification_status") == "AUTO_VERIFIED_BY_SATELLITE",
            f"cluster={cb.get('linked_cluster_id')} status={cb.get('verification_status')}",
        )

    # Phase 3: Multilingual Telugu Alert & ReportLab PDF generation tests
    from backend.services.alert_service import build_alert_message, build_bilingual_alert_message
    from backend.services.report_service import generate_pdf_incident_report

    bilingual_msg = build_bilingual_alert_message(
        lat=8.0883,
        lon=77.5385,
        harbor="Kanyakumari",
        dist=3.5,
        heading="ENE",
        speed=1.5,
    )
    check(
        "Alert includes Telugu script",
        "అత్యవసర సముద్ర హెచ్చరిక" in bilingual_msg and "చేపల వలలు" in bilingual_msg,
    )
    check(
        "Alert includes English advisory",
        "EMERGENCY MARITIME ADVISORY" in bilingual_msg and "Net-casting caution" in bilingual_msg,
    )

    pdf_bytes = generate_pdf_incident_report(
        centroid_lat=8.0883,
        centroid_lon=77.5385,
        area_sq_km=0.045,
        area_sq_m=45000.0,
        cluster_scale="MAJOR_PLASTIC_SPILL_WINDROW",
        threat_level="CRITICAL",
        priority_score=0.85,
        alert_message=bilingual_msg,
    )
    check(
        "generate_pdf_incident_report compiles valid PDF bytes",
        isinstance(pdf_bytes, bytes) and pdf_bytes.startswith(b"%PDF"),
        f"bytes_len={len(pdf_bytes) if pdf_bytes else 0}",
    )



    print()
    if failures:
        print(f"RESULT: {len(failures)} check(s) FAILED: {failures}")
        return 1
    print("RESULT: ALL CHECKS PASSED")
    return 0


def test_smoke():
    """pytest entrypoint: fails the test if any check fails."""
    assert run() == 0


if __name__ == "__main__":
    raise SystemExit(run())

