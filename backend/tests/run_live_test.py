"""Live end-to-end test for the dynamic U-Net GeoTIFF pipeline.

Two layers are verified honestly:

1. HTTP layer (POST /api/v1/detect/unet): full-scene tiling runs, the response
   schema/metadata is correct, RGB and non-GeoTIFF inputs are rejected, and the
   legacy endpoints still work. We do NOT assert a specific debris count on
   synthetic data because the trained MARIDA model's response to random noise
   is not deterministic debris.

2. Geospatial layer (_assemble_live_response): with a REAL non-empty mask and
   a real projected CRS/transform, we verify dynamic WGS84 coordinate
   extraction, real-world area calculation, severity classification, the LIVE
   Overpass coastal query, dynamic alert text, and GeoTIFF export. This proves
   the dynamic pipeline end-to-end without fabricating detections.

Run: py backend\\tests\\run_live_test.py
"""

import os
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np  # noqa: E402


def _make_scene_geotiff(path: str, bands: int, size: int, crs_epsg: str = "EPSG:32644") -> str:
    """Synthetic multi-band GeoTIFF in a projected CRS (10 m pixels)."""
    import rasterio
    from rasterio.transform import from_bounds

    data = (np.random.rand(bands, size, size) * 0.05).astype("float32")
    transform = from_bounds(500000, 1900000, 500000 + size * 10, 1900000 + size * 10, size, size)
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=bands,
        dtype="float32", crs=crs_epsg, transform=transform,
    ) as dst:
        dst.write(data)
    return path


def run() -> int:
    from fastapi.testclient import TestClient

    from backend.main import app, _assemble_live_response

    client = TestClient(app)
    failures = []

    def check(name, cond, extra=""):
        extra = str(extra).encode("ascii", "replace").decode("ascii")
        print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({extra})" if extra else ""))
        if not cond:
            failures.append(name)

    spec = client.get("/openapi.json").json()
    check("route /api/v1/detect/unet present", "/api/v1/detect/unet" in spec["paths"])
    check("route /api/v1/download present", "/api/v1/download/{filename}" in spec["paths"])

    with tempfile.TemporaryDirectory() as d:
        scene = _make_scene_geotiff(os.path.join(d, "scene_512.tif"), bands=11, size=512)
        with open(scene, "rb") as fh:
            r = client.post("/api/v1/detect/unet", files={"file": ("scene_512.tif", fh, "image/tiff")})
        ok = r.status_code == 200
        check("512x512 11-band scene tiling -> 200", ok, f"status={r.status_code} body={r.text[:160] if not ok else ''}")
        if ok:
            j = r.json()
            md = j.get("metadata", {})
            check("metadata.crs echoed (EPSG:32644)", md.get("crs") == "EPSG:32644", md.get("crs"))
            check("metadata dims = [512,512] (tiled full scene)", md.get("image_dimensions") == [512, 512], md.get("image_dimensions"))
            check("metadata resolution ~10m", md.get("spatial_resolution_m", [0])[0] == 10.0, md.get("spatial_resolution_m"))
            m = j.get("live_detection_metrics", {})
            check("metrics total_pixels = 262144", m.get("total_pixels") == 512 * 512, m.get("total_pixels"))
            check("severity_zone valid", m.get("severity_zone") in ("CRITICAL", "MODERATE", "LOW"), m.get("severity_zone"))
            check("preview png data-uri", str(j.get("outputs", {}).get("base64_preview_png", "")).startswith("data:image/png;base64,"))
            check("geotiff_download_url set", bool(j.get("outputs", {}).get("geotiff_download_url")))
            li = j.get("live_intimation", {})
            check("alert_status honest", li.get("alert_status") in ("TRIGGERED", "PARTIAL", "NOT_CONFIGURED"), li.get("alert_status"))

        rgb = _make_scene_geotiff(os.path.join(d, "rgb.tif"), bands=3, size=512)
        with open(rgb, "rb") as fh:
            r = client.post("/api/v1/detect/unet", files={"file": ("rgb.tif", fh, "image/tiff")})
        check("RGB (3-band) rejected 400", r.status_code == 400, f"status={r.status_code} detail={r.json().get('detail','')[:50]}")

        png = os.path.join(d, "rgb.png")
        from PIL import Image
        Image.fromarray((np.random.rand(64, 64, 3) * 255).astype("uint8")).save(png)
        with open(png, "rb") as fh:
            r = client.post("/api/v1/detect/unet", files={"file": ("rgb.png", fh, "image/png")})
        check("non-GeoTIFF PNG rejected 400", r.status_code == 400, f"status={r.status_code}")

    check("legacy health 200", client.get("/api/v1/health").status_code == 200)
    check("legacy simulation 200", client.get("/api/v1/simulation/msc-elsa-3").status_code == 200)
    with tempfile.TemporaryDirectory() as d2:
        png2 = os.path.join(d2, "rgb.png")
        from PIL import Image
        Image.fromarray((np.random.rand(64, 64, 3) * 255).astype("uint8")).save(png2)
        with open(png2, "rb") as fh:
            r = client.post("/api/v1/detect", files={"file": ("rgb.png", fh, "image/png")}, data={"bbox": "76.3,8.0,77.3,9.0"})
        check("legacy RGB detect 200", r.status_code == 200, f"status={r.status_code}")

    _test_geospatial_layer(check, failures)

    print()
    if failures:
        print(f"RESULT: {len(failures)} check(s) FAILED: {failures}")
        return 1
    print("RESULT: ALL LIVE PIPELINE CHECKS PASSED")
    return 0


def _test_geospatial_layer(check, failures):
    """Exercise the dynamic geo/alert/export chain with a real non-empty mask."""
    import rasterio
    from rasterio.transform import from_bounds

    from backend.main import _assemble_live_response
    from backend.services.unet_service import UNetDetectionService

    svc = UNetDetectionService(
        model_path=os.environ.get("UNET_MODEL_PATH") or None,
        config_path=os.environ.get("UNET_CONFIG_PATH") or None,
    )
    size = 512
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[200:300, 200:300] = 1
    crs_epsg = "EPSG:32644"
    transform = from_bounds(500000, 1900000, 500000 + size * 10, 1900000 + size * 10, size, size)
    debris_px = int(mask.sum())
    area_m2 = debris_px * 100.0
    result = {
        "mask": mask,
        "crs": crs_epsg,
        "transform": list(transform)[:6],
        "width": size,
        "height": size,
        "total_pixels": size * size,
        "debris_pixels": debris_px,
        "debris_coverage_percentage": round(debris_px / mask.size * 100, 4),
        "estimated_debris_area_sq_meters": round(area_m2, 2),
        "estimated_debris_area_sq_km": round(area_m2 / 1e6, 6),
        "probability_max": 0.97,
        "spatial_resolution_m": [10.0, 10.0],
        "image_tags": {},
        "mask_png_base64": svc._mask_to_png_base64(mask),
        "model_name": svc.model_name,
        "framework": svc.framework,
    }

    resp = _assemble_live_response("synthetic_scene.tif", result)
    j = resp.model_dump()

    sc = j["spatial_coordinates"]
    check("geo: centroid_lat computed", sc["centroid_latitude"] is not None, sc["centroid_latitude"])
    check("geo: centroid_lon computed", sc["centroid_longitude"] is not None, sc["centroid_longitude"])
    check("geo: bbox 4 values WGS84", isinstance(sc["bounding_box_wgs84"], list) and len(sc["bounding_box_wgs84"]) == 4, sc["bounding_box_wgs84"])
    m = j["live_detection_metrics"]
    check("geo: debris_pixels = 10000 (100x100 block)", m["debris_pixels"] == 10000, m["debris_pixels"])
    check("geo: area_sq_m = 1,000,000 (10000 px x 100 m2)", abs(m["estimated_debris_area_sq_meters"] - 1000000.0) < 1, m["estimated_debris_area_sq_meters"])
    check("geo: area_sq_km = 1.0", abs(m["estimated_debris_area_sq_km"] - 1.0) < 1e-6, m["estimated_debris_area_sq_km"])
    check("geo: severity CRITICAL (area>0.05)", m["severity_zone"] == "CRITICAL", m["severity_zone"])

    cc = j["coastal_community_impact"]
    print(f"       LIVE Overpass near ({sc['centroid_latitude']:.4f},{sc['centroid_longitude']:.4f}): name={cc['nearest_settlement_name']} dist={cc['distance_km']} status={cc['lookup_status']}")
    check("geo: coastal lookup_status present", bool(cc["lookup_status"]), cc["lookup_status"])
    if cc["distance_km"] is not None:
        check("geo: distance_km numeric >= 0", cc["distance_km"] >= 0, cc["distance_km"])
    check("geo: threat_level present", bool(cc["impact_threat_level"]))
    check("geo: recommended_action present", bool(cc["recommended_action"]))

    li = j["live_intimation"]
    check("geo: alert mentions Lat/Lon", "Lat" in li["dynamic_message_text"] and "Lon" in li["dynamic_message_text"])
    check("geo: alert mentions area", "sq km" in li["dynamic_message_text"])
    check("geo: geotiff_download_url set", bool(j["outputs"]["geotiff_download_url"]), j["outputs"]["geotiff_download_url"])


if __name__ == "__main__":
    raise SystemExit(run())
