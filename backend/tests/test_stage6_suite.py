import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient
from backend.main import app
from backend.services.alert_service import (
    build_bilingual_alert_message,
    dispatch_alert,
    _send_twilio,
    _send_email,
)
from backend.services.report_service import generate_pdf_digest, get_incident_pdf


def test_stage6_emergency_dispatch():
    client = TestClient(app)

    # 1. Bilingual alert generation
    msg = build_bilingual_alert_message(
        lat=17.6868,
        lon=83.3032,
        nearest_harbor="Visakhapatnam",
        distance_km=2.8,
        cardinal_direction="ENE",
        heading_deg=67.5,
        drift_speed_knots=1.8,
    )
    assert "అత్యవసర సముద్ర హెచ్చరిక" in msg, "Telugu text missing"
    assert "EMERGENCY MARITIME ADVISORY" in msg, "English text missing"
    assert "Visakhapatnam" in msg
    assert "2.8" in msg or "2.80" in msg
    print("[PASS] Bilingual alert format verified")

    # 2. Twilio and Email graceful mock fallbacks
    tw_res = _send_twilio("Test advisory message")
    assert isinstance(tw_res, list)
    assert any(c.get("mock") is True or c.get("status") in ("sent", "simulated_success") for c in tw_res)

    em_res = _send_email("Test advisory message")
    assert em_res.get("mock") is True or em_res.get("status") == "sent"

    disp_res = dispatch_alert(msg)
    assert disp_res["alert_status"] == "TRIGGERED"
    print("[PASS] Emergency dispatch with graceful mock fallbacks verified")

    # 3. POST /api/v1/alerts/dispatch endpoint
    r_disp = client.post("/api/v1/alerts/dispatch", json={"nearest_harbor": "Visakhapatnam", "distance_km": 2.8})
    assert r_disp.status_code == 200
    assert r_disp.json().get("alert_status") == "TRIGGERED"
    print("[PASS] POST /api/v1/alerts/dispatch verified")

    # 4. ReportLab PDF Incident Digest generation
    with tempfile.TemporaryDirectory() as td:
        pdf_path = os.path.join(td, "incident_digest_test.pdf")
        pdf_bytes = generate_pdf_digest(
            {
                "incident_id": "AQ-STAGE6TEST",
                "centroid_lat": 17.6868,
                "centroid_lon": 83.3032,
                "area_sq_km": 0.035,
                "area_sq_m": 35000.0,
                "threat_level": "CRITICAL",
                "priority_score": 0.82,
                "quality_gate": {"passed": True, "mean_confidence": 92.5},
            },
            output_path=pdf_path,
        )
        assert len(pdf_bytes) > 1000
        assert pdf_bytes[:4] == b"%PDF"
        assert os.path.exists(pdf_path)
        assert os.path.getsize(pdf_path) == len(pdf_bytes)
        print("[PASS] generate_pdf_digest file output verified")

    # 5. GET /api/v1/reports/pdf/{incident_id} endpoint
    r_pdf = client.get("/api/v1/reports/pdf/AQ-STAGE6TEST")
    assert r_pdf.status_code == 200
    assert r_pdf.headers["content-type"] == "application/pdf"
    assert r_pdf.content[:4] == b"%PDF"
    print("[PASS] GET /api/v1/reports/pdf/{incident_id} verified")

    # 6. Citizen crowdsourcing endpoint with proximity buffer and cluster matching
    cit_payload = {
        "latitude": 8.0900,
        "longitude": 77.5400,
        "location_name": "Kanyakumari Pier",
        "debris_type": "PLASTIC_NURDLES",
        "observation_notes": "Dense windrow observed washing ashore.",
        "reporter_contact": "+91-9123456789",
    }
    r_cit = client.post("/api/v1/citizen/report", json=cit_payload)
    assert r_cit.status_code == 201
    cb = r_cit.json()
    assert cb.get("report_id") is not None
    assert cb.get("status") in ("matched", "verified")
    assert cb.get("cluster_matched") is True
    assert cb.get("verified_proximity_km") is not None
    print("[PASS] POST /api/v1/citizen/report verified with cluster_matched and verified_proximity_km")

    print("\nALL STAGE 6 SUITE TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_stage6_emergency_dispatch()
