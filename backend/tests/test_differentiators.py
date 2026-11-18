import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from fastapi.testclient import TestClient

from backend.main import app
from backend.services import spectral_service, logistics_service, alert_service

def test_spectral_service():
    print("[TEST] Testing Sub-Pixel Spectral Service...")
    # Synthetic 11-band raster
    dummy_bands = np.zeros((11, 64, 64), dtype=np.float32)
    # Band 4 (Red), Band 8 (NIR), Band 11 (SWIR)
    dummy_bands[3, 10:20, 10:20] = 0.05
    dummy_bands[7, 10:20, 10:20] = 0.25 # High NIR
    dummy_bands[10, 10:20, 10:20] = 0.08
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:20, 10:20] = 255

    res = spectral_service.classify_material_composition(dummy_bands, mask)
    assert "dominant_material" in res
    assert "floating_debris_index_fdi" in res
    assert "plastic_surface_index_psi" in res
    assert "composition_breakdown" in res
    assert "hazard_classification" in res
    cb = res["composition_breakdown"]
    total_pct = sum(cb.values())
    assert 99.0 <= total_pct <= 101.0, f"Percentages should sum to ~100%, got {total_pct}"
    print(f"  -> Dominant Material: {res['dominant_material']}, FDI: {res['floating_debris_index_fdi']:.4f}")
    print(f"  -> Breakdown: {cb}")
    print("[PASS] Spectral Service verified successfully.")

def test_logistics_service():
    print("[TEST] Testing Autonomous Cleanup Fleet Logistics Service...")
    # Centroid off Kerala coast near Kochi
    center_lat, center_lon = 8.520, 76.650
    debris_area_km2 = 1.2
    drift_positions = [
        {"hour": 0, "latitude": 8.520, "longitude": 76.650},
        {"hour": 6, "latitude": 8.480, "longitude": 76.720},
        {"hour": 12, "latitude": 8.420, "longitude": 76.810},
        {"hour": 24, "latitude": 8.350, "longitude": 76.950},
    ]

    res = logistics_service.plan_fleet_intercept(center_lat, center_lon, debris_area_km2, drift_positions)
    assert res.get("status") == "success"
    assert "nearest_base_port" in res
    assert "geodesic_distance" in res
    assert "intercept_solution" in res
    assert "fleet_recommendation" in res
    port = res["nearest_base_port"]
    dist = res["geodesic_distance"]
    sol = res["intercept_solution"]
    fleet = res["fleet_recommendation"]
    print(f"  -> Nearest Port: {port['name']} ({port['state']})")
    print(f"  -> Transit Distance: {dist['nautical_miles']:.1f} NM ({dist['kilometers']:.1f} km)")
    print(f"  -> Intercept Waypoint: T+{sol['target_hour']}h at ({sol['intercept_latitude']:.3f}, {sol['intercept_longitude']:.3f})")
    print(f"  -> Transit Hours: {sol['vessel_transit_hours']:.1f} hrs @ {sol['cruise_speed_knots']} kts")
    print(f"  -> Fleet Sizing: {fleet['skimmer_vessels_count']} ASV Skimmers, Fuel: {fleet['fuel_consumption_liters']:.0f} L")
    assert fleet["skimmer_vessels_count"] >= 1
    assert dist["nautical_miles"] > 0
    print("[PASS] Fleet Logistics Service verified successfully.")

def test_voice_advisory_service():
    print("[TEST] Testing Multilingual Voice Advisory Generator (Unbranded & Operational)...")
    
    # 1. Test build_audio_advisory_texts directly
    en_txt, te_txt = alert_service.build_audio_advisory_texts(
        lat=8.2177,
        lon=76.7520,
        nearest_harbor="Vizhinjam",
        distance_km=14.5,
        threat_level="CRITICAL",
        debris_area_km2=0.52,
        cardinal_direction="ENE",
        heading_deg=67.5,
        drift_speed_knots=1.5,
    )
    for forbidden in ["AquaFlex AI", "AquaFlex", "AquaFlex Advisory"]:
        assert forbidden not in en_txt, f"Found forbidden '{forbidden}' in English text"
        assert forbidden not in te_txt, f"Found forbidden '{forbidden}' in Telugu text"

    assert "Emergency Maritime Advisory: Marine plastic debris detected" in en_txt
    assert "Status: Classified as CRITICAL hazard with an estimated surface area of 0.52 square kilometers" in en_txt
    assert "Cleanup and Safety Instructions: Deploy containment booms" in en_txt

    assert "అత్యవసర సముద్ర హెచ్చరిక: Vizhinjam తీరం నుండి 14.5 కిలోమీటర్ల దూరంలో" in te_txt
    assert "పరిస్థితి: తీవ్రమైన ప్రమాదం" in te_txt
    assert "పారిశుధ్య మరియు రక్షణ చర్యలు: తీరప్రాంత ముఖద్వారాల వద్ద కంటైన్మెంట్ బూమ్లను ఏర్పాటు చేయండి" in te_txt

    # 2. Test generate_voice_advisory file generation
    res = alert_service.generate_voice_advisory(
        incident_id="TEST-INCIDENT-001",
        lat=8.2177,
        lon=76.7520,
        nearest_harbor="Vizhinjam",
        distance_km=14.5,
        threat_level="CRITICAL",
        debris_area_km2=0.52,
        cardinal_direction="ENE",
        heading_deg=67.5,
        drift_speed_knots=1.5,
    )
    assert res.get("status") == "success"
    assert "telugu_audio_url" in res
    assert "english_audio_url" in res

    # Verify no branding in output texts
    for forbidden in ["AquaFlex AI", "AquaFlex", "AquaFlex Advisory"]:
        assert forbidden not in res["english_advisory_text"]
        assert forbidden not in res["telugu_advisory_text"]

    print(f"  -> English Spoken Text: {res['english_advisory_text'][:120]}...")
    print(f"  -> Telugu Spoken Text: {res['telugu_advisory_text'][:120]}...")
    print(f"  -> Telugu Audio URL: {res['telugu_audio_url']}")
    print(f"  -> English Audio URL: {res['english_audio_url']}")
    print("[PASS] Voice Advisory Service verified successfully (100% Unbranded & Operational).")

def test_api_endpoints():
    print("[TEST] Testing API Endpoints via TestClient...")
    client = TestClient(app)

    # 1. Voice advisory endpoint
    r_voice = client.get("/api/v1/alerts/voice/TEST-INCIDENT-001")
    assert r_voice.status_code == 200, f"Expected 200, got {r_voice.status_code}"
    voice_data = r_voice.json()
    assert voice_data.get("status") == "success"
    assert "telugu_audio_url" in voice_data
    print("  -> Voice Advisory endpoint returned 200 OK")

    # 2. Static audio streaming
    r_audio = client.get(voice_data["telugu_audio_url"])
    assert r_audio.status_code == 200, f"Expected 200, got {r_audio.status_code}"
    assert len(r_audio.content) > 0
    print("  -> Static Audio streaming endpoint returned 200 OK and valid bytes")

    # 3. Simulation payload includes all 3 modules
    r_sim = client.get("/api/v1/simulation/msc-elsa-3")
    assert r_sim.status_code == 200
    sim_data = r_sim.json()
    assert "material_classification" in sim_data
    assert "logistics_plan" in sim_data
    assert "voice_advisories" in sim_data
    print("  -> Simulation payload includes material_classification, logistics_plan, and voice_advisories")
    print("[PASS] All API endpoints verified successfully.")

if __name__ == "__main__":
    test_spectral_service()
    test_logistics_service()
    test_voice_advisory_service()
    test_api_endpoints()
    print("\n[ALL TESTS PASSED SUCCESSFULLY!]")
