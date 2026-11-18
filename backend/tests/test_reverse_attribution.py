import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from datetime import datetime
from backend.services.drift_service import (
    calculate_reverse_attribution,
    RIVER_OUTLETS,
    SHIPPING_CORRIDORS,
)
from backend.services.attribution_service import calculate_source_attribution
from backend.models.schemas import SimulationPayload


def test_reference_dictionaries_populated():
    """Verify reference river outlets and shipping corridors are loaded."""
    assert len(RIVER_OUTLETS) >= 10
    assert len(SHIPPING_CORRIDORS) >= 10
    
    river_names = [r["name"] for r in RIVER_OUTLETS]
    assert "Periyar River Estuary" in river_names
    assert "Godavari Delta Outflow" in river_names
    assert "Krishna Estuary" in river_names

    corr_names = [c["name"] for c in SHIPPING_CORRIDORS]
    assert "Kochi Port Approach Corridor" in corr_names
    assert "Colombo International Shipping Lane" in corr_names


def test_calculate_reverse_attribution_structure():
    """Verify structure and fields returned by calculate_reverse_attribution."""
    res = calculate_reverse_attribution(lat=8.2177, lon=76.7520, hours=72)

    assert res["status"] == "success"
    assert res["origin_type"] in ("RIVER_RUNOFF", "VESSEL_DISCHARGE", "OFFSHORE_SPILL", "COASTAL_OUTFLOW")
    
    # Coordinates check
    assert "latitude" in res["origin_coordinates"]
    assert "longitude" in res["origin_coordinates"]
    assert isinstance(res["origin_coordinates"]["latitude"], (int, float))
    assert isinstance(res["origin_coordinates"]["longitude"], (int, float))

    # Backtrack path check
    assert isinstance(res["backtrack_path"], list)
    assert len(res["backtrack_path"]) >= 4
    for node in res["backtrack_path"]:
        assert "hour" in node
        assert "latitude" in node
        assert "longitude" in node

    # Nearest source check
    ns = res["nearest_source"]
    assert "name" in ns
    assert "type" in ns
    assert "distance_km" in ns
    assert isinstance(ns["distance_km"], (int, float))

    # Confidence score check
    assert isinstance(res["confidence_score"], (int, float))
    assert 0.0 <= res["confidence_score"] <= 1.0

    # Timestamp check
    assert "discharge_timestamp_estimate" in res
    dt = datetime.fromisoformat(res["discharge_timestamp_estimate"])
    assert dt is not None


def test_river_attribution_proximity():
    """Test reverse attribution when initial debris is close to a known river mouth."""
    # Near Periyar River Estuary (9.9617, 76.2699)
    res = calculate_reverse_attribution(lat=9.97, lon=76.28, hours=12)
    assert res["status"] == "success"
    assert res["origin_type"] in ("RIVER_RUNOFF", "COASTAL_OUTFLOW")
    assert "Periyar" in res["nearest_source"]["name"]
    assert res["confidence_score"] >= 0.70


def test_attribution_service_integration():
    """Test attribution_service integrates reverse Lagrangian backtracking."""
    attr = calculate_source_attribution(centroid_lat=8.2177, centroid_lon=76.7520)
    assert attr["status"] == "success"
    assert "origin_type" in attr
    assert "origin_coordinates" in attr
    assert "backtrack_path" in attr
    assert "nearest_source" in attr
    assert "confidence_score" in attr
    assert "discharge_timestamp_estimate" in attr


def run_all_tests():
    print("Testing reference dictionaries...")
    test_reference_dictionaries_populated()
    print("[PASS] Reference dictionaries test passed.")

    print("Testing calculate_reverse_attribution structure...")
    test_calculate_reverse_attribution_structure()
    print("[PASS] calculate_reverse_attribution structure test passed.")

    print("Testing river attribution proximity...")
    test_river_attribution_proximity()
    print("[PASS] river attribution proximity test passed.")

    print("Testing attribution_service integration...")
    test_attribution_service_integration()
    print("[PASS] attribution_service integration test passed.")

    print("\nALL 4 REVERSE ATTRIBUTION TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_all_tests()
