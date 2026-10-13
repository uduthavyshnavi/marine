"""Physics-based drift & dispersion engine (INCOIS SARAT-style trajectory model).

Integrates surface ocean currents and a wind-leeway term using a kinematic
particle advection scheme to forecast where floating debris released at the
MSC ELSA 3 sinking site (~13 NM off Kerala) will travel over a 72-hour horizon,
and which coastal zones it will threaten.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

from backend.config import settings
from backend.models.schemas import RiskSeverity, Trajectory, TrajectoryPoint
from backend.services.ingestion_service import IngestionService, OceanVectorGrid

logger = logging.getLogger("aquaflex.drift")

METERS_PER_DEG_LAT = 111_320.0
NAUTICAL_MILE_M = 1852.0
KANYAKUMARI_CENTER = (77.5384, 8.0883)
RAMESWARAM_CENTER = (79.3129, 9.2876)
SINKING_SITE = (76.7520, 8.1980)


class DriftEngine:
    """Kinematic ocean-debris drift and dispersion simulator."""

    def __init__(
        self,
        ingestion: Optional[IngestionService] = None,
        vector_grid: Optional[OceanVectorGrid] = None,
    ) -> None:
        self.ingestion = ingestion or IngestionService()
        self.vector_grid = vector_grid or self.ingestion.get_ocean_vector_grid()
        self.leeway = settings.wind_leeway_factor
        self.timestep_hours = settings.drift_timestep_hours
        self.horizon_hours = settings.forecast_horizon_hours

    def simulate_trajectory(
        self,
        origin_lon: float,
        origin_lat: float,
        particle_id: str = "seep-001",
        horizon_hours: Optional[int] = None,
        record_hours: Optional[List[int]] = None,
    ) -> Trajectory:
        horizon_hours = horizon_hours or self.horizon_hours
        record_hours = sorted(record_hours or [24, 48, 72])

        lon = float(origin_lon)
        lat = float(origin_lat)
        dt_seconds = self.timestep_hours * 3600.0
        total_steps = int(round(horizon_hours / self.timestep_hours))

        points: List[TrajectoryPoint] = []
        for step in range(1, total_steps + 1):
            hour = step * self.timestep_hours
            u_cur, v_cur = self.vector_grid.current_at(lon, lat)
            wu, wv = self.vector_grid.wind_at(lon, lat)

            u_total = u_cur + self.leeway * wu
            v_total = v_cur + self.leeway * wv

            dx_m = u_total * dt_seconds
            dy_m = v_total * dt_seconds

            dlat = dy_m / METERS_PER_DEG_LAT
            dlon = dx_m / (METERS_PER_DEG_LAT * max(math.cos(math.radians(lat)), 1e-6))
            lat += dlat
            lon += dlon

            if any(abs(hour - rh) < 1e-6 for rh in record_hours):
                points.append(
                    TrajectoryPoint(
                        lat=round(lat, 5),
                        lon=round(lon, 5),
                        hour=int(round(hour)),
                        u=round(u_cur, 4),
                        v=round(v_cur, 4),
                        risk_status=self._risk_at(lat, lon),
                    )
                )

        return Trajectory(
            particle_id=particle_id,
            origin_lat=round(float(origin_lat), 5),
            origin_lon=round(float(origin_lon), 5),
            points=points,
        )

    def _risk_at(self, lat: float, lon: float) -> RiskSeverity:
        d_kanya = self._haversine_km(lat, lon, KANYAKUMARI_CENTER[1], KANYAKUMARI_CENTER[0])
        d_rames = self._haversine_km(lat, lon, RAMESWARAM_CENTER[1], RAMESWARAM_CENTER[0])
        nearest = min(d_kanya, d_rames)
        if nearest <= 30:
            return RiskSeverity.CRITICAL
        if nearest <= 75:
            return RiskSeverity.HIGH
        if nearest <= 150:
            return RiskSeverity.MODERATE
        return RiskSeverity.LOW

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        r = 6371.0088
        p1 = math.radians(lat1)
        p2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlmb = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
        return 2 * r * math.asin(math.sqrt(a))

    @staticmethod
    def _polygon_around(center: Tuple[float, float], radius_deg: float = 0.25) -> dict:
        clon, clat = center
        ring = [
            [clon - radius_deg, clat - radius_deg],
            [clon + radius_deg, clat - radius_deg],
            [clon + radius_deg, clat + radius_deg],
            [clon - radius_deg, clat + radius_deg],
            [clon - radius_deg, clat - radius_deg],
        ]
        return {"type": "Polygon", "coordinates": [ring]}

    def estimate_impact_eta_hours(
        self, trajectory: Trajectory, target_center: Tuple[float, float]
    ) -> Optional[float]:
        best_hour: Optional[float] = None
        best_dist = float("inf")
        for pt in trajectory.points:
            d = self._haversine_km(pt.lat, pt.lon, target_center[1], target_center[0])
            if d < best_dist:
                best_dist = d
                best_hour = float(pt.hour)
        return best_hour

    def hours_to_impact_countdown(self, eta_hours: Optional[float]) -> int:
        if eta_hours is None:
            return self.horizon_hours
        return int(max(0, self.horizon_hours - eta_hours))

    def build_priority_risk_zones(self, trajectory: Trajectory) -> list:
        from backend.models.schemas import PriorityRiskZone

        zones: List[PriorityRiskZone] = []

        eta_kanya = self.estimate_impact_eta_hours(trajectory, KANYAKUMARI_CENTER)
        kanya_risk = self._risk_at(KANYAKUMARI_CENTER[1], KANYAKUMARI_CENTER[0])
        zones.append(
            PriorityRiskZone(
                name="Kanyakumari",
                risk_level=kanya_risk,
                geometry=self._polygon_around(KANYAKUMARI_CENTER, 0.18),
                eta_hours=eta_kanya,
                recommendation=(
                    "Deploy shoreline cleanup crew and install debris booms along "
                    "the Kanyakumari coast for field verification."
                ),
            )
        )

        eta_rames = self.estimate_impact_eta_hours(trajectory, RAMESWARAM_CENTER)
        rames_risk = self._risk_at(RAMESWARAM_CENTER[1], RAMESWARAM_CENTER[0])
        zones.append(
            PriorityRiskZone(
                name="Rameswaram",
                risk_level=rames_risk,
                geometry=self._polygon_around(RAMESWARAM_CENTER, 0.20),
                eta_hours=eta_rames,
                recommendation=(
                    "Pre-position removal assets at Dhanushkodi/Rameswaram and "
                    "schedule targeted nurdle recovery."
                ),
            )
        )

        return zones

    def simulate_case_study(self) -> Tuple[Trajectory, list]:
        origin_lon, origin_lat = self._sinking_site_offset()
        trajectory = self.simulate_trajectory(
            origin_lon=origin_lon,
            origin_lat=origin_lat,
            particle_id="msc-elsa-3-primary",
        )
        risk_zones = self.build_priority_risk_zones(trajectory)
        return trajectory, risk_zones

    def _sinking_site_offset(self) -> Tuple[float, float]:
        offset_deg = (13.0 * NAUTICAL_MILE_M) / METERS_PER_DEG_LAT
        return SINKING_SITE[0], round(SINKING_SITE[1] + offset_deg, 5)


def get_drift_engine() -> DriftEngine:
    return DriftEngine()

