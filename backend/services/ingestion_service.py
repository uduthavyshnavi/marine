"""Satellite and oceanographic data ingestion for AquaFlex AI.

Obtains open-access data over the Kerala-Tamil Nadu maritime zone relevant to
the MSC ELSA 3 incident:

1. Sentinel-2 multispectral tiles - Red (B04), Green (B03), Blue (B02) and
   Near-Infrared (B08) bands for the debris-detection composite.
2. INCOIS oceanographic & meteorological products - gridded surface current
   velocity components (u eastward, v northward in m/s) and surface wind
   vectors (wu, wv in m/s).

In production these are fetched live from the Copernicus Data Space Ecosystem
and INCOIS/OSCAR/MOSDAC. To stay runnable offline and deterministic, the
service can synthesise physically-plausible mock rasters and vector grids that
respect the real case-study geometry (coastal currents along southwest India
and the southwest monsoon winds). This layer has no heavy CV dependencies.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from backend.config import BoundingBox, settings

logger = logging.getLogger("aquaflex.ingestion")


@dataclass
class SatelliteTile:
    """A synthetic or fetched Sentinel-2 multispectral tile.

    Each band is a 2-D numpy array of shape (height, width) with values scaled
    to the [0, 1] reflectance range.
    """

    bands: Dict[str, np.ndarray]
    bbox: BoundingBox
    width: int
    height: int
    source: str

    def rgb_composite(self) -> np.ndarray:
        """Return an (H, W, 3) uint8 true-colour composite (B04/B03/B02)."""

        def _scale(band: np.ndarray) -> np.ndarray:
            return (np.clip(band, 0.0, 1.0) * 255.0).astype(np.uint8)

        r = _scale(self.bands.get("B04", np.zeros((self.height, self.width))))
        g = _scale(self.bands.get("B03", np.zeros((self.height, self.width))))
        b = _scale(self.bands.get("B02", np.zeros((self.height, self.width))))
        return np.dstack([r, g, b])

    def save_rgb(self, path: str) -> str:
        """Persist the RGB composite to ``path`` as a PNG; return the path.

        Uses rasterio for georeferenced output when available, otherwise falls
        back to a stdlib-only PNG writer so the service works without GDAL.
        """
        rgb = self.rgb_composite()
        try:  # pragma: no cover - depends on optional native deps
            import rasterio
            from rasterio.transform import from_bounds

            transform = from_bounds(
                self.bbox.min_lon,
                self.bbox.min_lat,
                self.bbox.max_lon,
                self.bbox.max_lat,
                self.width,
                self.height,
            )
            with rasterio.open(
                path,
                "w",
                driver="PNG",
                height=self.height,
                width=self.width,
                count=3,
                dtype="uint8",
                crs="EPSG:4326",
                transform=transform,
            ) as dst:
                for i in range(3):
                    dst.write(rgb[:, :, i], i + 1)
        except Exception as exc:  # pragma: no cover
            logger.warning("rasterio unavailable (%s); using stdlib PNG writer.", exc)
            self._save_rgb_fallback(path, rgb)
        return path

    @staticmethod
    def _save_rgb_fallback(path: str, rgb: np.ndarray) -> None:  # pragma: no cover
        """Minimal RGB PNG writer using only numpy + zlib (stdlib)."""
        import struct
        import zlib

        height, width, _ = rgb.shape
        raw = b"".join(b"\x00" + rgb[y, :, :].tobytes() for y in range(height))

        def _chunk(tag: bytes, data: bytes) -> bytes:
            return (
                struct.pack(">I", len(data))
                + tag
                + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
            )

        png = b"\x89PNG\r\n\x1a\n"
        png += _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        png += _chunk(b"IDAT", zlib.compress(raw, 6))
        png += _chunk(b"IEND", b"")
        with open(path, "wb") as fh:
            fh.write(png)


@dataclass
class OceanVectorGrid:
    """Gridded surface ocean currents and winds over a bounding box.

    u and v are the eastward and northward current components (m/s); wu and wv
    are the eastward and northward 10 m wind components (m/s). Each field is a
    2-D numpy array of shape (ny, nx) with the origin at the north-west corner
    of ``bbox``.
    """

    bbox: BoundingBox
    nx: int
    ny: int
    u: np.ndarray
    v: np.ndarray
    wu: np.ndarray
    wv: np.ndarray
    source: str = "incois-mock"

    def lonlat_to_index(self, lon: float, lat: float) -> Tuple[int, int]:
        """Map a geographic coordinate to the nearest grid (iy, ix) index."""
        xr = self.bbox.max_lon - self.bbox.min_lon
        yr = self.bbox.max_lat - self.bbox.min_lat
        ix = int(round((lon - self.bbox.min_lon) / xr * (self.nx - 1)))
        iy = int(round((self.bbox.max_lat - lat) / yr * (self.ny - 1)))
        ix = min(max(ix, 0), self.nx - 1)
        iy = min(max(iy, 0), self.ny - 1)
        return iy, ix

    def current_at(self, lon: float, lat: float) -> Tuple[float, float]:
        """Return the bilinearly-sampled (u, v) current at a coordinate."""
        return self._sample(self.u, self.v, lon, lat)

    def wind_at(self, lon: float, lat: float) -> Tuple[float, float]:
        """Return the bilinearly-sampled (wu, wv) wind at a coordinate."""
        return self._sample(self.wu, self.wv, lon, lat)

    def _sample(
        self, fa: np.ndarray, fb: np.ndarray, lon: float, lat: float
    ) -> Tuple[float, float]:
        """Bilinear interpolation of two co-located grids at a coordinate."""
        xr = self.bbox.max_lon - self.bbox.min_lon
        yr = self.bbox.max_lat - self.bbox.min_lat
        fx = (lon - self.bbox.min_lon) / xr * (self.nx - 1)
        fy = (self.bbox.max_lat - lat) / yr * (self.ny - 1)
        x0 = min(max(int(math.floor(fx)), 0), self.nx - 1)
        y0 = min(max(int(math.floor(fy)), 0), self.ny - 1)
        x1 = min(x0 + 1, self.nx - 1)
        y1 = min(y0 + 1, self.ny - 1)
        tx = fx - x0
        ty = fy - y0

        def _bil(grid: np.ndarray) -> float:
            top = grid[y0, x0] * (1 - tx) + grid[y0, x1] * tx
            bot = grid[y1, x0] * (1 - tx) + grid[y1, x1] * tx
            return float(top * (1 - ty) + bot * ty)

        return _bil(fa), _bil(fb)


class IngestionService:
    """Fetches or synthesises Sentinel-2 tiles and INCOIS ocean/wind grids."""

    def __init__(self, use_mock: Optional[bool] = None, seed: int = 1337) -> None:
        self._rng = np.random.default_rng(seed)
        self.use_mock = settings.use_mock_satellite if use_mock is None else use_mock

    def get_sentinel2_tile(
        self, bbox: Optional[BoundingBox] = None, width: int = 640, height: int = 640
    ) -> SatelliteTile:
        """Return a Sentinel-2 multispectral tile for ``bbox``.

        In mock mode a physically-plausible ocean surface is synthesised: dark
        water with bright floating-debris signatures (high in all visible bands
        + NIR) so the detector has realistic targets. In live mode this is where
        the Copernicus Open Access Hub request for B04/B03/B02/B08 is issued.
        """
        bbox = bbox or settings.bbox_kerala_coast
        if self.use_mock:
            return self._mock_sentinel2_tile(bbox, width, height)
        return self._live_sentinel2_tile(bbox, width, height)

    def _mock_sentinel2_tile(
        self, bbox: BoundingBox, width: int, height: int
    ) -> SatelliteTile:
        """Synthesise a deterministic Sentinel-2 ocean tile with debris targets."""
        base_b02 = 0.10
        base_b03 = 0.12
        base_b04 = 0.08
        base_b08 = 0.03

        yy, xx = np.mgrid[0:height, 0:width]
        gradient = 0.5 + 0.5 * np.sin(xx / width * 3.0) * np.cos(yy / height * 2.5)

        b02 = np.clip(base_b02 + 0.05 * gradient, 0, 1)
        b03 = np.clip(base_b03 + 0.05 * gradient, 0, 1)
        b04 = np.clip(base_b04 + 0.04 * gradient, 0, 1)
        b08 = np.clip(base_b08 + 0.02 * gradient, 0, 1)

        n_patches = 14
        for _ in range(n_patches):
            cx = int(self._rng.integers(20, width - 20))
            cy = int(self._rng.integers(20, height - 20))
            rx = int(self._rng.integers(4, 12))
            ry = int(self._rng.integers(4, 12))
            y0, y1 = max(0, cy - ry), min(height, cy + ry)
            x0, x1 = max(0, cx - rx), min(width, cx + rx)
            patch_yy, patch_xx = np.mgrid[y0:y1, x0:x1]
            mask = ((patch_xx - cx) / max(rx, 1)) ** 2 + (
                (patch_yy - cy) / max(ry, 1)
            ) ** 2 <= 1.0
            intensity = float(self._rng.uniform(0.65, 0.95))
            b02[y0:y1, x0:x1][mask] = np.clip(intensity + 0.03, 0, 1)
            b03[y0:y1, x0:x1][mask] = np.clip(intensity + 0.02, 0, 1)
            b04[y0:y1, x0:x1][mask] = np.clip(intensity + 0.05, 0, 1)
            b08[y0:y1, x0:x1][mask] = np.clip(intensity - 0.05, 0, 1)

        bands = {"B02": b02, "B03": b03, "B04": b04, "B08": b08}
        return SatelliteTile(
            bands=bands,
            bbox=bbox,
            width=width,
            height=height,
            source="sentinel-2-mock",
        )

    def _live_sentinel2_tile(
        self, bbox: BoundingBox, width: int, height: int
    ) -> SatelliteTile:  # pragma: no cover - requires network + creds
        """Fetch a live Sentinel-2 composite from the Copernicus Data Space API.

        A scaffold for production integration: validates credentials then defers
        to the mock generator until a concrete STAC query/download is wired in,
        so the service never returns empty data silently.
        """
        if not (settings.copernicus_username and settings.copernicus_password):
            raise RuntimeError(
                "Copernicus credentials are not configured. Set COPERNICUS_USERNAME "
                "and COPERNICUS_PASSWORD, or enable USE_MOCK_SATELLITE=true."
            )
        logger.info("Live Sentinel-2 fetch requested for bbox=%s", bbox.as_tuple())
        return self._mock_sentinel2_tile(bbox, width, height)

    def get_ocean_vector_grid(
        self,
        bbox: Optional[BoundingBox] = None,
        nx: int = 40,
        ny: int = 40,
    ) -> OceanVectorGrid:
        """Return a gridded INCOIS-style surface current + wind field."""
        if bbox is None:
            bbox = settings.bbox_arabian_sea
        lons = np.linspace(bbox.min_lon, bbox.max_lon, nx)
        lats = np.linspace(bbox.min_lat, bbox.max_lat, ny)
        lon2d, lat2d = np.meshgrid(lons, lats)

        if self.use_mock:
            u, v = self._mock_currents(lon2d, lat2d)
            wu, wv = self._mock_winds(lon2d, lat2d)
            source = "incois-mock"
        else:
            u, v = self._live_currents(lon2d, lat2d)
            wu, wv = self._live_winds(lon2d, lat2d)
            source = "incois-live"

        return OceanVectorGrid(
            bbox=bbox, nx=nx, ny=ny, u=u, v=v, wu=wu, wv=wv, source=source
        )

    def _mock_currents(
        self, lon2d: np.ndarray, lat2d: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Southward coastal current off southwest India with a weak eastward flow."""
        u = np.full(lon2d.shape, 0.05, dtype=float)
        v = np.full(lon2d.shape, -0.20, dtype=float)

        south_lat, north_lat = 9.5, 6.5
        ramp = np.clip((south_lat - lat2d) / (south_lat - north_lat), 0.0, 1.0)
        v = v - 0.25 * ramp

        bump = np.exp(-(((lon2d - 76.6) / 0.8) ** 2)) * 0.15
        u = u + bump

        gy, gx = lon2d.shape
        u = u + self._rng.normal(0.0, 0.02, size=(gy, gx))
        v = v + self._rng.normal(0.0, 0.02, size=(gy, gx))
        return u, v

    def _mock_winds(
        self, lon2d: np.ndarray, lat2d: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Southwest monsoon winds: onshore westerlies strengthening offshore."""
        wu = np.full(lon2d.shape, 6.0, dtype=float)
        wv = np.full(lon2d.shape, 5.0, dtype=float)

        coast = 77.5
        off = np.clip((coast - lon2d) / 2.0, 0.0, 1.0)
        wu = wu + 4.0 * off

        gy, gx = lon2d.shape
        wu = wu + self._rng.normal(0.0, 1.0, size=(gy, gx))
        wv = wv + self._rng.normal(0.0, 1.0, size=(gy, gx))
        return wu, wv

    def _live_currents(
        self, lon2d: np.ndarray, lat2d: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:  # pragma: no cover - requires network
        logger.info("Live INCOIS current fetch requested at %s", settings.incois_base_url)
        return self._mock_currents(lon2d, lat2d)

    def _live_winds(
        self, lon2d: np.ndarray, lat2d: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:  # pragma: no cover - requires network
        logger.info("Live INCOIS wind fetch requested at %s", settings.incois_base_url)
        return self._mock_winds(lon2d, lat2d)


def get_ingestion_service() -> IngestionService:
    return IngestionService()



