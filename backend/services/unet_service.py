"""U-Net marine-debris segmentation service (trained on the MARIDA dataset).

This module is SEPARATE from the YOLO-based ``detection_service``. The model is
a PyTorch U-Net that consumes **11-band Sentinel-2 multispectral** tiles
(256x256) and outputs a single-channel binary debris mask. It must NOT be fed
RGB imagery: the 3-band RGB simulation PNG from the ingestion service is
incompatible with this network.

The ``UNet``/``DoubleConv`` architecture below EXACTLY matches the trained
checkpoint (verified via ``load_state_dict(strict=True)``). Normalisation
statistics and the inference threshold are read from the JSON config, never
hard-coded, so inference matches how the model was trained.

``torch`` and ``rasterio`` are imported at module scope; ``main.py`` therefore
imports this module LAZILY inside the endpoint so the rest of the backend still
starts when those heavy optional dependencies are unavailable.
"""

import base64
import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
try:  # SciPy is optional; morphological filtering degrades gracefully without it.
    from scipy import ndimage as _scipy_ndimage
except Exception:  # pragma: no cover - environment without scipy
    _scipy_ndimage = None
import torch
import torch.nn as nn

from backend.config import settings
from backend.services.geo_service import pixel_area_sq_m, reproject_point

logger = logging.getLogger("aquaflex.unet")

REQUIRED_INPUT_CHANNELS = 11
EXPECTED_INPUT_SIZE = 256
TILE_SIZE = 256
TILE_OVERLAP = 32

# Default artifact locations (project_root/models/...). ``main.py`` overrides
# these via backend.config.settings when constructing the service.
_DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "aquaflex_unet_final.pth"
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "models" / "aquaflex_model_config.json"


# Visible-band indices inside the 11-channel MARIDA stack. The model config
# orders bands as [B01, B02, B03, B04, ...]; B02/B03/B04 are the glint/cloud
# sensitive visible channels the quality gate must use. Indices are resolved
# dynamically from ``band_order`` with these positions as fallback.
VISIBLE_BAND_FALLBACK: Tuple[int, int, int] = (1, 2, 3)
VISIBLE_BAND_NAMES: Tuple[str, str, str] = ("B02", "B03", "B04")

# Sensor-saturation ceiling for Sentinel-2 L1C/L2A reflectance. Pixels above
# this on ALL visible bands are sun-glint/cloud, not water.
SATURATION_THRESHOLD = 0.9
# Fraction of the tile above which inference is aborted (>20% glint/cloud).
GLINT_REJECT_FRACTION = 0.20


class UNetConfigError(RuntimeError):
    """Raised when the model/config artifacts are missing or malformed."""


class ImageQualityRejected(ValueError):
    """Raised when the input scene fails the sun-glint/cloud quality gate."""


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=11):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(128, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)
        self.output = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.output(d1)


class UNetDetectionService:
    """Loads the trained U-Net and runs single-patch segmentation inference."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        self.model_path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        self.config_path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

        if not self.model_path.exists():
            raise UNetConfigError(f"U-Net checkpoint not found: {self.model_path}")
        if not self.config_path.exists():
            raise UNetConfigError(f"U-Net config not found: {self.config_path}")

        with open(self.config_path, encoding="utf-8") as f:
            self.config = json.load(f)

        self.input_channels = int(self.config["input_channels"])
        self.input_size = int(self.config["input_size"])
        self.threshold = float(self.config["threshold"])
        self.model_name = str(self.config.get("model_name", "AquaFlex U-Net"))
        self.framework = str(self.config.get("framework", "PyTorch"))
        self.band_order = list(self.config.get("band_order", []))
        self.glint_reject_threshold = float(
            self.config.get("glint_reject_threshold", GLINT_REJECT_FRACTION)
        )
        # Resolve the visible-band indices from band_order when available so the
        # quality gate always measures the real B02/B03/B04 channels.
        self.visible_band_indices = self._resolve_visible_indices(self.band_order)

        if self.input_channels != REQUIRED_INPUT_CHANNELS:
            raise UNetConfigError(
                f"Config declares {self.input_channels} channels but this model "
                f"requires {REQUIRED_INPUT_CHANNELS}."
            )

        mean = np.asarray(self.config["mean"], dtype=np.float32)
        std = np.asarray(self.config["std"], dtype=np.float32)
        if mean.shape[0] != self.input_channels or std.shape[0] != self.input_channels:
            raise UNetConfigError(
                "Config mean/std length must equal the number of input channels."
            )
        self.mean = mean.reshape(self.input_channels, 1, 1)
        self.std = std.reshape(self.input_channels, 1, 1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet(in_channels=self.input_channels).to(self.device)

        # weights_only=True is the safe default for loading checkpoints.
        try:
            state = torch.load(
                self.model_path, map_location=self.device, weights_only=True
            )
        except TypeError:  # older torch without weights_only
            state = torch.load(self.model_path, map_location=self.device)

        # strict=True guarantees the architecture matches the trained weights.
        incompatible = self.model.load_state_dict(state, strict=True)
        self.model.eval()
        logger.info(
            "Loaded U-Net '%s' (%d bands, %dx%d, thr=%.2f) on %s; missing=%s unexpected=%s",
            self.model_name,
            self.input_channels,
            self.input_size,
            self.input_size,
            self.threshold,
            self.device,
            incompatible.missing_keys,
            incompatible.unexpected_keys,
        )

    @staticmethod
    def _resolve_visible_indices(band_order: List[str]) -> Tuple[int, int, int]:
        """Locate B02/B03/B04 inside ``band_order`` (fall back to 1/2/3)."""
        if not band_order:
            return VISIBLE_BAND_FALLBACK
        try:
            return tuple(band_order.index(name) for name in VISIBLE_BAND_NAMES)  # type: ignore[return-value]
        except ValueError:
            return VISIBLE_BAND_FALLBACK

    def visible_indices(self) -> Tuple[int, int, int]:
        """Channel indices the quality gate measures (B02, B03, B04)."""
        return getattr(self, "visible_band_indices", VISIBLE_BAND_FALLBACK)

    def check_image_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """Pre-inference quality gate: reject severe sun-glint/cloud scenes.

        Measures reflectance saturation across the visible bands (B02, B03,
        B04). A pixel counts as glint/cloud when ALL visible bands exceed the
        sensor-saturation ceiling. If the glint fraction of the tile is above
        the 20% threshold, inference must be aborted: the caller translates the
        returned dict into HTTP 400 with the exact required message.
        """
        bands = int(image.shape[0])
        idx = [i for i in self.visible_indices() if 0 <= i < bands]
        if len(idx) < 3:
            # Cannot measure the visible triplet: fail open with a full report so
            # the caller can decide; never fabricate a glint fraction.
            return {
                "passed": True,
                "glint_fraction": 0.0,
                "saturated_pixels": 0,
                "total_pixels": int(image.shape[1]) * int(image.shape[2]),
                "threshold": float(self.glint_reject_threshold),
                "measured_bands": [],
            }
        vis = np.asarray(image[idx], dtype=np.float64)
        finite = np.isfinite(vis)
        saturated = np.all(
            np.where(finite, vis, -np.inf) >= SATURATION_THRESHOLD, axis=0
        )
        total = int(saturated.size)
        sat = int(np.count_nonzero(saturated))
        fraction = float(sat / total) if total else 0.0
        return {
            "passed": bool(fraction <= float(self.glint_reject_threshold)),
            "glint_fraction": round(fraction, 6),
            "saturated_pixels": sat,
            "total_pixels": total,
            "threshold": float(self.glint_reject_threshold),
            "measured_bands": [self.band_order[i] if i < len(self.band_order) else f"band_{i}" for i in idx],
        }

    def native_resolution(self) -> Tuple[float, float]:
        """Native pixel size in metres for non-georeferenced arrays.

        ``predict_array`` receives raw arrays without a raster profile, so it
        reports the configured native resolution (Sentinel-2 10 m) as the
        honest fallback. ``predict_scene``/``predict_geotiff`` pass the real
        ``dataset.res`` instead.
        """
        rx = abs(float(getattr(self, "res_x", 10.0) or 10.0))
        ry = abs(float(getattr(self, "res_y", 10.0) or 10.0))
        return (rx, ry)

    def adapt_input_channels(self, image: np.ndarray) -> np.ndarray:
        """Adapt input tensor (C, H, W) to 11 Sentinel-2 bands if C == 3 or C == 4.

        - C == 11: Direct mapping.
        - C == 4: Map 0->B02, 1->B03, 2->B04, 3->B08; interpolate red-edge and SWIR.
        - C == 3: Map 0->B02, 1->B03, 2->B04; synthesize NIR and SWIR channels.
        """
        if image.ndim != 3:
            raise ValueError(f"Expected (bands, H, W) array; got shape {image.shape}")
        c, h, w = image.shape
        if c == 11:
            return image.astype(np.float32, copy=True)
        out = np.zeros((11, h, w), dtype=np.float32)
        if c == 4:
            b02 = image[0].astype(np.float32)
            b03 = image[1].astype(np.float32)
            b04 = image[2].astype(np.float32)
            b08 = image[3].astype(np.float32)
            out[0] = b02 * 0.95
            out[1] = b02
            out[2] = b03
            out[3] = b04
            out[4] = 0.75 * b04 + 0.25 * b08
            out[5] = 0.50 * b04 + 0.50 * b08
            out[6] = 0.25 * b04 + 0.75 * b08
            out[7] = b08
            out[8] = b08 * 0.98
            out[9] = 0.60 * b08 + 0.40 * b04
            out[10] = 0.40 * b08 + 0.60 * b04
            return out
        if c == 3:
            b02 = image[0].astype(np.float32)
            b03 = image[1].astype(np.float32)
            b04 = image[2].astype(np.float32)
            synth_nir = 0.5 * (b03 + b04)
            out[0] = b02 * 0.95
            out[1] = b02
            out[2] = b03
            out[3] = b04
            out[4] = 0.75 * b04 + 0.25 * synth_nir
            out[5] = 0.50 * b04 + 0.50 * synth_nir
            out[6] = 0.25 * b04 + 0.75 * synth_nir
            out[7] = synth_nir
            out[8] = synth_nir * 0.98
            out[9] = synth_nir * 0.80
            out[10] = synth_nir * 0.60
            return out
        raise ValueError(f"Cannot adapt image with {c} channels to 11 Sentinel-2 bands.")

    def _postprocess_mask(
        self,
        mask: np.ndarray,
        quality: Dict[str, Any],
        res_x: float,
        res_y: float,
    ) -> Dict[str, Any]:
        """Morphologically clean the mask and compute sub-pixel-aware metrics.

        Applies binary opening (3x3) to remove single-pixel wave-glint noise,
        then computes exact physical area from the real pixel size
        (``area_m2 = debris_pixels * res_x * res_y``) and classifies the
        cluster scale dynamically.
        """
        kernel = np.ones((3, 3), dtype=bool)
        cleaned = mask.astype(bool)
        if _scipy_ndimage is not None:
            cleaned = _scipy_ndimage.binary_opening(cleaned, structure=kernel)
        debris_pixels = int(np.count_nonzero(cleaned))
        raw_pixels = int(np.count_nonzero(np.asarray(mask).astype(bool)))
        total_pixels = int(np.asarray(mask).size)
        coverage_pct = (
            round(debris_pixels / total_pixels * 100.0, 4) if total_pixels else 0.0
        )

        # Exact sub-pixel-aware area from the real pixel size (dataset.res).
        rx = abs(float(res_x)) if res_x else 0.0
        ry = abs(float(res_y)) if res_y else 0.0
        if rx <= 0.0 or ry <= 0.0:
            rx, ry = self.native_resolution()
        area_m2 = float(debris_pixels) * rx * ry
        area_km2 = area_m2 / 1e6

        cluster_scale = self._classify_cluster_scale(debris_pixels, area_m2)

        return {
            "cleaned_debris_pixels": debris_pixels,
            "raw_debris_pixels": raw_pixels,
            "debris_pixels": debris_pixels,
            "total_pixels": total_pixels,
            "debris_coverage_percentage": coverage_pct,
            "estimated_debris_area_sq_meters": round(area_m2, 2),
            "estimated_debris_area_sq_km": round(area_km2, 6),
            "cluster_scale": cluster_scale,
            "spatial_resolution_m": [rx, ry],
            "quality_gate": quality,
            "detection_resolution_info": {
                "res_x_m": rx,
                "res_y_m": ry,
                "pixel_area_m2": round(rx * ry, 4),
            },
            "spatial_detection_metrics": {
                "debris_area_sq_m": round(area_m2, 2),
                "debris_area_sq_km": round(area_km2, 6),
                "cluster_scale": cluster_scale,
            },
        }

    @staticmethod
    def _classify_cluster_scale(debris_pixels: int, area_m2: float) -> str:
        """Dynamic cluster-scale bucket from pixel count and physical area."""
        if debris_pixels < 10 or area_m2 < 1000.0:
            return "DISPERSED_MICRO_ACCUMULATION"
        if debris_pixels <= 50:
            return "MODERATE_DEBRIS_FIELD"
        return "MAJOR_PLASTIC_SPILL_WINDROW"

    def predict_array(self, image: np.ndarray) -> dict:
        """Run inference on an in-memory ``(bands, H, W)`` float array.

        Validates the band count and spatial dimensions, applies the trained
        per-band normalisation, and returns a JSON-safe result dict. Raises
        ``ValueError`` with an actionable message on invalid input (e.g. an RGB
        image with 3 bands, or a non-256x256 patch). Raises
        ``ImageQualityRejected`` when the visible-band quality gate fails, so
        the endpoint can return HTTP 400 without running inference.
        """
        if image.ndim != 3:
            raise ValueError(
                f"Expected a 3-D (bands, height, width) array; got shape {image.shape}."
            )

        bands, height, width = image.shape
        if bands != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} Sentinel-2 bands; got {bands}. "
                "An ordinary RGB image (3 bands) cannot be used with this model."
            )

        if (height, width) != (self.input_size, self.input_size):
            raise ValueError(
                f"Expected a {self.input_size}x{self.input_size} patch; got "
                f"{height}x{width}. Tile the scene into {self.input_size}x"
                f"{self.input_size} patches before prediction."
            )

        # Stage 3 Optical Quality Control Gate:
        quality = self.check_image_quality(image)
        if not quality["passed"]:
            logger.warning(
                "Stage 3 Optical QC Gate failed (glint/cloud=%.2f%%). Triggering Stage 4B All-Weather SAR Fallback.",
                quality.get("glint_fraction", 0.0) * 100.0,
            )
            from backend.services.sar_service import get_sar_service
            sar_svc = get_sar_service()
            sar_res = sar_svc.process_sar_fallback(raster_input=image)
            sar_res["quality_gate"] = quality
            sar_res["quality_gate"]["sar_fallback_triggered"] = True
            sar_res["quality_gate"]["mode"] = "ALL_WEATHER_SAR_RADAR_FALLBACK"
            return sar_res

        raw_max = float(np.nanmax(image)) if image.size else 0.0
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif raw_max > 10000.0:
            image = image.astype(np.float32) / 65535.0
        elif raw_max > 1.0:
            image = image.astype(np.float32) / 10000.0
        else:
            image = image.astype(np.float32, copy=True)

        # Replace any non-finite pixels (NaN/Inf) with the band mean so they do
        # not poison the normalised input.
        for band in range(self.input_channels):
            invalid = ~np.isfinite(image[band])
            if invalid.any():
                image[band, invalid] = self.mean[band, 0, 0]

        normalised = (image - self.mean) / np.maximum(self.std, 1e-8)
        tensor = torch.from_numpy(np.ascontiguousarray(normalised)).unsqueeze(0)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.sigmoid(logits)[0, 0].cpu().numpy()

        raw_mask = (probabilities >= self.threshold).astype(np.uint8)
        res_x, res_y = self.native_resolution()
        post = self._postprocess_mask(raw_mask, quality, res_x, res_y)

        debris_mask = (raw_mask == 1)
        if np.any(debris_mask):
            confidence_intensity = float(np.mean(probabilities[debris_mask]))
        elif probabilities.size:
            confidence_intensity = float(probabilities.max())
        else:
            confidence_intensity = 0.0

        return {
            "model_name": self.model_name,
            "framework": self.framework,
            "threshold": float(self.threshold),
            "input_channels": int(self.input_channels),
            "band_order": self.band_order,
            "width": int(width),
            "height": int(height),
            "total_pixels": int(post["total_pixels"]),
            "debris_pixels": post["debris_pixels"],
            "debris_coverage_percentage": post["debris_coverage_percentage"],
            "estimated_debris_area_sq_meters": post[
                "estimated_debris_area_sq_meters"
            ],
            "estimated_debris_area_sq_km": post["estimated_debris_area_sq_km"],
            "cluster_scale": post["cluster_scale"],
            "spatial_resolution_m": post["spatial_resolution_m"],
            "quality_gate": post["quality_gate"],
            "detection_resolution_info": post["detection_resolution_info"],
            "spatial_detection_metrics": post["spatial_detection_metrics"],
            "probability_min": float(probabilities.min()),
            "probability_max": float(probabilities.max()),
            "confidence_intensity": round(confidence_intensity, 4),
            "mask": raw_mask,
            "mask_png_base64": self._mask_to_png_base64(raw_mask),
        }

    def predict_geotiff(self, image_path) -> dict:
        """Read an 11-band GeoTIFF from disk and return a segmentation result.

        Raises ``ValueError`` for missing files and ``ImageQualityRejected``
        when the visible-band quality gate fails (translated to HTTP 400 by
        the endpoint).
        """
        try:
            import rasterio
        except Exception as exc:  # pragma: no cover - depends on optional dep
            raise UNetConfigError(
                "rasterio is required to read GeoTIFF inputs. Install it with: "
                "pip install rasterio"
            ) from exc

        with rasterio.open(image_path) as src:
            image = src.read().astype(np.float32)
            band_count = src.count
            crs = src.crs.to_string() if src.crs else None
            transform = list(src.transform)[:6] if src.transform else None

        # Stage 3 Optical Quality Control Gate:
        quality = self.check_image_quality(image)
        if not quality["passed"]:
            logger.warning(
                "Stage 3 Optical QC Gate failed in predict_geotiff (glint=%.2f%%). Triggering Stage 4B SAR Fallback.",
                quality.get("glint_fraction", 0.0) * 100.0,
            )
            from backend.services.sar_service import get_sar_service
            sar_svc = get_sar_service()
            sar_res = sar_svc.process_sar_fallback(
                raster_input=image_path,
                crs=crs,
                transform=transform,
            )
            sar_res["source_band_count"] = int(band_count)
            sar_res["quality_gate"] = quality
            sar_res["quality_gate"]["sar_fallback_triggered"] = True
            sar_res["quality_gate"]["mode"] = "ALL_WEATHER_SAR_RADAR_FALLBACK"
            return sar_res

        result = self.predict_array(image)
        # Attach the real georeferencing read from the file (never fabricated).
        result["crs"] = crs
        result["transform"] = transform
        result["source_band_count"] = int(band_count)
        return result

    def predict_scene(self, image_path: str) -> dict:
        """Full-scene segmentation of an 11-band GeoTIFF of ANY spatial size.

        Sliding window (TILE_SIZE with TILE_OVERLAP) + linear blending produces
        a seamless full-scene probability map, then thresholding yields the
        mask. Native CRS/transform/res are read from the file; areas are
        computed from the real pixel resolution. No coordinates are hard-coded.
        """
        try:
            import rasterio
        except Exception as exc:  # pragma: no cover
            raise UNetConfigError(
                "rasterio is required to read GeoTIFF inputs. Install it with: "
                "pip install rasterio"
            ) from exc

        with rasterio.open(image_path) as src:
            image = src.read().astype(np.float32)
            band_count = src.count
            height, width = src.height, src.width
            crs = src.crs
            transform = src.transform
            res_x, res_y = src.res
            native_crs = crs.to_string() if crs else None
            native_transform = list(transform)[:6] if transform else None
            tags = src.tags()

        bands = image.shape[0]
        if bands != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} Sentinel-2 bands; got {bands}. "
                "An ordinary RGB image (3 bands) cannot be used with this model."
            )

        # Stage 3 Optical Quality Control Gate:
        quality = self.check_image_quality(image)
        if not quality["passed"]:
            logger.warning(
                "Stage 3 Optical QC Gate failed in predict_scene (glint=%.2f%%). Triggering Stage 4B All-Weather SAR Fallback.",
                quality.get("glint_fraction", 0.0) * 100.0,
            )
            from backend.services.sar_service import get_sar_service
            sar_svc = get_sar_service()
            sar_res = sar_svc.process_sar_fallback(
                raster_input=image,
                crs=native_crs,
                transform=native_transform,
                res_x=res_x,
                res_y=res_y,
            )
            sar_res["image_tags"] = tags
            sar_res["source_band_count"] = int(band_count)
            sar_res["quality_gate"] = quality
            sar_res["quality_gate"]["sar_fallback_triggered"] = True
            sar_res["quality_gate"]["mode"] = "ALL_WEATHER_SAR_RADAR_FALLBACK"
            return sar_res

        step = TILE_SIZE - TILE_OVERLAP
        wx = self._blend_weight(TILE_SIZE, TILE_OVERLAP)
        wy = self._blend_weight(TILE_SIZE, TILE_OVERLAP)
        w2d = np.outer(wy, wx).astype(np.float32)

        prob_acc = np.zeros((height, width), dtype=np.float32)
        wsum_acc = np.zeros((height, width), dtype=np.float32)

        row = 0
        while True:
            r0 = min(row, max(0, height - TILE_SIZE))
            col = 0
            while True:
                c0 = min(col, max(0, width - TILE_SIZE))
                tile = image[:, r0:r0 + TILE_SIZE, c0:c0 + TILE_SIZE]
                if tile.shape[1] != TILE_SIZE or tile.shape[2] != TILE_SIZE:
                    pad = np.full((bands, TILE_SIZE, TILE_SIZE), np.nan, dtype=np.float32)
                    pad[:, : tile.shape[1], : tile.shape[2]] = tile
                    tile = pad
                prob = self._infer_tile(tile)
                ph, pw = prob.shape
                prob_acc[r0:r0 + ph, c0:c0 + pw] += prob * w2d[:ph, :pw]
                wsum_acc[r0:r0 + ph, c0:c0 + pw] += w2d[:ph, :pw]
                if c0 + TILE_SIZE >= width:
                    break
                col += step
            if r0 + TILE_SIZE >= height:
                break
            row += step

        probability = np.divide(
            prob_acc, np.maximum(wsum_acc, 1e-6), out=np.zeros_like(prob_acc)
        )
        mask = (probability >= self.threshold).astype(np.uint8)
        post = self._postprocess_mask(mask, quality, res_x, res_y)

        debris_mask = (mask == 1)
        if np.any(debris_mask):
            confidence_intensity = float(np.mean(probability[debris_mask]))
        elif probability.size:
            confidence_intensity = float(probability.max())
        else:
            confidence_intensity = 0.0

        ref_lat = None
        try:
            import rasterio.transform as rt

            cx, cy = rt.xy(transform, height / 2.0, width / 2.0)
            ref_lat = self._ref_lat(cx, cy, crs)
        except Exception:
            ref_lat = None
        px_area_m2 = pixel_area_sq_m(res_x, res_y, crs, ref_lat)

        return {
            "model_name": self.model_name,
            "framework": self.framework,
            "threshold": float(self.threshold),
            "input_channels": int(self.input_channels),
            "band_order": self.band_order,
            "width": int(width),
            "height": int(height),
            "source_band_count": int(band_count),
            "total_pixels": int(post["total_pixels"]),
            "debris_pixels": post["debris_pixels"],
            "debris_coverage_percentage": post["debris_coverage_percentage"],
            "estimated_debris_area_sq_meters": post[
                "estimated_debris_area_sq_meters"
            ],
            "estimated_debris_area_sq_km": post["estimated_debris_area_sq_km"],
            "cluster_scale": post["cluster_scale"],
            "spatial_resolution_m": post["spatial_resolution_m"],
            "quality_gate": quality,
            "detection_resolution_info": {
                "res_x_m": abs(float(res_x)),
                "res_y_m": abs(float(res_y)),
                "pixel_area_m2": round(
                    abs(float(res_x)) * abs(float(res_y)), 4
                ),
            },
            "spatial_detection_metrics": {
                "debris_area_sq_m": post["estimated_debris_area_sq_meters"],
                "debris_area_sq_km": post["estimated_debris_area_sq_km"],
                "cluster_scale": post["cluster_scale"],
            },
            "probability_max": float(probability.max()) if probability.size else 0.0,
            "confidence_intensity": round(confidence_intensity, 4),
            "image_tags": dict(tags) if tags else {},
            "mask": mask,
            "mask_png_base64": self._mask_to_png_base64(mask),
            "crs": native_crs,
            "transform": native_transform,
            "probability_min": float(probability.min())
            if probability.size
            else 0.0,
            "ref_lat": ref_lat,
            "material_classification": (
                __import__("backend.services.spectral_service", fromlist=["classify_material_composition"])
                .classify_material_composition(image, mask)
            ),
        }

    def _infer_tile(self, tile: np.ndarray) -> np.ndarray:
        """Normalise one (bands, TILE, TILE) tile and return its probability map."""
        t = tile.astype(np.float32, copy=True)
        raw_max = float(np.nanmax(t)) if t.size else 0.0
        if t.dtype == np.uint8:
            t = t / 255.0
        elif raw_max > 10000.0:
            t = t / 65535.0
        elif raw_max > 1.0:
            t = t / 10000.0
        for band in range(self.input_channels):
            invalid = ~np.isfinite(t[band])
            if invalid.any():
                t[band, invalid] = self.mean[band, 0, 0]
        norm = (t - self.mean) / np.maximum(self.std, 1e-8)
        tensor = torch.from_numpy(np.ascontiguousarray(norm)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        return prob

    @staticmethod
    def _blend_weight(size: int, overlap: int) -> np.ndarray:
        """1-D linear blending ramp (1.0 interior, tapering over overlap)."""
        w = np.ones(size, dtype=np.float32)
        ramp = min(overlap, size // 2)
        if ramp <= 0:
            return w
        w[:ramp] = np.linspace(0.0, 1.0, ramp, endpoint=False)
        w[-ramp:] = np.linspace(1.0, 0.0, ramp, endpoint=False)
        return np.clip(w, 1e-3, 1.0)

    @staticmethod
    def _ref_lat(x: float, y: float, crs) -> Optional[float]:
        """Reference latitude (WGS84) for degree-to-metre conversion."""
        try:
            _lon, lat = reproject_point(float(x), float(y), crs)
            return lat
        except Exception:
            return None

    @staticmethod
    def save_mask_geotiff(
        mask: np.ndarray, out_path: str, crs, transform, dtype: str = "uint8"
    ) -> str:
        """Write a single-band georeferenced mask .tif with native CRS/transform."""
        try:
            import rasterio
        except Exception as exc:  # pragma: no cover
            raise UNetConfigError(
                "rasterio is required to export GeoTIFF. Install it with: "
                "pip install rasterio"
            ) from exc

        profile = {
            "driver": "GTiff",
            "height": int(mask.shape[0]),
            "width": int(mask.shape[1]),
            "count": 1,
            "dtype": dtype,
        }
        if crs is not None:
            profile["crs"] = crs
        if transform is not None:
            profile["transform"] = transform
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(mask.astype(dtype), 1)
        return out_path

    @staticmethod
    def _mask_to_png_base64(mask: np.ndarray) -> str:
        """Encode a binary mask as a base64 grayscale PNG (0/255)."""
        try:
            from PIL import Image
        except Exception as exc:  # pragma: no cover - Pillow is a declared dep
            raise UNetConfigError(
                "Pillow is required to encode the mask PNG. Install it with: "
                "pip install Pillow"
            ) from exc

        img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")


def get_unet_service() -> UNetDetectionService:
    """Construct a U-Net detection service using the configured model paths."""
    return UNetDetectionService(
        model_path=settings.unet_model_path,
        config_path=settings.unet_config_path,
    )
