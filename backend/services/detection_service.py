"""AI/ML marine plastic debris detection using Ultralytics YOLOv8."""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import numpy as np

from backend.config import BoundingBox, settings
from backend.models.schemas import (
    Coordinate2D,
    DebrisDetection,
    GeoJSONFeature,
    GeoJSONFeatureCollection,
)

logger = logging.getLogger("aquaflex.detection")


class DetectionService:
    """Detects floating marine plastic debris in satellite imagery via YOLOv8."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: Optional[float] = None,
    ) -> None:
        self.model_path = model_path or settings.yolo_model
        self.conf_threshold = (
            conf_threshold
            if conf_threshold is not None
            else settings.detection_confidence_threshold
        )
        self._model = None
        self._class_names = {}
        self._load_model()

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
        except Exception as exc:
            logger.warning("Ultralytics not available (%s); using fallback detector.", exc)
            self._model = None
            return
        try:
            self._model = YOLO(self.model_path)
            self._class_names = dict(getattr(self._model, "names", {}))
            logger.info("Loaded YOLOv8 model '%s'.", self.model_path)
        except Exception as exc:
            logger.warning(
                "Could not load '%s' (%s); using fallback detector.", self.model_path, exc
            )
            self._model = None

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    def detect_debris(
        self,
        image_path: str,
        bbox: Optional[BoundingBox] = None,
    ) -> Tuple[List[DebrisDetection], GeoJSONFeatureCollection]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        if self._model is not None:
            raw = self._run_yolo(image_path)
        else:
            raw = self._fallback_detect(image_path)

        detections: List[DebrisDetection] = []
        for item in raw:
            conf = float(item["confidence"])
            if conf < self.conf_threshold:
                continue
            x1, y1, x2, y2 = item["bbox"]
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            centroid = (
                self._pixel_to_geo(cx, cy, image_path, bbox) if bbox is not None else None
            )
            label = item.get("label") or self._infer_label(image_path, cx, cy, area)
            detections.append(
                DebrisDetection(
                    bbox=[round(float(v), 2) for v in (x1, y1, x2, y2)],
                    label=label,
                    confidence=round(conf, 4),
                    confidence_percent=round(conf * 100.0, 2),
                    centroid=centroid,
                    area_px=round(float(area), 2),
                )
            )

        return detections, self._to_geojson(detections)

    def _run_yolo(self, image_path: str) -> List[dict]:
        results = self._model.predict(
            source=image_path, conf=self.conf_threshold, verbose=False
        )
        raw: List[dict] = []
        for result in results:
            names = getattr(result, "names", self._class_names)
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, clss):
                label = names.get(int(cls)) if isinstance(names, dict) else None
                raw.append(
                    {
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": float(conf),
                        "label": label,
                    }
                )
        return raw

    def _fallback_detect(self, image_path: str) -> List[dict]:
        rgb = self._load_image(image_path)
        h, w = rgb.shape[:2]
        gray = rgb.mean(axis=2)
        thr = max(float(np.percentile(gray, 99.0)), 200.0)
        mask = gray >= thr
        raw: List[dict] = []
        for (x1, y1, x2, y2) in self._connected_boxes(mask, w, h):
            area = float((x2 - x1) * (y2 - y1))
            conf = float(np.clip(0.55 + area / float(w * h), 0.3, 0.95))
            raw.append(
                {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": conf,
                }
            )
        return raw

    @staticmethod
    def _load_image(image_path: str) -> np.ndarray:
        try:
            from PIL import Image

            return np.asarray(Image.open(image_path).convert("RGB"))
        except Exception:
            pass
        try:
            import rasterio

            with rasterio.open(image_path) as src:
                arr = src.read()
            return np.transpose(arr[:3], (1, 2, 0))
        except Exception as exc:
            raise RuntimeError(f"Unable to read image '{image_path}': {exc}") from exc

    @staticmethod
    def _connected_boxes(mask: np.ndarray, w: int, h: int) -> List[Tuple[int, int, int, int]]:
        visited = np.zeros_like(mask, dtype=bool)
        boxes: List[Tuple[int, int, int, int]] = []
        ys, xs = np.where(mask)
        for sy, sx in zip(ys, xs):
            sy, sx = int(sy), int(sx)
            if visited[sy, sx]:
                continue
            stack = [(sy, sx)]
            visited[sy, sx] = True
            min_x = max_x = sx
            min_y = max_y = sy
            while stack:
                cy, cx = stack.pop()
                min_x = min(min_x, cx)
                max_x = max(max_x, cx)
                min_y = min(min_y, cy)
                max_y = max(max_y, cy)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = cy + dy, cx + dx
                        if (
                            0 <= ny < h
                            and 0 <= nx < w
                            and mask[ny, nx]
                            and not visited[ny, nx]
                        ):
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            if (max_x - min_x) >= 1 and (max_y - min_y) >= 1:
                boxes.append((min_x, min_y, max_x, max_y))
        return boxes

    def _infer_label(self, image_path: str, cx: float, cy: float, area: float) -> str:
        brightness = 200.0
        try:
            rgb = self._load_image(image_path)
            h, w = rgb.shape[:2]
            ix = int(np.clip(cx, 0, w - 1))
            iy = int(np.clip(cy, 0, h - 1))
            patch = rgb[max(0, iy - 2):iy + 3, max(0, ix - 2):ix + 3]
            if patch.size:
                brightness = float(patch.mean())
        except Exception:
            pass
        if area < 400 and brightness > 210:
            return "Nurdles"
        return "Macro-Plastic"

    @staticmethod
    def _pixel_to_geo(cx: float, cy: float, image_path: str, bbox: BoundingBox) -> Coordinate2D:
        try:
            rgb = DetectionService._load_image(image_path)
            h, w = rgb.shape[:2]
        except Exception:
            h = w = 1
        fx = float(np.clip(cx / max(w, 1), 0.0, 1.0))
        fy = float(np.clip(cy / max(h, 1), 0.0, 1.0))
        lon = bbox.min_lon + fx * (bbox.max_lon - bbox.min_lon)
        lat = bbox.max_lat - fy * (bbox.max_lat - bbox.min_lat)
        return Coordinate2D(lon=round(lon, 6), lat=round(lat, 6))

    @staticmethod
    def _to_geojson(detections: List[DebrisDetection]) -> GeoJSONFeatureCollection:
        features: List[GeoJSONFeature] = []
        for i, det in enumerate(detections):
            centroid = det.centroid or Coordinate2D(lon=0.0, lat=0.0)
            geometry = {"type": "Point", "coordinates": [centroid.lon, centroid.lat]}
            properties = {
                "id": f"DET-{i + 1:03d}",
                "label": det.label,
                "confidence": det.confidence,
                "confidence_percent": det.confidence_percent,
                "bbox": det.bbox,
                "area_px": det.area_px,
            }
            features.append(GeoJSONFeature(geometry=geometry, properties=properties))
        return GeoJSONFeatureCollection(features=features)


def get_detection_service() -> DetectionService:
    return DetectionService()

