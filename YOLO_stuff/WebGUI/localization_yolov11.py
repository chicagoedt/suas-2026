#!/usr/bin/env python3

"""Standalone YOLOv11 detection geolocalization utility.

Assumptions:
- Drone attitude inputs follow MAVSDK-style Euler degrees
- Yaw is required and is interpreted as heading clockwise from north
- Altitude is provided in feet relative to the home/takeoff elevation
- Ground is approximated as a flat plane near the home elevation
- Detection anchor is the bounding-box center
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any


class YOLOv11Localizer:
    """Estimate target latitude/longitude from YOLOv11 detections."""

    EARTH_RADIUS_M = 6_378_137.0
    FT_TO_M = 0.3048

    def __init__(
        self,
        dataset_path: str | Path,
        camera_hfov_deg: float = 21.8,
        camera_vfov_deg: float = 16.4,
        image_width_px: int = 1920,
        image_height_px: int = 1080,
        camera_mount_pitch_deg: float = 45.0,
        detection_anchor: str = "box_center",
    ) -> None:
        if image_width_px <= 0 or image_height_px <= 0:
            raise ValueError("image resolution must be positive")
        if camera_hfov_deg <= 0 or camera_vfov_deg <= 0:
            raise ValueError("camera field of view must be positive")
        if detection_anchor != "box_center":
            raise ValueError("only `box_center` is supported in this version")

        self.dataset_path = Path(dataset_path)
        self.class_names = self._load_dataset_names(self.dataset_path)
        self.camera_hfov_deg = float(camera_hfov_deg)
        self.camera_vfov_deg = float(camera_vfov_deg)
        self.image_width_px = int(image_width_px)
        self.image_height_px = int(image_height_px)
        self.camera_mount_pitch_deg = float(camera_mount_pitch_deg)
        self.detection_anchor = detection_anchor

        half_width = self.image_width_px / 2.0
        half_height = self.image_height_px / 2.0
        self._focal_x_px = half_width / math.tan(math.radians(self.camera_hfov_deg) / 2.0)
        self._focal_y_px = half_height / math.tan(math.radians(self.camera_vfov_deg) / 2.0)
        self._center_x_px = half_width
        self._center_y_px = half_height

    def localize_results(
        self,
        results: Any,
        drone_lat_deg: float,
        drone_lon_deg: float,
        altitude_ft: float,
        pitch_deg: float,
        roll_deg: float,
        yaw_deg: float,
    ) -> list[dict[str, Any]]:
        """Localize each detection from an Ultralytics-style Results object."""

        localized: list[dict[str, Any]] = []
        for result in self._normalize_results(results):
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            xyxy_values = self._coerce_box_list(getattr(boxes, "xyxy", []))
            class_values = self._coerce_scalar_list(getattr(boxes, "cls", []))
            confidence_values = self._coerce_scalar_list(getattr(boxes, "conf", []))

            for index, bbox_xyxy_px in enumerate(xyxy_values):
                class_id = int(class_values[index]) if index < len(class_values) else None
                confidence = float(confidence_values[index]) if index < len(confidence_values) else None
                localized.append(
                    self.localize_detection(
                        bbox_xyxy_px=bbox_xyxy_px,
                        class_id=class_id,
                        confidence=confidence,
                        drone_lat_deg=drone_lat_deg,
                        drone_lon_deg=drone_lon_deg,
                        altitude_ft=altitude_ft,
                        pitch_deg=pitch_deg,
                        roll_deg=roll_deg,
                        yaw_deg=yaw_deg,
                    )
                )

        return localized

    def localize_detection(
        self,
        bbox_xyxy_px: list[float] | tuple[float, float, float, float],
        class_id: int | None,
        confidence: float | None,
        drone_lat_deg: float,
        drone_lon_deg: float,
        altitude_ft: float,
        pitch_deg: float,
        roll_deg: float,
        yaw_deg: float,
    ) -> dict[str, Any]:
        """Localize one detection and return a structured result dict."""

        bbox = self._validate_bbox(bbox_xyxy_px)
        anchor_x_px, anchor_y_px = self._bbox_center_anchor(bbox)
        class_name = self.class_names.get(class_id)

        if altitude_ft <= 0:
            return self._invalid_result(
                bbox=bbox,
                anchor_x_px=anchor_x_px,
                anchor_y_px=anchor_y_px,
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                status="non_positive_altitude",
            )

        altitude_m = float(altitude_ft) * self.FT_TO_M
        ray_ned = self._pixel_to_ned_ray(
            pixel_x_px=anchor_x_px,
            pixel_y_px=anchor_y_px,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
            yaw_deg=yaw_deg,
        )

        down_component = ray_ned[2]
        if down_component <= 1e-6:
            return self._invalid_result(
                bbox=bbox,
                anchor_x_px=anchor_x_px,
                anchor_y_px=anchor_y_px,
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                status="no_ground_intersection",
            )

        ray_scale = altitude_m / down_component
        north_offset_m = ray_ned[0] * ray_scale
        east_offset_m = ray_ned[1] * ray_scale
        ground_distance_m = math.hypot(north_offset_m, east_offset_m)

        latitude_deg, longitude_deg = self._offsets_to_lat_lon(
            drone_lat_deg=drone_lat_deg,
            drone_lon_deg=drone_lon_deg,
            north_offset_m=north_offset_m,
            east_offset_m=east_offset_m,
        )

        return {
            "valid": True,
            "status": "ok",
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "bbox_xyxy_px": bbox,
            "anchor_px": [anchor_x_px, anchor_y_px],
            "north_offset_m": north_offset_m,
            "east_offset_m": east_offset_m,
            "ground_distance_m": ground_distance_m,
            "latitude": latitude_deg,
            "longitude": longitude_deg,
        }

    def _pixel_to_ned_ray(
        self,
        pixel_x_px: float,
        pixel_y_px: float,
        pitch_deg: float,
        roll_deg: float,
        yaw_deg: float,
    ) -> list[float]:
        x_cam = (pixel_x_px - self._center_x_px) / self._focal_x_px
        y_cam = (pixel_y_px - self._center_y_px) / self._focal_y_px
        ray_cam = self._normalize_vector([x_cam, y_cam, 1.0])

        # Camera frame is x-right, y-down, z-forward.
        # Body/NED logic uses forward-right-down (FRD).
        ray_body = [ray_cam[2], ray_cam[0], ray_cam[1]]
        ray_body = self._rotate_body_about_y(
            ray_body,
            -math.radians(self.camera_mount_pitch_deg),
        )
        ray_ned = self._body_to_ned(ray_body, roll_deg=roll_deg, pitch_deg=pitch_deg, yaw_deg=yaw_deg)
        return self._normalize_vector(ray_ned)

    def _offsets_to_lat_lon(
        self,
        drone_lat_deg: float,
        drone_lon_deg: float,
        north_offset_m: float,
        east_offset_m: float,
    ) -> tuple[float, float]:
        latitude_rad = math.radians(drone_lat_deg)
        latitude_deg = drone_lat_deg + math.degrees(north_offset_m / self.EARTH_RADIUS_M)
        longitude_scale = self.EARTH_RADIUS_M * max(math.cos(latitude_rad), 1e-9)
        longitude_deg = drone_lon_deg + math.degrees(east_offset_m / longitude_scale)
        return latitude_deg, longitude_deg

    @staticmethod
    def _body_to_ned(
        vector_body: list[float],
        roll_deg: float,
        pitch_deg: float,
        yaw_deg: float,
    ) -> list[float]:
        roll_rad = math.radians(roll_deg)
        pitch_rad = -math.radians(pitch_deg)
        yaw_rad = math.radians(yaw_deg)

        sr = math.sin(roll_rad)
        cr = math.cos(roll_rad)
        sp = math.sin(pitch_rad)
        cp = math.cos(pitch_rad)
        sy = math.sin(yaw_rad)
        cy = math.cos(yaw_rad)

        rotation = [
            [cp * cy, cp * sy, -sp],
            [sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, sr * cp],
            [cr * sp * cy + sr * sy, cr * sp * sy - sr * cy, cr * cp],
        ]
        return YOLOv11Localizer._matmul_vec(rotation, vector_body)

    @staticmethod
    def _rotate_body_about_y(vector_body: list[float], angle_rad: float) -> list[float]:
        ca = math.cos(angle_rad)
        sa = math.sin(angle_rad)
        rotation = [
            [ca, 0.0, sa],
            [0.0, 1.0, 0.0],
            [-sa, 0.0, ca],
        ]
        return YOLOv11Localizer._matmul_vec(rotation, vector_body)

    @staticmethod
    def _matmul_vec(matrix: list[list[float]], vector: list[float]) -> list[float]:
        return [
            matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
            matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
            matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
        ]

    @staticmethod
    def _normalize_vector(vector: list[float]) -> list[float]:
        magnitude = math.sqrt(sum(component * component for component in vector))
        if magnitude <= 1e-12:
            raise ValueError("cannot normalize a zero-length vector")
        return [component / magnitude for component in vector]

    @staticmethod
    def _bbox_center_anchor(bbox_xyxy_px: list[float]) -> tuple[float, float]:
        return (
            (bbox_xyxy_px[0] + bbox_xyxy_px[2]) / 2.0,
            (bbox_xyxy_px[1] + bbox_xyxy_px[3]) / 2.0,
        )

    @staticmethod
    def _validate_bbox(bbox_xyxy_px: list[float] | tuple[float, float, float, float]) -> list[float]:
        if len(bbox_xyxy_px) != 4:
            raise ValueError("bbox_xyxy_px must contain exactly 4 values")
        return [float(value) for value in bbox_xyxy_px]

    def _invalid_result(
        self,
        bbox: list[float],
        anchor_x_px: float,
        anchor_y_px: float,
        class_id: int | None,
        class_name: str | None,
        confidence: float | None,
        status: str,
    ) -> dict[str, Any]:
        return {
            "valid": False,
            "status": status,
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "bbox_xyxy_px": bbox,
            "anchor_px": [anchor_x_px, anchor_y_px],
            "north_offset_m": None,
            "east_offset_m": None,
            "ground_distance_m": None,
            "latitude": None,
            "longitude": None,
        }

    @staticmethod
    def _normalize_results(results: Any) -> list[Any]:
        if results is None:
            return []
        if hasattr(results, "boxes"):
            return [results]
        if isinstance(results, (str, bytes, dict)):
            raise TypeError("results must be an Ultralytics-style Results object or iterable of results")

        try:
            normalized = list(results)
        except TypeError as exc:
            raise TypeError(
                "results must be an Ultralytics-style Results object or iterable of results"
            ) from exc

        return [result for result in normalized if hasattr(result, "boxes")]

    @staticmethod
    def _coerce_box_list(value: Any) -> list[list[float]]:
        if value is None:
            return []
        if hasattr(value, "tolist"):
            value = value.tolist()
        if not value:
            return []
        if isinstance(value[0], (int, float)):
            return [[float(component) for component in value]]
        return [[float(component) for component in box] for box in value]

    @staticmethod
    def _coerce_scalar_list(value: Any) -> list[float]:
        if value is None:
            return []
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, (int, float)):
            return [float(value)]
        return [float(item) for item in value]

    @classmethod
    def _load_dataset_names(cls, dataset_path: Path) -> dict[int, str]:
        if not dataset_path.exists():
            raise FileNotFoundError(f"dataset metadata file not found: {dataset_path}")

        dataset_text = dataset_path.read_text(encoding="utf-8")
        names = cls._extract_names_mapping(dataset_text)
        if not names:
            raise ValueError(f"dataset metadata file does not define usable class names: {dataset_path}")
        return names

    @classmethod
    def _extract_names_mapping(cls, dataset_text: str) -> dict[int, str]:
        lines = dataset_text.splitlines()

        for index, raw_line in enumerate(lines):
            line_without_comment = raw_line.split("#", 1)[0].rstrip()
            stripped = line_without_comment.strip()
            if not stripped or not stripped.startswith("names:"):
                continue

            remainder = stripped[len("names:") :].strip()
            if remainder:
                names = cls._parse_inline_names(remainder)
                if names:
                    return names
                raise ValueError("unsupported inline `names` definition in dataset metadata")

            base_indent = len(line_without_comment) - len(line_without_comment.lstrip(" "))
            block_lines: list[str] = []
            for follow_line in lines[index + 1 :]:
                follow_without_comment = follow_line.split("#", 1)[0].rstrip()
                follow_stripped = follow_without_comment.strip()
                if not follow_stripped:
                    continue

                indent = len(follow_without_comment) - len(follow_without_comment.lstrip(" "))
                if indent <= base_indent:
                    break

                block_lines.append(follow_without_comment[indent:])

            names = cls._parse_block_names(block_lines)
            if names:
                return names

        raise ValueError("dataset metadata is missing a parsable `names` section")

    @classmethod
    def _parse_inline_names(cls, names_text: str) -> dict[int, str] | None:
        try:
            parsed = ast.literal_eval(names_text)
        except (SyntaxError, ValueError):
            return None

        return cls._normalize_names(parsed)

    @classmethod
    def _parse_block_names(cls, block_lines: list[str]) -> dict[int, str] | None:
        if not block_lines:
            return None

        stripped_lines = [line.strip() for line in block_lines if line.strip()]
        if not stripped_lines:
            return None

        if all(line.startswith("- ") for line in stripped_lines):
            return {
                index: cls._strip_scalar(line[2:].strip())
                for index, line in enumerate(stripped_lines)
            }

        names: dict[int, str] = {}
        for line in stripped_lines:
            if ":" not in line:
                return None
            key_text, value_text = line.split(":", 1)
            key_text = key_text.strip()
            value_text = value_text.strip()
            if not key_text.isdigit():
                return None
            names[int(key_text)] = cls._strip_scalar(value_text)

        return names or None

    @classmethod
    def _normalize_names(cls, parsed_names: Any) -> dict[int, str] | None:
        if isinstance(parsed_names, list):
            return {index: str(name) for index, name in enumerate(parsed_names)}

        if isinstance(parsed_names, dict):
            normalized: dict[int, str] = {}
            for key, value in parsed_names.items():
                normalized[int(key)] = str(value)
            return normalized

        return None

    @staticmethod
    def _strip_scalar(value: str) -> str:
        value = value.strip()
        if (value.startswith("'") and value.endswith("'")) or (
            value.startswith('"') and value.endswith('"')
        ):
            return value[1:-1]
        return value

