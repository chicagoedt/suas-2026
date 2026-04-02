#!/usr/bin/env python3

"""Simple ground geolocation utility from a bounding-box center pixel."""

from __future__ import annotations

import math


class BoundingBoxCenterLocalizer:
    """Estimate ground geolocation from a bounding-box center pixel."""

    EARTH_RADIUS_M = 6_378_137.0
    FT_TO_M = 0.3048

    def __init__(
        self,
        camera_hfov_deg: float = 21.8,
        camera_vfov_deg: float = 16.4,
        image_width_px: int = 1920,
        image_height_px: int = 1080,
        camera_mount_pitch_deg: float = 45.0,
    ) -> None:
        if image_width_px <= 0 or image_height_px <= 0:
            raise ValueError("image resolution must be positive")
        if camera_hfov_deg <= 0 or camera_vfov_deg <= 0:
            raise ValueError("camera field of view must be positive")

        self.camera_hfov_deg = float(camera_hfov_deg)
        self.camera_vfov_deg = float(camera_vfov_deg)
        self.image_width_px = int(image_width_px)
        self.image_height_px = int(image_height_px)
        self.camera_mount_pitch_deg = float(camera_mount_pitch_deg)

        half_width = self.image_width_px / 2.0
        half_height = self.image_height_px / 2.0
        self._focal_x_px = half_width / math.tan(math.radians(self.camera_hfov_deg) / 2.0)
        self._focal_y_px = half_height / math.tan(math.radians(self.camera_vfov_deg) / 2.0)
        self._center_x_px = half_width
        self._center_y_px = half_height

    def localize_bbox_center(
        self,
        center_x_px: float,
        center_y_px: float,
        drone_lat_deg: float,
        drone_lon_deg: float,
        altitude_ft: float,
        pitch_deg: float,
        roll_deg: float,
        yaw_deg: float,
    ) -> dict[str, float | str | bool | None | list[float]]:
        if altitude_ft <= 0:
            return self._invalid_result(center_x_px, center_y_px, "non_positive_altitude")

        altitude_m = float(altitude_ft) * self.FT_TO_M
        ray_ned = self.pixel_to_ned_ray(
            pixel_x_px=center_x_px,
            pixel_y_px=center_y_px,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
            yaw_deg=yaw_deg,
        )

        down_component = ray_ned[2]
        if down_component <= 1e-6:
            return self._invalid_result(center_x_px, center_y_px, "no_ground_intersection")

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
            "anchor_px": [float(center_x_px), float(center_y_px)],
            "north_offset_m": north_offset_m,
            "east_offset_m": east_offset_m,
            "ground_distance_m": ground_distance_m,
            "latitude": latitude_deg,
            "longitude": longitude_deg,
        }

    def estimate_geolocation(
        self,
        center_x_px: float,
        center_y_px: float,
        drone_lat_deg: float,
        drone_lon_deg: float,
        altitude_ft: float,
        pitch_deg: float,
        roll_deg: float,
        yaw_deg: float,
    ) -> dict[str, float | str | bool | None | list[float]]:
        """Simple public alias for localizing one bounding-box center."""

        return self.localize_bbox_center(
            center_x_px=center_x_px,
            center_y_px=center_y_px,
            drone_lat_deg=drone_lat_deg,
            drone_lon_deg=drone_lon_deg,
            altitude_ft=altitude_ft,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
            yaw_deg=yaw_deg,
        )

    def pixel_to_ned_ray(
        self,
        pixel_x_px: float,
        pixel_y_px: float,
        pitch_deg: float,
        roll_deg: float,
        yaw_deg: float,
    ) -> list[float]:
        x_cam = (float(pixel_x_px) - self._center_x_px) / self._focal_x_px
        y_cam = (float(pixel_y_px) - self._center_y_px) / self._focal_y_px
        ray_cam = self._normalize_vector([x_cam, y_cam, 1.0])

        # Camera frame is x-right, y-down, z-forward.
        # Body/NED logic uses forward-right-down (FRD).
        ray_body = [ray_cam[2], ray_cam[0], ray_cam[1]]
        ray_body = self._rotate_body_about_y(ray_body, -math.radians(self.camera_mount_pitch_deg))
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
        return BoundingBoxCenterLocalizer._matmul_vec(rotation, vector_body)

    @staticmethod
    def _rotate_body_about_y(vector_body: list[float], angle_rad: float) -> list[float]:
        ca = math.cos(angle_rad)
        sa = math.sin(angle_rad)
        rotation = [
            [ca, 0.0, sa],
            [0.0, 1.0, 0.0],
            [-sa, 0.0, ca],
        ]
        return BoundingBoxCenterLocalizer._matmul_vec(rotation, vector_body)

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
    def _invalid_result(
        center_x_px: float,
        center_y_px: float,
        status: str,
    ) -> dict[str, float | str | bool | None | list[float]]:
        return {
            "valid": False,
            "status": status,
            "anchor_px": [float(center_x_px), float(center_y_px)],
            "north_offset_m": None,
            "east_offset_m": None,
            "ground_distance_m": None,
            "latitude": None,
            "longitude": None,
        }
