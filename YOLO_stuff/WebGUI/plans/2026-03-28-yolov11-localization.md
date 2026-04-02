# Plan: Standalone YOLOv11 Geolocation Utility

## Summary

Add a standalone Python localization class that estimates target geographic
position from YOLOv11 detections and drone pose.

- Load class names from a YOLO dataset metadata path
- Use configurable camera geometry with defaults for the current lens and mount
- Estimate north/east offsets plus latitude/longitude for each detection
- Assume a flat ground plane and altitude in feet relative to home elevation

## Key Changes

- Add `localization_yolov11.py` with a `YOLOv11Localizer` class
- Accept Ultralytics-style `Results` objects at runtime without hard-importing
  the `ultralytics` package
- Parse YOLO dataset `names` metadata at initialization
- Convert box-center pixels into a camera ray, then intersect that ray with the
  ground plane using altitude, pitch, roll, and yaw

## Test Plan

- Confirm the module imports and compiles
- Confirm a synthetic centered detection produces a valid localization result
- Confirm dataset metadata parsing works for common YOLO `names` formats
- Confirm invalid cases such as non-positive altitude or no ground intersection
  return structured invalid results

## Assumptions

- Yaw is available at runtime and is interpreted as heading from north
- Altitude input is relative-home feet, not terrain-aware AGL
- Bounding-box center is the target anchor point for this version
