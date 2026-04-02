# Plan: Simple Bounding-Box-Center Geolocation Utility

## Summary

Add a second standalone localization helper that does not parse YOLO results.

- Accept the bounding-box center pixel directly
- Reuse the same default camera assumptions as the YOLOv11 helper
- Estimate ground latitude/longitude from drone GPS, altitude, pitch, roll, and yaw
- Keep the API simple and independent from the Flask app

## Key Changes

- Add `localization_bbox_center.py` with a `BoundingBoxCenterLocalizer` class
- Use default camera parameters:
  - horizontal FOV `21.8°`
  - vertical FOV `16.4°`
  - resolution `1920 x 1080`
  - fixed camera mount pitch `45°` downward
- Accept a center pixel instead of a YOLO results object
- Convert the center pixel into a camera ray and intersect it with a flat ground plane

## Test Plan

- Confirm the module imports and compiles
- Confirm a centered pixel with neutral attitude produces a valid forward ground estimate
- Confirm non-positive altitude returns a structured invalid result

## Assumptions

- Altitude input is relative-home feet
- Yaw is provided as heading from north
- Flat-ground approximation is acceptable for this version
