# WebGUI PRD

## Working Conventions

- Store implementation plans in `plans/`
- Update `PRD.md` when a new plan is added or a feature is completed

## Current Status

- [x] Create local `venv` and pinned `requirements.txt`
- [x] Add `.gitignore` entries for local Python artifacts and `venv/`
- [x] Add `scripts/run.sh` launcher for `app.py`
- [x] Create `app.py` as the active application entrypoint
- [x] Replace the single telemetry page with a four-panel operator dashboard
- [x] Document `PRD.md` usage and plan tracking in `README.md`
- [x] Add relative altitude to `Flight Data` via MAVSDK position telemetry
- [x] Add cross-platform MAVLink serial auto-detection with 5s retry
- [x] Integrate a live camera feed into `Camera Feed`
- [x] Implement payload operator controls over USB serial
- [x] Add ESP32 test firmware for payload serial integration
- [x] Add Raspberry Pi Pico test firmware for payload serial integration
- [x] Tighten macOS payload serial detection to avoid Bluetooth serial devices
- [x] Set the Pico demo sketch to default open and mirror state on the LED
- [x] Add a standalone YOLOv11 localization utility
- [x] Add a simple bounding-box-center localization utility
- [ ] Populate `Misc` with mission utilities or diagnostics

## Latest Plan Summary

- Added plan file: `plans/2026-03-24-payload-operator-dashboard-layout.md`
- Completed the dashboard layout for `Pay Operator Interface`
- Preserved the existing MAVSDK telemetry backend and `/gyro` endpoint contract
- Simplified dashboard headers by removing quadrant position labels
- Enabled Flask debug mode in `app.py` for development-time reload behavior
- Added plan file: `plans/2026-03-24-flight-data-altitude.md`
- Extended `Flight Data` to show MAVSDK pitch and relative altitude
- Guarded the telemetry thread against Flask debug reloader double-starts
- Added plan file: `plans/2026-03-24-auto-detect-serial-device.md`
- Replaced the hardcoded serial path with cross-platform MAVLink serial auto-detect
- Fixed the MAVSDK attitude rate call to use the installed euler-rate API
- Switched the displayed altitude unit in `Flight Data` from meters to feet
- Added plan file: `plans/2026-03-25-gstreamer-camera-feed.md`
- Replaced the `Camera Feed` placeholder with a GStreamer webcam device `1` 480p stream
- Added plan file: `plans/2026-03-25-payload-serial-control.md`
- Replaced the `Payload Operator` placeholder with serial-backed Open/Close controls
- Added auto-detected USB payload-controller state sync and ESP32 test firmware
- Added plan file: `plans/2026-03-26-pico-payload-test-firmware.md`
- Added a Raspberry Pi Pico Arduino IDE test sketch for the payload serial protocol
- Added plan file: `plans/2026-03-26-macos-payload-port-filter.md`
- Tightened macOS payload-controller probing to prefer `usbmodem`-style USB ports and skip Bluetooth serial endpoints
- Added plan file: `plans/2026-03-26-pico-led-state.md`
- Updated the Pico demo so startup defaults to open and the onboard LED mirrors payload state
- Added plan file: `plans/2026-03-28-yolov11-localization.md`
- Added a standalone YOLOv11-based geolocation helper using drone GPS, altitude, pitch, roll, yaw, and camera parameters
- Added plan file: `plans/2026-03-28-bbox-center-localization.md`
- Added a simpler localization helper that takes a bounding-box center pixel directly without YOLO parsing
