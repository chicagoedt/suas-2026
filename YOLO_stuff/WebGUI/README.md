# MAVSDK Telemetry Web GUI

This project exposes live MAVSDK telemetry through a small Flask web interface.
It connects to a MAVLink system, streams IMU and attitude data, and renders a
simple browser-based view for quick inspection during development.

## What It Does

- Connects to a MAVSDK-compatible vehicle or simulator
- Streams gyroscope data from `telemetry.imu()`
- Streams pitch, roll, and relative altitude data from MAVSDK telemetry
- Auto-detects likely MAVLink serial devices on macOS, Linux, and Windows
- Streams `Camera Feed` through GStreamer using webcam device index `1` at 640x480
- Auto-detects a USB payload controller and sends `OPEN` / `CLOSE` serial commands
- Reads confirmed payload `OPEN` / `CLOSED` state from the payload controller
- Includes a standalone YOLOv11 localization utility for estimating target
  lat/lon from detections and drone pose
- Includes a simpler bounding-box-center localization utility with no YOLO dependency
- Serves a lightweight web UI for viewing current telemetry state

## Project Layout

- `app.py`: main application entrypoint
- `scripts/run.sh`: convenience launcher for the app
- `requirements.txt`: pinned Python dependencies
- `localization_bbox_center.py`: simple center-pixel geolocation helper
- `localization_yolov11.py`: standalone YOLOv11 geolocation helper
- `firmware/`: test microcontroller firmware
- `venv/`: local virtual environment
- `plans/`: project plans and task breakdowns
- `PRD.md`: high-level feature summary and completion tracking

## PRD Tracking

`PRD.md` is the lightweight project tracker for this repository.

- Add a short summary to `PRD.md` whenever a new plan is created
- Update the checklist in `PRD.md` when work is completed
- Keep `plans/` for detailed implementation plans and `PRD.md` for current status

## Prerequisites

- Python 3.14 or compatible Python 3 installation
- Access to the MAVLink endpoint you want to connect to
- A separate USB serial payload controller if you want the `Payload Operator`
  controls to connect to hardware
- A GStreamer installation with `gst-launch-1.0`
- A GStreamer webcam source plugin such as `avfvideosrc`, `v4l2src`, or `ksvideosrc`
  if you want the live `Camera Feed`
- On macOS/Linux, permission to bind to the configured web port if it is below
  `1024`

## Setup

1. Create the virtual environment:

```bash
python3 -m venv venv
```

2. Activate it:

```bash
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running The App

Run the launcher from the project root:

```bash
./scripts/run.sh
```

By default, the application auto-detects likely MAVLink serial devices and
retries every 5 seconds if no suitable serial device is found.

The payload controller is also auto-detected over USB serial and retries every
5 seconds if it is missing or disconnected.

To use a different system address, pass it as the first argument:

```bash
./scripts/run.sh udp://:14540
```

If the application starts successfully, it serves the web UI at:

```text
http://127.0.0.1:67
```

## Payload Serial Protocol

The payload controller integration uses newline-delimited ASCII commands at
`9600` baud.

Host to controller:

```text
OPEN
CLOSE
STATE?
```

Controller to host:

```text
STATE:OPEN
STATE:CLOSED
ERROR:<message>
```

The web UI only changes the displayed payload state after the controller sends a
confirmed `STATE:...` message.

## YOLOv11 Localization

The repository includes a standalone utility module at `localization_yolov11.py`
for estimating target latitude/longitude from YOLO detections.

It is not wired into the Flask UI yet. It expects:

- a YOLO dataset metadata path at initialization
- drone latitude / longitude at runtime
- altitude in feet relative to home elevation
- pitch / roll / yaw in degrees
- Ultralytics-style detection results

Default camera assumptions:

- field of view: `21.8° x 16.4°`
- resolution: `1920 x 1080`
- fixed camera mount pitch: `45°` downward

Current localization assumptions:

- flat-ground approximation
- box-center anchor point
- yaw required for full geographic localization

## Bounding-Box Center Localization

The repository also includes a simpler utility module at
`localization_bbox_center.py`.

It does not parse YOLO results. Instead, it takes:

- the center pixel of a bounding box
- drone latitude / longitude
- altitude in feet relative to home elevation
- pitch / roll / yaw in degrees

Default camera assumptions match the YOLO helper:

- field of view: `21.8° x 16.4°`
- resolution: `1920 x 1080`
- fixed camera mount pitch: `45°` downward

Use `BoundingBoxCenterLocalizer.estimate_geolocation(...)` for the simplest API.

## Test Firmware

Test controller firmware is provided for both ESP32 and Raspberry Pi Pico:

```text
firmware/esp32_payload_test/esp32_payload_test.ino
firmware/pico_payload_test/pico_payload_test.ino
```

Both targets use the Arduino framework. They only simulate payload state for
integration testing and do not drive the real payload hardware yet.
The Pico sketch starts in the `OPEN` state and mirrors state on the onboard
LED, where LED on means `OPEN`.

## Important Port Note

`app.py` is currently configured to run on port `67`. On macOS and Linux, ports
below `1024` are privileged. Running the app as a normal user will typically
fail with a permission error unless you:

- run it with elevated privileges, or
- change `WEB_PORT` in `app.py` to a non-privileged port such as `5000`

## Troubleshooting

- `mavsdk is not installed`
  Install dependencies with `pip install -r requirements.txt` inside `venv`.

- `Operation not permitted` when starting Flask
  Change `WEB_PORT` in `app.py` or run with sufficient privileges.

- `failed to connect within 15s`
  Verify the MAVSDK system address and confirm the flight controller or
  simulator is reachable.

- App keeps retrying for a MAVLink serial device
  Check that the flight controller is connected over USB, visible to the OS,
  and not already in use by another ground-control or serial program.

- `Payload Operator` stays unavailable
  Check that the payload controller is connected over USB, appears as a serial
  device, speaks the `OPEN` / `CLOSE` / `STATE?` protocol at `9600` baud, and
  is not the same serial device being used for MAVLink.
  On macOS, the payload auto-detect logic now prefers USB serial devices such
  as `usbmodem*` or `usbserial*` and ignores unrelated Bluetooth serial ports.

- `Camera Feed` stays unavailable
  Check that `gst-launch-1.0` is installed, that a webcam source plugin is
  present (`avfvideosrc`, `v4l2src`, or `ksvideosrc`), and that webcam device
  index `1` is valid on the target device.

- Browser shows `waiting for gyro stream...`
  The web server is running, but telemetry has not started streaming yet.
  Check the MAVLink connection and device state.

## Development Notes

- Place project plans in `plans/`
- Update `PRD.md` when a plan is added or a feature is completed
- Keep helper scripts in `scripts/`
