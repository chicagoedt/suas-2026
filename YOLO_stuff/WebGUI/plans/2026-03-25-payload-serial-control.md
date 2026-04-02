# Plan: Payload Serial Control and ESP32 Test Firmware

## Summary

Replace the placeholder `Payload Operator` panel with live controls backed by a
USB serial payload controller.

- Auto-detect a separate Arduino, ESP32, or Pico-style USB serial device
- Send `OPEN` and `CLOSE` commands over serial
- Read confirmed `STATE:OPEN` and `STATE:CLOSED` updates from the controller
- Add a test ESP32 Arduino sketch that simulates payload state only

## Key Changes

### Payload controller backend

- Add a dedicated payload serial worker thread in `app.py`
- Probe likely microcontroller USB serial ports and retry every 5 seconds
- Exclude the MAVLink flight-controller serial port from payload detection
- Use a `STATE?` handshake at `9600` baud before marking a controller connected
- Expose `GET /payload_status`, `POST /payload/open`, and `POST /payload/close`

### Payload Operator UI

- Replace the placeholder cards with:
  - payload controller connection status
  - detected serial device
  - confirmed payload state badge
  - `Open` and `Close` buttons
  - operator-facing status/error text
- Keep the displayed state confirmation-driven rather than optimistic
- Poll payload status every 500 ms so controller-originated changes appear in
  the UI

### ESP32 test firmware

- Add `firmware/esp32_payload_test/esp32_payload_test.ino`
- Implement newline-delimited ASCII commands:
  - `OPEN`
  - `CLOSE`
  - `STATE?`
- Respond with:
  - `STATE:OPEN`
  - `STATE:CLOSED`
  - `ERROR:UNKNOWN_COMMAND`
- Do not implement real payload hardware actuation in this version

## Test Plan

- Confirm `app.py` still compiles
- Confirm the payload endpoints return the expected status and command responses
- Confirm the payload panel renders real controls instead of placeholders
- Confirm the ESP32 test sketch parses `OPEN`, `CLOSE`, and `STATE?`
- Confirm disconnected-controller behavior reports retrying status cleanly

## Assumptions

- The payload controller is a separate USB serial device from the flight
  controller
- The firmware target for the test controller is an ESP32 Arduino sketch
- The app still uses polling and a single-process Flask server
