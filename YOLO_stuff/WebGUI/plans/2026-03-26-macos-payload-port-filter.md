# Plan: Tighten macOS Payload Port Filtering

## Summary

Prevent the payload-controller auto-detect flow on macOS from probing unrelated
Bluetooth or virtual serial devices.

- Restrict macOS payload probing to USB microcontroller-style serial devices
- Prefer Pico-style `usbmodem` ports
- Keep existing cross-platform payload auto-detect behavior intact

## Key Changes

- Add a helper in `app.py` to recognize USB microcontroller serial ports
- On macOS, exclude non-USB `cu.*` or `tty.*` devices from payload probing
- Prioritize `usbmodem` ports higher than generic serial candidates

## Test Plan

- Confirm the Pico USB serial port remains in the payload candidate list
- Confirm Bluetooth and other non-USB serial ports are excluded on macOS
- Confirm `app.py` still compiles after the filter change

## Assumptions

- The Pico appears on macOS as a `usbmodem`-style USB serial device
- Filtering is scoped to payload-controller detection, not the MAVLink detector
