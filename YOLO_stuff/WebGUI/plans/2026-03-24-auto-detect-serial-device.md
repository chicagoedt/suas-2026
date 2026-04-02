# Plan: Cross-Platform MAVLink Serial Auto-Detect

## Summary

Replace the hardcoded serial device path with cross-platform serial-port
enumeration and automatic MAVLink probing.

- Use serial-port discovery on macOS, Linux, and Windows
- Keep a CLI system-address override for manual use
- Retry every 5 seconds when no MAVLink serial device is found
- Keep the existing MAVSDK telemetry streams once connected

## Key Changes

### Connection flow

- Add serial-port enumeration using `pyserial`
- Score and sort likely MAVLink-capable ports before probing them
- Probe each candidate as `serial://<device>:57600`
- Retry the scan every 5 seconds if no candidate responds to MAVLink
- Preserve manual `system_address` CLI override behavior

### Flight Data UI

- Make the displayed system address live instead of fixed at initial page render
- Show probing / searching status while auto-detect is running
- Keep the existing pitch, altitude, gyro, and attitude indicators once a device connects

### Runtime safety

- Keep the debug-mode guard that prevents duplicate MAVSDK threads under Flask reloads
- Add the new `pyserial` dependency to `requirements.txt`
- Update `run.sh` so it installs dependencies if `serial` is missing

## Test Plan

- Confirm the app still compiles
- Confirm `/gyro` exposes the current `system_address`
- Confirm the rendered page updates the displayed system address from live state
- Confirm missing-device states surface a retrying message every 5 seconds
- Confirm CLI override still bypasses auto-detect

## Assumptions

- MAVLink serial devices use `57600` baud by default in this project
- Port-name and USB metadata heuristics are acceptable for candidate ordering
- Auto-detect should keep retrying indefinitely until a MAVLink serial device is found
