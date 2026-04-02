# Plan: Add MAVSDK Altitude to Flight Data

## Summary

Extend the existing `Flight Data` implementation in `app.py` rather than
rebuilding it.

- Keep the current MAVSDK connection flow and reuse the existing telemetry thread
- Keep the current `pitch_deg` source from `telemetry.attitude_euler()`
- Add relative altitude above home/takeoff reference using MAVSDK position telemetry
- Keep the current extras in `Flight Data`:
  - connection state
  - gyro readout
  - attitude indicator
- Update `plans/` and `PRD.md` as part of the implementation

## Key Changes

### Telemetry backend

- Extend `_telemetry_state` with `relative_altitude_m`
- Add a new MAVSDK stream coroutine for position/altitude data, using
  `drone.telemetry.position()`
- Read `relative_altitude_m` from the MAVSDK position object and write it into
  shared state
- Attempt to set a reasonable position stream rate alongside the existing
  IMU/attitude rate setup; continue if unsupported
- Keep the current `/gyro` route name for compatibility, but expand its JSON
  payload with altitude data

### Flight Data panel

- Keep the current panel layout and live polling model
- Add a dedicated altitude readout in `Flight Data`, using meters and a waiting
  state until data arrives
- Keep the existing pitch/roll attitude indicator and gyro text block
- Add pitch and altitude as the primary flight values so they are immediately
  visible without reading the gyro block
- Preserve current connection/error handling behavior; if altitude is
  unavailable, show a clear waiting/error state without breaking pitch updates

### Frontend data handling

- Extend the existing `updateFlightData()` logic to consume
  `relative_altitude_m`
- Render altitude with a fixed precision suitable for operator display
- Keep current status-pill behavior and fetch cadence
- Do not add a new frontend polling endpoint or change panel names

### Project tracking

- Add a new implementation plan file under `plans/` for the altitude addition
- Update `PRD.md` with the new completed work item and latest plan summary

## Test Plan

- Confirm Flask route `/` still renders the dashboard and `Flight Data` panel
- Confirm `/gyro` still returns existing fields and now includes
  `relative_altitude_m`
- Confirm `Flight Data` shows:
  - connection state
  - pitch
  - relative altitude
  - existing gyro data
  - existing attitude indicator
- Confirm waiting states render correctly when altitude data has not arrived yet
- Confirm connection failures and unsupported rate-setting still surface as
  readable status/error text
- Run Python syntax validation on `app.py` and route-level checks with the Flask
  test client

## Assumptions

- Altitude means MAVSDK relative altitude above the takeoff/home reference, not
  AMSL
- The existing `/gyro` route remains the transport for `Flight Data` to avoid
  unnecessary route churn
- `Flight Data` stays richer than a minimal pitch/altitude-only panel
- No changes are made to the app port, debug mode, or the other three dashboard
  panels
