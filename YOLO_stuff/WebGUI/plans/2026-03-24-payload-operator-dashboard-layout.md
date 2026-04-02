# Plan: 2x2 Payload Operator Dashboard Layout

## Summary

Refactor the inline HTML/CSS/JS in `app.py` from a single telemetry view into a
full-width dashboard with four titled panels.

- Change the browser page title and visible page heading to `Pay Operator Interface`
- Replace the current single-column body with a 2x2 dashboard layout
- Keep all existing MAVSDK telemetry behavior and `/gyro` polling logic
- Move the current gyroscope/attitude UI into the top-left `Flight Data` panel
- Render the other three panels as styled placeholders for now:
  - top-right: `Camera Feed`
  - bottom-left: `Payload Operator`
  - bottom-right: `Misc`

## Key Changes

### UI structure

- Update `INDEX_HTML` in `app.py` to use a dashboard shell:
  - compact page header with `Pay Operator Interface`
  - two-column, two-row panel grid that spans the available page width
  - equal-size cards/panels on desktop
- Use exact panel placement and titles:
  - top-left: `Flight Data`
  - top-right: `Camera Feed`
  - bottom-left: `Payload Operator`
  - bottom-right: `Misc`
- Remove the old standalone `MAVSDK Gyroscope Test` heading and title text

### Flight Data panel

- Move the existing live telemetry elements into the `Flight Data` panel:
  - system address
  - connection/status text
  - attitude box / horizon line
  - pitch/roll summary
  - gyro readout
- Keep the current JS update loop and `/gyro` fetch contract unchanged
- Keep current telemetry state keys unchanged; no backend API changes

### Placeholder panels

- Add non-interactive placeholder content for `Camera Feed`, `Payload Operator`,
  and `Misc`
- Use simple empty-state copy such as `Coming soon` / `No source configured`
- Do not add new endpoints, video streams, or payload actions in this pass

### Styling and responsiveness

- Replace the minimal inline styles with a panel-based dashboard style
- Make the layout use full page width and a balanced two-column grid on desktop
- Add a mobile fallback that stacks panels vertically on narrow screens
- Preserve the existing attitude indicator behavior inside the Flight Data panel

## Test Plan

- Load `/` and confirm the page title is `Pay Operator Interface`
- Confirm four panels render in the correct positions with the correct titles
- Confirm `Flight Data` continues polling `/gyro` and updates:
  - status text
  - gyro values
  - pitch/roll text
  - attitude indicator transform
- Confirm `Camera Feed`, `Payload Operator`, and `Misc` render placeholder content without JS errors
- Confirm the dashboard uses full width on desktop and stacks cleanly on smaller screens
- Confirm no changes are required to `/gyro` response shape or MAVSDK telemetry threading

## Assumptions

- `Pay Operator Interface` is intentional and should be used exactly as written
  for the page title/header
- `Flight Data` will use title case for consistency
- This is a layout/UI refactor only; no live camera feed, payload controls, or
  misc integrations are included yet
- Implementation remains inside `app.py` using the existing inline `INDEX_HTML`
  approach rather than moving to a separate template file
