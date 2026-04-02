# Plan: Raspberry Pi Pico Payload Test Script

## Summary

Add a Raspberry Pi Pico test sketch that speaks the same payload serial
protocol as the existing ESP32 test firmware.

- Target the Arduino IDE on Raspberry Pi Pico
- Keep the same `OPEN`, `CLOSE`, and `STATE?` protocol
- Return `STATE:OPEN`, `STATE:CLOSED`, and `ERROR:UNKNOWN_COMMAND`
- Do not add real payload hardware logic in this version

## Key Changes

- Add `firmware/pico_payload_test/pico_payload_test.ino`
- Read newline-delimited commands from USB serial
- Maintain an in-memory `OPEN` / `CLOSED` state only
- Write protocol responses back over USB serial
- Use the onboard LED to reflect the current simulated payload state

## Test Plan

- Confirm the Pico script file exists in the firmware tree
- Confirm the command handlers match the existing payload protocol
- Do not compile, flash, or run the script in this pass

## Assumptions

- The Pico target uses the Arduino IDE and USB serial
- The host app talks to the Pico using the same ASCII payload protocol as the ESP32 test sketch
