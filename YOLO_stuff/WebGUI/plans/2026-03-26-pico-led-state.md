# Plan: Pico LED State Mirror

## Summary

Update the Raspberry Pi Pico demo sketch so the onboard LED mirrors payload
state and the controller starts in the `OPEN` state.

- Default startup state is `OPEN`
- LED on means `OPEN`
- LED off means `CLOSED`
- Keep the existing serial protocol unchanged

## Key Changes

- Initialize the Pico payload state as `OPEN`
- Drive `LED_BUILTIN` from the current payload state
- Update the LED whenever `OPEN` or `CLOSE` is received

## Test Plan

- Confirm the sketch still exposes the same `OPEN`, `CLOSE`, and `STATE?` protocol
- Confirm startup state in the source is `OPEN`
- Confirm LED updates are tied to payload state transitions
- Do not compile or flash the sketch in this pass

## Assumptions

- The selected Arduino IDE Pico core supports `LED_BUILTIN`
- `HIGH` means LED on for the target Pico board
