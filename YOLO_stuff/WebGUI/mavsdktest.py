#!/usr/bin/env python3

import asyncio
import sys
import threading
import time

from flask import Flask, jsonify, render_template_string

try:
    from mavsdk import System
except ModuleNotFoundError:
    print("Error: mavsdk is not installed. Run: python3 -m pip install mavsdk")
    raise SystemExit(1)


SYSTEM_ADDRESS = sys.argv[1] if len(sys.argv) > 1 else "serial:///dev/tty.usbmodem11201:57600"
CONNECT_TIMEOUT_S = 15.0
WEB_PORT = 67

app = Flask(__name__)

_state_lock = threading.Lock()
_telemetry_state = {
    "status": "starting",
    "system_address": SYSTEM_ADDRESS,
    "forward_rad_s": None,
    "right_rad_s": None,
    "down_rad_s": None,
    "last_update_unix_s": None,
    "error": None,
}

INDEX_HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>MAVSDK Gyroscope Test</title>
  </head>
  <body>
    <h2>MAVSDK Gyroscope Test</h2>
    <p>System address: <code>{{ system_address }}</code></p>
    <p id="status">status: starting...</p>
    <pre id="gyro">waiting for data...</pre>

    <script>
      async function updateGyro() {
        try {
          const response = await fetch("/gyro");
          const data = await response.json();
          document.getElementById("status").innerText = "status: " + data.status;

          if (data.error) {
            document.getElementById("gyro").innerText = "error: " + data.error;
            return;
          }

          if (data.forward_rad_s === null) {
            document.getElementById("gyro").innerText = "waiting for gyro stream...";
            return;
          }

          document.getElementById("gyro").innerText =
            "forward_rad_s: " + Number(data.forward_rad_s).toFixed(4) + "\\n" +
            "right_rad_s:   " + Number(data.right_rad_s).toFixed(4) + "\\n" +
            "down_rad_s:    " + Number(data.down_rad_s).toFixed(4);
        } catch (err) {
          document.getElementById("status").innerText = "status: fetch error";
          document.getElementById("gyro").innerText = String(err);
        }
      }

      updateGyro();
      setInterval(updateGyro, 200);
    </script>
  </body>
</html>
"""


def _set_state(**kwargs) -> None:
    with _state_lock:
        _telemetry_state.update(kwargs)


async def wait_until_connected(drone: System) -> None:
    async for state in drone.core.connection_state():
        if state.is_connected:
            return


async def telemetry_task() -> None:
    drone = System()
    _set_state(status=f"connecting to {SYSTEM_ADDRESS}", error=None)

    try:
        await drone.connect(system_address=SYSTEM_ADDRESS)
        await asyncio.wait_for(wait_until_connected(drone), timeout=CONNECT_TIMEOUT_S)
    except asyncio.TimeoutError:
        _set_state(status="error", error=f"failed to connect within {CONNECT_TIMEOUT_S:.0f}s")
        return
    except Exception as exc:
        _set_state(status="error", error=f"failed to connect: {exc}")
        return

    _set_state(status="connected")

    # Try to request a reasonable IMU stream rate; continue even if unsupported.
    try:
        await drone.telemetry.set_rate_imu(20.0)
    except Exception as exc:
        _set_state(status="connected (imu rate defaulted)", error=f"could not set IMU rate: {exc}")

    try:
        async for imu in drone.telemetry.imu():
            gyro = imu.angular_velocity_frd
            _set_state(
                status="streaming",
                forward_rad_s=gyro.forward_rad_s,
                right_rad_s=gyro.right_rad_s,
                down_rad_s=gyro.down_rad_s,
                last_update_unix_s=time.time(),
                error=None,
            )
    except Exception as exc:
        _set_state(status="error", error=f"failed while reading gyroscope data: {exc}")


def start_telemetry_thread() -> None:
    thread = threading.Thread(target=lambda: asyncio.run(telemetry_task()), daemon=True)
    thread.start()


@app.route("/")
def index():
    return render_template_string(INDEX_HTML, system_address=SYSTEM_ADDRESS)


@app.route("/gyro")
def gyro():
    with _state_lock:
        return jsonify(dict(_telemetry_state))


if __name__ == "__main__":
    start_telemetry_thread()
    print(f"Web UI running on http://127.0.0.1:{WEB_PORT}")
    app.run(host="0.0.0.0", port=WEB_PORT, debug=False)
