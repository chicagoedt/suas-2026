#!/usr/bin/env python3

import asyncio
import os
import queue
import re
import shutil
import socket
import subprocess
import sys
import threading
import time

from flask import Flask, Response, jsonify, render_template_string

try:
    from mavsdk import System
except ModuleNotFoundError:
    print("Error: mavsdk is not installed. Run: python3 -m pip install mavsdk")
    raise SystemExit(1)

try:
    import serial
    from serial.tools import list_ports
except ModuleNotFoundError:
    print("Error: pyserial is not installed. Run: python3 -m pip install pyserial")
    raise SystemExit(1)


USER_SYSTEM_ADDRESS = sys.argv[1] if len(sys.argv) > 1 else None
DEFAULT_SERIAL_BAUD = 57600
CONNECT_TIMEOUT_S = 15.0
AUTO_DETECT_CONNECT_TIMEOUT_S = 4.0
AUTO_DETECT_RETRY_S = 5.0
AUTO_DETECT_SYSTEM_ADDRESS_LABEL = "auto-detecting serial device"
DEBUG_MODE = True
WEB_PORT = 5002
CAMERA_SENSOR_ID = 1
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FRAMERATE = 30
CAMERA_TCP_HOST = "127.0.0.1"
CAMERA_TCP_PORT = 5600
CAMERA_STREAM_BOUNDARY = "frame"
CAMERA_START_TIMEOUT_S = 3.0
PAYLOAD_SERIAL_BAUD = 9600
PAYLOAD_RETRY_S = 5.0
PAYLOAD_SERIAL_TIMEOUT_S = 0.2
PAYLOAD_HANDSHAKE_TIMEOUT_S = 2.0
PAYLOAD_AUTO_DETECT_LABEL = "auto-detecting payload controller"
PAYLOAD_STATE_OPEN = "open"
PAYLOAD_STATE_CLOSED = "closed"
PAYLOAD_STATE_UNKNOWN = "unknown"

app = Flask(__name__)

_state_lock = threading.Lock()
_telemetry_state = {
    "status": "starting",
    "system_address": USER_SYSTEM_ADDRESS or AUTO_DETECT_SYSTEM_ADDRESS_LABEL,
    "forward_rad_s": None,
    "right_rad_s": None,
    "down_rad_s": None,
    "pitch_deg": None,
    "roll_deg": None,
    "relative_altitude_m": None,
    "last_update_unix_s": None,
    "error": None,
}

_camera_lock = threading.Lock()
_camera_process = None
_camera_state = {
    "status": "idle",
    "pipeline_running": False,
    "source": None,
    "sensor_id": CAMERA_SENSOR_ID,
    "width": CAMERA_WIDTH,
    "height": CAMERA_HEIGHT,
    "error": None,
    "stream_url": "/camera_feed",
}

_payload_lock = threading.Lock()
_payload_command_queue = queue.Queue()
_payload_state = {
    "status": "searching",
    "device": PAYLOAD_AUTO_DETECT_LABEL,
    "payload_state": PAYLOAD_STATE_UNKNOWN,
    "pending_command": None,
    "error": None,
    "last_update_unix_s": None,
}

INDEX_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>UIC Team Air Payload Operator Interface</title>
    <style>
      :root {
        --bg-top: #efe4d2;
        --bg-bottom: #d8e4eb;
        --ink: #1e2831;
        --muted: #5f6972;
        --panel: rgba(255, 252, 247, 0.86);
        --panel-soft: rgba(255, 255, 255, 0.55);
        --border: rgba(30, 40, 49, 0.14);
        --border-strong: rgba(30, 40, 49, 0.32);
        --shadow: 0 28px 60px rgba(52, 64, 76, 0.18);
        --accent: #c96a28;
        --accent-secondary: #28748e;
        --accent-success: #2f7d63;
        --accent-amber: #a66927;
      }

      * {
        box-sizing: border-box;
      }

      html,
      body {
        min-height: 100%;
      }

      body {
        margin: 0;
        font-family: "Trebuchet MS", "Avenir Next", "Segoe UI", sans-serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top left, rgba(255, 255, 255, 0.65), transparent 34%),
          linear-gradient(145deg, var(--bg-top), var(--bg-bottom));
      }

      body::before,
      body::after {
        content: "";
        position: fixed;
        z-index: 0;
        border-radius: 999px;
        filter: blur(20px);
        opacity: 0.45;
        pointer-events: none;
      }

      body::before {
        top: 6vh;
        right: -10vw;
        width: 32vw;
        height: 32vw;
        background: rgba(201, 106, 40, 0.18);
      }

      body::after {
        left: -8vw;
        bottom: -6vh;
        width: 28vw;
        height: 28vw;
        background: rgba(40, 116, 142, 0.18);
      }

      .app-shell {
        position: relative;
        z-index: 1;
        min-height: 100vh;
        padding: clamp(18px, 3vw, 34px);
        display: grid;
        grid-template-rows: auto 1fr;
        gap: clamp(16px, 2vw, 24px);
      }

      .page-header {
        display: flex;
        align-items: end;
        justify-content: space-between;
        gap: 16px;
      }

      .page-kicker,
      .card-kicker {
        margin: 0 0 8px;
        color: var(--muted);
        letter-spacing: 0.24em;
        text-transform: uppercase;
        font-size: 0.72rem;
      }

      h1,
      .panel-title,
      .placeholder-visual,
      .status-pill {
        font-family: "Georgia", "Iowan Old Style", "Palatino Linotype", serif;
      }

      h1 {
        margin: 0;
        font-size: clamp(2.25rem, 4vw, 4.5rem);
        line-height: 0.95;
        letter-spacing: -0.04em;
      }

      .page-subtitle {
        margin: 10px 0 0;
        max-width: 60rem;
        color: var(--muted);
        font-size: 1rem;
      }

      .dashboard-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        grid-template-rows: repeat(2, minmax(300px, 1fr));
        gap: clamp(16px, 2vw, 24px);
        min-height: calc(100vh - 150px);
      }

      .panel {
        --panel-accent: var(--accent-secondary);
        position: relative;
        display: grid;
        grid-template-rows: auto 1fr;
        gap: 16px;
        min-height: 0;
        padding: clamp(18px, 2vw, 24px);
        border: 1px solid var(--border);
        border-radius: 26px;
        background: var(--panel);
        box-shadow: var(--shadow);
        overflow: hidden;
        backdrop-filter: blur(14px);
        animation: rise-in 460ms ease-out both;
      }

      .panel:nth-child(2) {
        animation-delay: 50ms;
      }

      .panel:nth-child(3) {
        animation-delay: 100ms;
      }

      .panel:nth-child(4) {
        animation-delay: 150ms;
      }

      .panel::before {
        content: "";
        position: absolute;
        inset: 0 0 auto 0;
        height: 6px;
        background: linear-gradient(90deg, var(--panel-accent), transparent 72%);
      }

      .panel::after {
        content: "";
        position: absolute;
        inset: auto auto -22% -10%;
        width: 52%;
        height: 52%;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.6), transparent 72%);
        opacity: 0.45;
        pointer-events: none;
      }

      .panel-flight {
        --panel-accent: var(--accent-secondary);
      }

      .panel-camera {
        --panel-accent: var(--accent);
      }

      .panel-payload {
        --panel-accent: var(--accent-success);
      }

      .panel-misc {
        --panel-accent: var(--accent-amber);
      }

      .panel-header {
        display: flex;
        align-items: start;
        justify-content: space-between;
        gap: 16px;
      }

      .panel-title {
        margin: 0;
        font-size: clamp(1.55rem, 2.6vw, 2.15rem);
        line-height: 1;
      }

      .panel-body {
        position: relative;
        z-index: 1;
        min-height: 0;
        display: flex;
        flex-direction: column;
        gap: 16px;
      }

      .status-pill {
        padding: 10px 16px;
        border-radius: 999px;
        border: 1px solid var(--border);
        background: rgba(255, 255, 255, 0.76);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.65);
        font-size: 0.98rem;
        text-align: center;
      }

      .status-pill.is-idle {
        color: var(--muted);
      }

      .status-pill.is-warn {
        color: #8a5718;
        background: rgba(255, 242, 219, 0.92);
      }

      .status-pill.is-live {
        color: #145c71;
        background: rgba(221, 246, 255, 0.92);
      }

      .status-pill.is-error {
        color: #8c2f32;
        background: rgba(255, 226, 226, 0.94);
      }

      .flight-layout {
        display: grid;
        grid-template-columns: minmax(0, 1.2fr) minmax(220px, 0.8fr);
        gap: 16px;
        min-height: 0;
        flex: 1;
      }

      .data-stack {
        display: grid;
        grid-template-rows: auto auto 1fr;
        gap: 14px;
        min-height: 0;
      }

      .metrics-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
      }

      .info-card,
      .metric-card,
      .placeholder-card,
      .misc-row,
      .placeholder-visual {
        border: 1px solid var(--border);
        border-radius: 18px;
        background: var(--panel-soft);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.65);
      }

      .info-card,
      .metric-card,
      .placeholder-card,
      .misc-row {
        padding: 14px 16px;
      }

      .info-label,
      .placeholder-title {
        margin: 0 0 10px;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: var(--muted);
      }

      .system-address {
        display: block;
        font-size: 0.98rem;
        line-height: 1.45;
        word-break: break-word;
      }

      .metric-value {
        margin: 0;
        font-family: "Georgia", "Iowan Old Style", "Palatino Linotype", serif;
        font-size: clamp(1.55rem, 2vw, 2.35rem);
        line-height: 1;
      }

      .metric-caption {
        margin: 10px 0 0;
        color: var(--muted);
        font-size: 0.86rem;
        line-height: 1.35;
      }

      .status-copy {
        margin: 12px 0 0;
        font-size: 1.02rem;
        line-height: 1.45;
        font-weight: 600;
      }

      .grow-card {
        min-height: 0;
        display: flex;
        flex-direction: column;
      }

      #gyro,
      #attitude-meta {
        margin: 0;
        font-family: "SFMono-Regular", "Menlo", "Consolas", monospace;
      }

      #gyro {
        flex: 1;
        min-height: 100%;
        white-space: pre-wrap;
        word-break: break-word;
        font-size: 0.95rem;
        line-height: 1.55;
      }

      .attitude-shell {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 16px;
        padding: 18px;
        border: 1px solid var(--border);
        border-radius: 22px;
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.55), rgba(234, 239, 244, 0.72));
      }

      #attitude-box {
        width: min(100%, 260px);
        aspect-ratio: 1 / 1;
        border: 2px solid var(--border-strong);
        border-radius: 24px;
        background:
          linear-gradient(180deg, rgba(64, 157, 190, 0.15), rgba(64, 157, 190, 0) 42%),
          linear-gradient(0deg, rgba(201, 106, 40, 0.1), rgba(201, 106, 40, 0) 38%),
          linear-gradient(180deg, rgba(255, 255, 255, 0.72), rgba(221, 231, 239, 0.72));
        position: relative;
        overflow: hidden;
      }

      #attitude-box::before {
        content: "";
        position: absolute;
        inset: 12px;
        border-radius: 16px;
        background:
          linear-gradient(180deg, transparent 49%, rgba(30, 40, 49, 0.18) 50%, transparent 51%),
          linear-gradient(90deg, transparent 49%, rgba(30, 40, 49, 0.12) 50%, transparent 51%);
        opacity: 0.65;
      }

      #attitude-line {
        width: 72%;
        height: 4px;
        border-radius: 999px;
        background: linear-gradient(90deg, var(--accent-secondary), var(--accent));
        position: absolute;
        left: 50%;
        top: 50%;
        transform: translate(-50%, -50%);
        transform-origin: center center;
        box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.6);
      }

      .attitude-center {
        position: absolute;
        left: 50%;
        top: 50%;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        border: 3px solid rgba(30, 40, 49, 0.58);
        background: rgba(255, 255, 255, 0.82);
        transform: translate(-50%, -50%);
      }

      #attitude-meta {
        width: 100%;
        padding: 12px 14px;
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.72);
        font-size: 0.92rem;
        text-align: center;
      }

      .placeholder-visual {
        min-height: 180px;
        display: grid;
        place-items: center;
        padding: 20px;
        font-size: clamp(1.35rem, 2vw, 1.9rem);
        line-height: 1.1;
        text-align: center;
      }

      .camera-shell {
        position: relative;
        min-height: 220px;
        border: 1px solid var(--border);
        border-radius: 18px;
        overflow: hidden;
        background:
          linear-gradient(135deg, rgba(201, 106, 40, 0.16), rgba(40, 116, 142, 0.12)),
          repeating-linear-gradient(
            135deg,
            rgba(30, 40, 49, 0.08) 0,
            rgba(30, 40, 49, 0.08) 18px,
            transparent 18px,
            transparent 36px
          );
      }

      .camera-stream {
        width: 100%;
        height: 100%;
        display: block;
        object-fit: cover;
        background: rgba(12, 15, 19, 0.92);
      }

      .camera-empty {
        position: absolute;
        inset: 0;
        display: grid;
        place-items: center;
        padding: 20px;
        text-align: center;
        color: rgba(255, 252, 247, 0.92);
        background: linear-gradient(180deg, rgba(15, 18, 23, 0.3), rgba(15, 18, 23, 0.7));
      }

      .placeholder-copy {
        display: grid;
        gap: 10px;
      }

      .placeholder-copy p,
      .placeholder-card p,
      .misc-row p {
        margin: 0;
      }

      .placeholder-card strong,
      .misc-row strong {
        display: block;
        margin-bottom: 8px;
        font-size: 1.02rem;
      }

      .payload-layout {
        display: grid;
        grid-template-columns: minmax(0, 1.1fr) minmax(240px, 0.9fr);
        gap: 16px;
        min-height: 0;
        flex: 1;
      }

      .payload-stack {
        display: grid;
        grid-template-rows: auto auto auto;
        gap: 12px;
        min-height: 0;
      }

      .payload-actions {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
      }

      .action-button {
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 16px 18px;
        font: inherit;
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--ink);
        background: rgba(255, 255, 255, 0.82);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.72);
        cursor: pointer;
        transition: transform 160ms ease, box-shadow 160ms ease, background 160ms ease;
      }

      .action-button:hover:enabled {
        transform: translateY(-1px);
        box-shadow: 0 10px 22px rgba(30, 40, 49, 0.12);
      }

      .action-button:disabled {
        cursor: not-allowed;
        opacity: 0.5;
        transform: none;
        box-shadow: none;
      }

      .action-button.is-open {
        background: rgba(221, 246, 232, 0.94);
      }

      .action-button.is-close {
        background: rgba(255, 240, 221, 0.94);
      }

      .payload-state-shell {
        min-height: 0;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 14px;
        padding: 18px;
        border: 1px solid var(--border);
        border-radius: 22px;
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.55), rgba(234, 244, 239, 0.72));
        text-align: center;
      }

      .payload-badge {
        min-width: min(100%, 220px);
        padding: 18px 22px;
        border-radius: 20px;
        font-family: "Georgia", "Iowan Old Style", "Palatino Linotype", serif;
        font-size: clamp(1.8rem, 2.8vw, 2.8rem);
        line-height: 1;
        border: 1px solid var(--border);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.72);
      }

      .payload-badge.is-open {
        color: #166046;
        background: rgba(222, 246, 232, 0.96);
      }

      .payload-badge.is-closed {
        color: #8a5718;
        background: rgba(255, 242, 219, 0.96);
      }

      .payload-badge.is-unknown {
        color: var(--muted);
        background: rgba(255, 255, 255, 0.82);
      }

      .payload-note {
        margin: 0;
        font-size: 0.98rem;
        line-height: 1.45;
      }

      .misc-stack {
        display: grid;
        gap: 12px;
      }

      @keyframes rise-in {
        from {
          opacity: 0;
          transform: translateY(18px);
        }

        to {
          opacity: 1;
          transform: translateY(0);
        }
      }

      @media (max-width: 1040px) {
        .dashboard-grid {
          grid-template-columns: 1fr;
          grid-template-rows: none;
          min-height: auto;
        }

        .flight-layout,
        .payload-layout {
          grid-template-columns: 1fr;
        }

        .page-header {
          align-items: start;
        }
      }

      @media (max-width: 680px) {
        .app-shell {
          padding: 16px;
        }

        .panel-header {
          flex-direction: column;
          align-items: start;
        }

        .status-pill {
          width: 100%;
        }

        .payload-actions {
          grid-template-columns: 1fr;
        }

        .placeholder-visual {
          min-height: 140px;
        }
      }
    </style>
  </head>
  <body>
    <main class="app-shell">
      <header class="page-header">
        <div>
          <p class="page-kicker">Mission Console</p>
          <h1>Pay Operator Interface</h1>
          <p class="page-subtitle">
            Wide-screen operator dashboard for flight telemetry, camera operations,
            payload workflow, and auxiliary mission tools.
          </p>
        </div>
      </header>

      <section class="dashboard-grid">
        <section class="panel panel-flight">
          <div class="panel-header">
            <div>
              <h2 class="panel-title">Flight Data</h2>
            </div>
            <div id="status-pill" class="status-pill is-idle">Starting</div>
          </div>
          <div class="panel-body flight-layout">
            <div class="data-stack">
              <div class="metrics-grid">
                <article class="metric-card">
                  <p class="info-label">Pitch</p>
                  <p id="pitch-primary" class="metric-value">waiting</p>
                  <p class="metric-caption">Euler pitch from MAVSDK attitude telemetry.</p>
                </article>
                <article class="metric-card">
                  <p class="info-label">Altitude</p>
                  <p id="altitude-primary" class="metric-value">waiting</p>
                  <p class="metric-caption">Relative altitude above the home/takeoff reference, shown in feet.</p>
                </article>
              </div>
              <div class="info-card">
                <p class="info-label">Vehicle Link</p>
                <code id="system-address" class="system-address">{{ system_address }}</code>
                <p id="status" class="status-copy">status: starting...</p>
              </div>
              <div class="info-card grow-card">
                <p class="info-label">Gyroscope Stream</p>
                <pre id="gyro">waiting for data...</pre>
              </div>
            </div>
            <div class="attitude-shell">
              <div id="attitude-box">
                <div class="attitude-center"></div>
                <div id="attitude-line"></div>
              </div>
              <div id="attitude-meta">pitch_deg: waiting | roll_deg: waiting</div>
            </div>
          </div>
        </section>

        <section class="panel panel-camera">
          <div class="panel-header">
            <div>
              <h2 class="panel-title">Camera Feed</h2>
            </div>
          </div>
          <div class="panel-body">
            <div class="camera-shell">
              <img id="camera-stream" class="camera-stream" alt="Sensor 1 camera stream" />
              <div id="camera-empty" class="camera-empty">Starting camera feed...</div>
            </div>
            <div class="placeholder-copy">
              <p class="placeholder-title">Video Source</p>
              <p id="camera-status">Initializing GStreamer camera pipeline...</p>
              <p id="camera-detail">Webcam device 1, 640x480, MJPEG over GStreamer.</p>
            </div>
          </div>
        </section>

        <section class="panel panel-payload">
          <div class="panel-header">
            <div>
              <h2 class="panel-title">Payload Operator</h2>
            </div>
          </div>
          <div class="panel-body payload-layout">
            <div class="payload-stack">
              <div class="info-card">
                <p class="info-label">Controller Link</p>
                <code id="payload-device" class="system-address">auto-detecting payload controller</code>
                <p id="payload-connection" class="status-copy">status: searching...</p>
              </div>
              <div class="info-card">
                <p class="info-label">Operator Status</p>
                <p id="payload-note" class="payload-note">Waiting for payload controller response...</p>
              </div>
              <div class="payload-actions">
                <button id="payload-open-button" class="action-button is-open" type="button" disabled>Open</button>
                <button id="payload-close-button" class="action-button is-close" type="button" disabled>Close</button>
              </div>
            </div>
            <div class="payload-state-shell">
              <p class="info-label">Payload State</p>
              <div id="payload-state-badge" class="payload-badge is-unknown">Unknown</div>
              <p id="payload-state-caption" class="metric-caption">Awaiting confirmed payload state from the controller.</p>
            </div>
          </div>
        </section>

        <section class="panel panel-misc">
          <div class="panel-header">
            <div>
              <h2 class="panel-title">Misc</h2>
            </div>
          </div>
          <div class="panel-body">
            <div class="misc-stack">
              <article class="misc-row">
                <p class="card-kicker">Utilities</p>
                <strong>Auxiliary Widgets</strong>
                <p>Reserve this area for mission timers, notes, or secondary indicators.</p>
              </article>
              <article class="misc-row">
                <p class="card-kicker">Diagnostics</p>
                <strong>System Messages</strong>
                <p>Use this block for non-flight-critical alerts or operator-facing debug info.</p>
              </article>
              <article class="misc-row">
                <p class="card-kicker">Expansion</p>
                <strong>Future Modules</strong>
                <p>Keep this panel flexible for additional tooling that does not fit the other sections.</p>
              </article>
            </div>
          </div>
        </section>
      </section>
    </main>

    <script>
      const MAX_PITCH_DEG = 45.0;
      const MAX_PITCH_OFFSET_PX = 60.0;
      const METERS_TO_FEET = 3.28084;
      let cameraRetryTimer = null;

      function getStatusTone(statusText, hasError) {
        const lowered = String(statusText || "").toLowerCase();

        if (hasError || lowered.includes("error") || lowered.includes("failed")) {
          return "is-error";
        }

        if (lowered.includes("stream") || lowered.includes("connected")) {
          return "is-live";
        }

        if (lowered.includes("connect")) {
          return "is-warn";
        }

        return "is-idle";
      }

      function getStatusBadge(statusText, hasError) {
        const lowered = String(statusText || "").toLowerCase();

        if (hasError || lowered.includes("error") || lowered.includes("failed")) {
          return "Attention";
        }

        if (lowered.includes("stream")) {
          return "Live";
        }

        if (lowered.includes("connected")) {
          return "Connected";
        }

        if (lowered.includes("connect")) {
          return "Connecting";
        }

        return "Idle";
      }

      function setStatus(statusText, hasError) {
        const pill = document.getElementById("status-pill");
        const tone = getStatusTone(statusText, hasError);

        document.getElementById("status").innerText = "status: " + statusText;
        pill.innerText = getStatusBadge(statusText, hasError);
        pill.className = "status-pill " + tone;
      }

      function clamp(value, min, max) {
        return Math.min(max, Math.max(min, value));
      }

      function formatPrimaryValue(value, decimals, unit) {
        if (value === null || value === undefined) {
          return "waiting";
        }

        return Number(value).toFixed(decimals) + " " + unit;
      }

      function metersToFeet(value) {
        if (value === null || value === undefined) {
          return null;
        }

        return Number(value) * METERS_TO_FEET;
      }

      function showCameraPlaceholder(message) {
        const placeholder = document.getElementById("camera-empty");
        placeholder.hidden = false;
        placeholder.innerText = message;
      }

      function hideCameraPlaceholder() {
        document.getElementById("camera-empty").hidden = true;
      }

      function startCameraStream() {
        document.getElementById("camera-stream").src = "/camera_feed?ts=" + Date.now();
      }

      function scheduleCameraRetry() {
        if (cameraRetryTimer !== null) {
          return;
        }

        cameraRetryTimer = setTimeout(() => {
          cameraRetryTimer = null;
          startCameraStream();
        }, 2000);
      }

      async function updateCameraStatus() {
        try {
          const response = await fetch("/camera_status");

          if (!response.ok) {
            throw new Error("HTTP " + response.status);
          }

          const data = await response.json();
          document.getElementById("camera-status").innerText = data.status;

          if (data.error) {
            document.getElementById("camera-detail").innerText = data.error;
            showCameraPlaceholder("Camera feed unavailable");
            return;
          }

          document.getElementById("camera-detail").innerText =
            (data.source || "webcam") + " | device " + data.sensor_id + " | " +
            data.width + "x" + data.height + " | GStreamer MJPEG";

          if (!data.pipeline_running) {
            showCameraPlaceholder("Starting camera feed...");
          }
        } catch (err) {
          document.getElementById("camera-status").innerText = "camera status unavailable";
          document.getElementById("camera-detail").innerText = String(err);
          showCameraPlaceholder("Waiting for camera status...");
        }
      }

      function getPayloadStateLabel(payloadState) {
        if (payloadState === "open") {
          return "Open";
        }

        if (payloadState === "closed") {
          return "Closed";
        }

        return "Unknown";
      }

      function getPayloadStateClass(payloadState) {
        if (payloadState === "open") {
          return "payload-badge is-open";
        }

        if (payloadState === "closed") {
          return "payload-badge is-closed";
        }

        return "payload-badge is-unknown";
      }

      function getPayloadCaption(data) {
        if (data.pending_command) {
          return "Command sent. Waiting for confirmed payload state from the controller.";
        }

        if (data.payload_state === "open") {
          return "Reported open by the payload controller.";
        }

        if (data.payload_state === "closed") {
          return "Reported closed by the payload controller.";
        }

        if (data.status === "connected") {
          return "Connected, but no confirmed payload state has been reported yet.";
        }

        return "Awaiting confirmed payload state from the controller.";
      }

      function getPayloadNote(data) {
        if (data.error) {
          return data.error;
        }

        if (data.pending_command) {
          return "Waiting for controller to confirm " + String(data.pending_command).toUpperCase() + ".";
        }

        if (data.status === "connected") {
          return "Controller connected and ready for OPEN/CLOSE commands.";
        }

        return "Searching for a USB payload controller...";
      }

      function setPayloadButtons(data) {
        const openButton = document.getElementById("payload-open-button");
        const closeButton = document.getElementById("payload-close-button");
        const connected = data.status === "connected";
        const pending = Boolean(data.pending_command);

        openButton.disabled = !connected || pending || data.payload_state === "open";
        closeButton.disabled = !connected || pending || data.payload_state === "closed";
      }

      async function updatePayloadStatus() {
        try {
          const response = await fetch("/payload_status");

          if (!response.ok) {
            throw new Error("HTTP " + response.status);
          }

          const data = await response.json();
          document.getElementById("payload-device").innerText =
            data.device || "auto-detecting payload controller";
          document.getElementById("payload-connection").innerText =
            "status: " + (data.status || "searching");
          document.getElementById("payload-note").innerText = getPayloadNote(data);
          document.getElementById("payload-state-badge").innerText =
            getPayloadStateLabel(data.payload_state);
          document.getElementById("payload-state-badge").className =
            getPayloadStateClass(data.payload_state);
          document.getElementById("payload-state-caption").innerText =
            getPayloadCaption(data);
          setPayloadButtons(data);
        } catch (err) {
          document.getElementById("payload-connection").innerText = "status: unavailable";
          document.getElementById("payload-note").innerText = String(err);
          document.getElementById("payload-state-badge").innerText = "Unknown";
          document.getElementById("payload-state-badge").className = "payload-badge is-unknown";
          document.getElementById("payload-state-caption").innerText =
            "Payload controller status could not be loaded.";
          document.getElementById("payload-open-button").disabled = true;
          document.getElementById("payload-close-button").disabled = true;
        }
      }

      async function sendPayloadCommand(routePath) {
        try {
          const response = await fetch(routePath, {method: "POST"});
          let data = {};

          try {
            data = await response.json();
          } catch (parseErr) {
            data = {};
          }

          if (!response.ok) {
            throw new Error(data.error || data.status || ("HTTP " + response.status));
          }

          if (data.status) {
            document.getElementById("payload-note").innerText = data.status;
          }

          await updatePayloadStatus();
        } catch (err) {
          document.getElementById("payload-note").innerText = String(err);
          await updatePayloadStatus();
        }
      }

      async function updateFlightData() {
        try {
          const response = await fetch("/gyro");

          if (!response.ok) {
            throw new Error("HTTP " + response.status);
          }

          const data = await response.json();
          setStatus(data.status, Boolean(data.error));
          document.getElementById("system-address").innerText =
            data.system_address || "auto-detecting serial device";

          document.getElementById("pitch-primary").innerText =
            formatPrimaryValue(data.pitch_deg, 1, "deg");
          document.getElementById("altitude-primary").innerText =
            formatPrimaryValue(metersToFeet(data.relative_altitude_m), 1, "ft");

          if (data.error) {
            document.getElementById("gyro").innerText = "error: " + data.error;
            return;
          }

          if (data.forward_rad_s === null) {
            document.getElementById("gyro").innerText = "waiting for gyro stream...";
          } else {
            document.getElementById("gyro").innerText =
              "forward_rad_s: " + Number(data.forward_rad_s).toFixed(4) + "\\n" +
              "right_rad_s:   " + Number(data.right_rad_s).toFixed(4) + "\\n" +
              "down_rad_s:    " + Number(data.down_rad_s).toFixed(4);
          }

          if (data.pitch_deg === null || data.roll_deg === null) {
            document.getElementById("attitude-meta").innerText = "pitch_deg: waiting | roll_deg: waiting";
            return;
          }

          const pitchDeg = Number(data.pitch_deg);
          const rollDeg = Number(data.roll_deg);
          const pitchOffsetPx = clamp(
            (pitchDeg / MAX_PITCH_DEG) * MAX_PITCH_OFFSET_PX,
            -MAX_PITCH_OFFSET_PX,
            MAX_PITCH_OFFSET_PX
          );

          document.getElementById("attitude-line").style.transform =
            `translate(-50%, -50%) translateY(${pitchOffsetPx.toFixed(1)}px) rotate(${rollDeg.toFixed(2)}deg)`;

          document.getElementById("attitude-meta").innerText =
            "pitch_deg: " + pitchDeg.toFixed(2) + " | roll_deg: " + rollDeg.toFixed(2);
        } catch (err) {
          setStatus("fetch error", true);
          document.getElementById("gyro").innerText = String(err);
        }
      }

      document.getElementById("camera-stream").addEventListener("load", hideCameraPlaceholder);
      document.getElementById("camera-stream").addEventListener("error", () => {
        showCameraPlaceholder("Retrying camera feed...");
        scheduleCameraRetry();
      });
      document.getElementById("payload-open-button").addEventListener("click", () => {
        sendPayloadCommand("/payload/open");
      });
      document.getElementById("payload-close-button").addEventListener("click", () => {
        sendPayloadCommand("/payload/close");
      });

      startCameraStream();
      updateCameraStatus();
      setInterval(updateCameraStatus, 1000);
      updatePayloadStatus();
      setInterval(updatePayloadStatus, 500);
      updateFlightData();
      setInterval(updateFlightData, 200);
    </script>
  </body>
</html>
"""


def _set_state(**kwargs) -> None:
    with _state_lock:
        _telemetry_state.update(kwargs)


def _set_camera_state(**kwargs) -> None:
    with _camera_lock:
        _camera_state.update(kwargs)


def _set_payload_state(**kwargs) -> None:
    with _payload_lock:
        _payload_state.update(kwargs)


def _clear_payload_command_queue() -> None:
    while True:
        try:
            _payload_command_queue.get_nowait()
        except queue.Empty:
            return


def _current_flight_controller_device() -> str | None:
    with _state_lock:
        system_address = str(_telemetry_state.get("system_address") or "")

    match = re.fullmatch(r"serial://(.+):(\d+)", system_address)
    if not match:
        return None

    return match.group(1)


def _parse_payload_line(line: str) -> tuple[str | None, str | None]:
    normalized = line.strip()
    if not normalized:
        return None, None

    upper = normalized.upper()
    if upper == "STATE:OPEN":
        return "state", PAYLOAD_STATE_OPEN
    if upper == "STATE:CLOSED":
        return "state", PAYLOAD_STATE_CLOSED
    if upper.startswith("ERROR:"):
        return "error", normalized.split(":", 1)[1].strip() or "unknown controller error"

    return None, None


def _looks_like_usb_microcontroller_port(port) -> bool:
    device = str(getattr(port, "device", ""))
    device_lower = device.lower()
    text = _serial_port_text(port)

    macos_usb_prefixes = (
        "/dev/cu.usbmodem",
        "/dev/tty.usbmodem",
        "/dev/cu.usbserial",
        "/dev/tty.usbserial",
        "/dev/cu.slab_usbtouart",
        "/dev/tty.slab_usbtouart",
        "/dev/cu.wchusbserial",
        "/dev/tty.wchusbserial",
    )
    microcontroller_tokens = (
        "arduino",
        "esp32",
        "pico",
        "rp2040",
        "raspberry pi",
        "ch340",
        "wch",
        "cp210",
        "ftdi",
        "usb serial",
        "usb uart",
        "silicon labs",
        "cdc",
    )

    if any(device_lower.startswith(prefix) for prefix in macos_usb_prefixes):
        return True

    if device_lower.startswith("com") or "ttyacm" in device_lower or "ttyusb" in device_lower:
        return True

    if any(token in text or token in device_lower for token in microcontroller_tokens):
        return True

    return getattr(port, "vid", None) is not None or getattr(port, "pid", None) is not None


def _read_payload_line(payload_serial: serial.Serial) -> str | None:
    raw = payload_serial.readline()
    if not raw:
        return None

    line = raw.decode("utf-8", errors="ignore").strip()
    return line or None


def _write_payload_line(payload_serial: serial.Serial, command: str) -> None:
    payload_serial.write(f"{command}\n".encode("ascii"))
    payload_serial.flush()


def _handle_payload_message(message_type: str | None, value: str | None, *, device: str) -> bool:
    if message_type == "state" and value:
        _set_payload_state(
            status="connected",
            device=device,
            payload_state=value,
            pending_command=None,
            error=None,
            last_update_unix_s=time.time(),
        )
        return True

    if message_type == "error":
        _set_payload_state(
            status="connected",
            device=device,
            pending_command=None,
            error=value or "controller error",
            last_update_unix_s=time.time(),
        )

    return False


def _payload_port_priority(port) -> tuple[int, str]:
    device = str(getattr(port, "device", ""))
    device_lower = device.lower()
    text = _serial_port_text(port)
    score = 0

    controller_tokens = (
        "arduino",
        "esp32",
        "pico",
        "rp2040",
        "raspberry pi",
        "ch340",
        "wch",
        "cp210",
        "ftdi",
        "usb serial",
        "usb uart",
        "cdc",
        "silicon labs",
    )
    usb_transport_tokens = (
        "usbmodem",
        "ttyacm",
        "ttyusb",
        "com",
    )
    flight_controller_tokens = (
        "mavlink",
        "pixhawk",
        "cube",
        "ardupilot",
        "px4",
        "flight controller",
        "autopilot",
    )

    for token in controller_tokens:
        if token in text or token in device_lower:
            score += 60

    for token in usb_transport_tokens:
        if token in text or token in device_lower:
            score += 25

    for token in flight_controller_tokens:
        if token in text:
            score -= 200

    if getattr(port, "vid", None) is not None or getattr(port, "pid", None) is not None:
        score += 5

    if device_lower.startswith("/dev/cu.usbmodem") or device_lower.startswith("/dev/tty.usbmodem"):
        score += 40

    return (-score, device_lower)


def _exclude_payload_port(port) -> bool:
    if _exclude_serial_port(port):
        return True

    device = str(getattr(port, "device", ""))
    if not device:
        return True

    flight_controller_device = _current_flight_controller_device()
    if flight_controller_device and device == flight_controller_device:
        return True

    if sys.platform == "darwin" and not _looks_like_usb_microcontroller_port(port):
        return True

    text = _serial_port_text(port)
    flight_controller_tokens = (
        "mavlink",
        "pixhawk",
        "cube",
        "ardupilot",
        "px4",
        "flight controller",
        "autopilot",
    )
    return any(token in text for token in flight_controller_tokens)


def _iter_payload_candidates():
    ports = [port for port in list_ports.comports() if not _exclude_payload_port(port)]
    return sorted(ports, key=_payload_port_priority)


def _perform_payload_handshake(payload_serial: serial.Serial, device: str) -> None:
    _write_payload_line(payload_serial, "STATE?")
    deadline = time.time() + PAYLOAD_HANDSHAKE_TIMEOUT_S

    while time.time() < deadline:
        line = _read_payload_line(payload_serial)
        if line is None:
            continue

        message_type, value = _parse_payload_line(line)
        if message_type == "state":
            _handle_payload_message(message_type, value, device=device)
            return
        if message_type == "error":
            raise RuntimeError(f"{device} rejected handshake: {value}")

    raise RuntimeError(f"{device} did not report payload state")


def _payload_session(device: str) -> None:
    with serial.Serial(
        device,
        PAYLOAD_SERIAL_BAUD,
        timeout=PAYLOAD_SERIAL_TIMEOUT_S,
        write_timeout=PAYLOAD_SERIAL_TIMEOUT_S,
    ) as payload_serial:
        try:
            payload_serial.reset_input_buffer()
            payload_serial.reset_output_buffer()
        except Exception:
            pass

        _set_payload_state(
            status="connected",
            device=device,
            payload_state=PAYLOAD_STATE_UNKNOWN,
            pending_command=None,
            error=None,
            last_update_unix_s=time.time(),
        )
        _perform_payload_handshake(payload_serial, device)

        while True:
            try:
                queued_command = _payload_command_queue.get_nowait()
            except queue.Empty:
                queued_command = None

            if queued_command:
                _write_payload_line(payload_serial, queued_command)
                _set_payload_state(
                    status="connected",
                    device=device,
                    pending_command=queued_command.lower(),
                    error=None,
                    last_update_unix_s=time.time(),
                )

            line = _read_payload_line(payload_serial)
            if line is None:
                continue

            message_type, value = _parse_payload_line(line)
            _handle_payload_message(message_type, value, device=device)


def payload_task() -> None:
    while True:
        candidate_ports = _iter_payload_candidates()

        if not candidate_ports:
            _clear_payload_command_queue()
            _set_payload_state(
                status="searching",
                device=PAYLOAD_AUTO_DETECT_LABEL,
                payload_state=PAYLOAD_STATE_UNKNOWN,
                pending_command=None,
                error=f"no payload controller found; retrying in {PAYLOAD_RETRY_S:.0f}s",
                last_update_unix_s=time.time(),
            )
            time.sleep(PAYLOAD_RETRY_S)
            continue

        last_error = "no payload controller responded"

        for port in candidate_ports:
            device = str(getattr(port, "device", ""))
            _clear_payload_command_queue()
            _set_payload_state(
                status="searching",
                device=device,
                payload_state=PAYLOAD_STATE_UNKNOWN,
                pending_command=None,
                error=f"probing {device}",
                last_update_unix_s=time.time(),
            )

            try:
                _payload_session(device)
            except (serial.SerialException, OSError, RuntimeError) as exc:
                last_error = f"{device}: {exc}"
                continue

            last_error = f"{device}: payload session ended"

        _clear_payload_command_queue()
        _set_payload_state(
            status="error",
            device=PAYLOAD_AUTO_DETECT_LABEL,
            payload_state=PAYLOAD_STATE_UNKNOWN,
            pending_command=None,
            error=f"{last_error}; retrying in {PAYLOAD_RETRY_S:.0f}s",
            last_update_unix_s=time.time(),
        )
        time.sleep(PAYLOAD_RETRY_S)


def _gst_plugin_available(plugin_name: str) -> bool:
    gst_inspect = shutil.which("gst-inspect-1.0")
    if not gst_inspect:
        return False

    result = subprocess.run(
        [gst_inspect, plugin_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _camera_command():
    gst_launch = shutil.which("gst-launch-1.0")
    if not gst_launch:
        return None, None, "gst-launch-1.0 is not installed"

    candidates = [
        ("avfvideosrc", ["avfvideosrc", f"device-index={CAMERA_SENSOR_ID}"]),
        ("v4l2src", ["v4l2src", f"device=/dev/video{CAMERA_SENSOR_ID}"]),
        ("ksvideosrc", ["ksvideosrc", f"device-index={CAMERA_SENSOR_ID}"]),
        ("nvarguscamerasrc", ["nvarguscamerasrc", f"sensor-id={CAMERA_SENSOR_ID}"]),
    ]

    for source_name, source_args in candidates:
        if not _gst_plugin_available(source_name):
            continue

        if source_name == "nvarguscamerasrc":
            pipeline = [
                *source_args,
                "!",
                f"video/x-raw(memory:NVMM),width={CAMERA_WIDTH},height={CAMERA_HEIGHT},framerate={CAMERA_FRAMERATE}/1",
                "!",
                "nvvidconv",
                "!",
                f"video/x-raw,width={CAMERA_WIDTH},height={CAMERA_HEIGHT},format=I420",
            ]
        else:
            pipeline = [
                *source_args,
                "!",
                f"video/x-raw,width={CAMERA_WIDTH},height={CAMERA_HEIGHT},framerate={CAMERA_FRAMERATE}/1",
                "!",
                "videoconvert",
                "!",
                "video/x-raw,format=I420",
            ]

        return (
            source_name,
            [
                gst_launch,
                "-q",
                *pipeline,
                "!",
                "jpegenc",
                "!",
                "multipartmux",
                f"boundary={CAMERA_STREAM_BOUNDARY}",
                "!",
                "tcpserversink",
                f"host={CAMERA_TCP_HOST}",
                f"port={CAMERA_TCP_PORT}",
                "sync=false",
            ],
            None,
        )

    return (
        None,
        None,
        "no supported GStreamer camera source is available (tried avfvideosrc, v4l2src, ksvideosrc, nvarguscamerasrc)",
    )


def _camera_socket_ready(timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection((CAMERA_TCP_HOST, CAMERA_TCP_PORT), timeout=0.25):
                return True
        except OSError:
            time.sleep(0.1)

    return False


def _watch_camera_process(process: subprocess.Popen) -> None:
    global _camera_process

    stderr_output = ""
    if process.stderr is not None:
        stderr_output = process.stderr.read().strip()

    return_code = process.wait()
    error_text = stderr_output.splitlines()[-1] if stderr_output else f"gstreamer exited with code {return_code}"

    with _camera_lock:
        if _camera_process is process:
            _camera_process = None
        _camera_state.update(
            status="camera pipeline stopped",
            pipeline_running=False,
            error=error_text,
        )


def ensure_camera_pipeline_started() -> bool:
    global _camera_process

    with _camera_lock:
        process = _camera_process

        if process is None or process.poll() is not None:
            source_name, command, command_error = _camera_command()
            if command is None:
                _camera_state.update(
                    status="camera unavailable",
                    pipeline_running=False,
                    error=command_error,
                )
                return False

            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            except Exception as exc:
                _camera_process = None
                _camera_state.update(
                    status="camera unavailable",
                    pipeline_running=False,
                    error=f"failed to start gstreamer: {exc}",
                )
                return False

            _camera_process = process
            _camera_state.update(
                status=f"starting {source_name} device {CAMERA_SENSOR_ID} at {CAMERA_WIDTH}x{CAMERA_HEIGHT}",
                pipeline_running=False,
                source=source_name,
                error=None,
            )
            threading.Thread(target=_watch_camera_process, args=(process,), daemon=True).start()

    if _camera_socket_ready(CAMERA_START_TIMEOUT_S):
        _set_camera_state(status="streaming", pipeline_running=True, error=None)
        return True

    with _camera_lock:
        process = _camera_process
        if process is not None and process.poll() is None:
            _camera_state.update(
                status="starting camera pipeline",
                pipeline_running=False,
                error=None,
            )

    return False


def _serial_port_text(port) -> str:
    fields = (
        getattr(port, "device", ""),
        getattr(port, "name", ""),
        getattr(port, "description", ""),
        getattr(port, "manufacturer", ""),
        getattr(port, "product", ""),
        getattr(port, "interface", ""),
        getattr(port, "hwid", ""),
    )
    return " ".join(str(field) for field in fields if field).lower()


def _exclude_serial_port(port) -> bool:
    text = _serial_port_text(port)
    excluded_tokens = (
        "bluetooth",
        "wireless iap",
        "debug console",
        "debug-console",
    )
    return any(token in text for token in excluded_tokens)


def _serial_port_priority(port) -> tuple[int, str]:
    device = str(getattr(port, "device", ""))
    device_lower = device.lower()
    text = _serial_port_text(port)
    score = 0

    high_confidence_tokens = (
        "mavlink",
        "pixhawk",
        "cube",
        "ardupilot",
        "px4",
        "flight controller",
        "autopilot",
    )
    usb_serial_tokens = (
        "usbmodem",
        "usbserial",
        "ttyacm",
        "ttyusb",
        "usb serial",
        "serial port",
        "cp210",
        "ch340",
        "ftdi",
        "stm32",
        "silicon labs",
    )

    for token in high_confidence_tokens:
        if token in text:
            score += 100

    for token in usb_serial_tokens:
        if token in text or token in device_lower:
            score += 25

    if device_lower.startswith("com"):
        score += 10

    if getattr(port, "vid", None) is not None or getattr(port, "pid", None) is not None:
        score += 5

    return (-score, device_lower)


def _iter_serial_candidates():
    ports = [port for port in list_ports.comports() if not _exclude_serial_port(port)]
    return sorted(ports, key=_serial_port_priority)


def _serial_system_address(device: str) -> str:
    return f"serial://{device}:{DEFAULT_SERIAL_BAUD}"


async def wait_until_connected(drone: System) -> None:
    async for state in drone.core.connection_state():
        if state.is_connected:
            return


async def connect_with_timeout(system_address: str, timeout_s: float) -> System:
    drone = System()
    await drone.connect(system_address=system_address)
    await asyncio.wait_for(wait_until_connected(drone), timeout=timeout_s)
    return drone


async def stream_imu(drone: System) -> None:
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


async def stream_attitude(drone: System) -> None:
    try:
        async for attitude in drone.telemetry.attitude_euler():
            _set_state(
                status="streaming",
                pitch_deg=attitude.pitch_deg,
                roll_deg=attitude.roll_deg,
                last_update_unix_s=time.time(),
                error=None,
            )
    except Exception as exc:
        _set_state(status="error", error=f"failed while reading attitude data: {exc}")


async def stream_position(drone: System) -> None:
    try:
        async for position in drone.telemetry.position():
            _set_state(
                status="streaming",
                relative_altitude_m=position.relative_altitude_m,
                last_update_unix_s=time.time(),
                error=None,
            )
    except Exception as exc:
        _set_state(status="error", error=f"failed while reading altitude data: {exc}")


async def auto_detect_serial_connection() -> tuple[System, str]:
    while True:
        candidate_ports = _iter_serial_candidates()

        if not candidate_ports:
            _set_state(
                status="searching for MAVLink serial device",
                system_address=AUTO_DETECT_SYSTEM_ADDRESS_LABEL,
                error=f"no MAVLink serial device found; retrying in {AUTO_DETECT_RETRY_S:.0f}s",
            )
            await asyncio.sleep(AUTO_DETECT_RETRY_S)
            continue

        last_error = "no MAVLink serial device responded"

        for port in candidate_ports:
            system_address = _serial_system_address(port.device)
            _set_state(status=f"probing {port.device}", system_address=system_address, error=None)

            try:
                drone = await connect_with_timeout(system_address, AUTO_DETECT_CONNECT_TIMEOUT_S)
                return drone, system_address
            except asyncio.TimeoutError:
                last_error = f"no MAVLink response from {port.device}"
            except Exception as exc:
                last_error = f"{port.device}: {exc}"

        _set_state(
            status="searching for MAVLink serial device",
            system_address=AUTO_DETECT_SYSTEM_ADDRESS_LABEL,
            error=f"{last_error}; retrying in {AUTO_DETECT_RETRY_S:.0f}s",
        )
        await asyncio.sleep(AUTO_DETECT_RETRY_S)


async def telemetry_task() -> None:
    if USER_SYSTEM_ADDRESS:
        system_address = USER_SYSTEM_ADDRESS
        _set_state(status=f"connecting to {system_address}", system_address=system_address, error=None)

        try:
            drone = await connect_with_timeout(system_address, CONNECT_TIMEOUT_S)
        except asyncio.TimeoutError:
            _set_state(status="error", error=f"failed to connect within {CONNECT_TIMEOUT_S:.0f}s")
            return
        except Exception as exc:
            _set_state(status="error", error=f"failed to connect: {exc}")
            return
    else:
        drone, system_address = await auto_detect_serial_connection()

    _set_state(status="connected", system_address=system_address, error=None)

    # Try to request a reasonable IMU stream rate; continue even if unsupported.
    rate_messages = []
    try:
        await drone.telemetry.set_rate_imu(20.0)
    except Exception as exc:
        rate_messages.append(f"could not set IMU rate: {exc}")

    try:
        await drone.telemetry.set_rate_attitude_euler(20.0)
    except Exception as exc:
        rate_messages.append(f"could not set attitude rate: {exc}")

    try:
        await drone.telemetry.set_rate_position(10.0)
    except Exception as exc:
        rate_messages.append(f"could not set position rate: {exc}")

    if rate_messages:
        _set_state(status="connected (rate defaulted)", error="; ".join(rate_messages))

    await asyncio.gather(stream_imu(drone), stream_attitude(drone), stream_position(drone))


def start_telemetry_thread() -> None:
    thread = threading.Thread(target=lambda: asyncio.run(telemetry_task()), daemon=True)
    thread.start()


def start_payload_thread() -> None:
    thread = threading.Thread(target=payload_task, daemon=True)
    thread.start()


@app.route("/")
def index():
    return render_template_string(
        INDEX_HTML,
        system_address=USER_SYSTEM_ADDRESS or AUTO_DETECT_SYSTEM_ADDRESS_LABEL,
    )


@app.route("/gyro")
def gyro():
    with _state_lock:
        return jsonify(dict(_telemetry_state))


@app.route("/payload_status")
def payload_status():
    with _payload_lock:
        return jsonify(dict(_payload_state))


def _queue_payload_command(target_state: str, wire_command: str):
    with _payload_lock:
        status = str(_payload_state.get("status") or "")
        current_state = str(_payload_state.get("payload_state") or PAYLOAD_STATE_UNKNOWN)
        pending_command = _payload_state.get("pending_command")

    if status != "connected":
        return (
            jsonify({"error": "payload controller not connected", "status": "payload controller unavailable"}),
            503,
        )

    if pending_command:
        return jsonify({"status": f"{pending_command} command already pending"}), 200

    if current_state == target_state:
        return jsonify({"status": f"payload already {target_state}"}), 200

    _payload_command_queue.put_nowait(wire_command)
    _set_payload_state(
        pending_command=target_state,
        error=None,
        last_update_unix_s=time.time(),
    )
    return jsonify({"status": f"{target_state} command queued"}), 202


@app.route("/payload/open", methods=["POST"])
def payload_open():
    return _queue_payload_command(PAYLOAD_STATE_OPEN, "OPEN")


@app.route("/payload/close", methods=["POST"])
def payload_close():
    return _queue_payload_command(PAYLOAD_STATE_CLOSED, "CLOSE")


@app.route("/camera_status")
def camera_status():
    with _camera_lock:
        return jsonify(dict(_camera_state))


@app.route("/camera_feed")
def camera_feed():
    if not ensure_camera_pipeline_started():
        return Response("camera feed unavailable", status=503, mimetype="text/plain")

    def generate():
        try:
            with socket.create_connection((CAMERA_TCP_HOST, CAMERA_TCP_PORT), timeout=CAMERA_START_TIMEOUT_S) as camera_socket:
                while True:
                    chunk = camera_socket.recv(65536)
                    if not chunk:
                        break
                    yield chunk
        except OSError as exc:
            _set_camera_state(
                status="camera pipeline stopped",
                pipeline_running=False,
                error=f"camera stream disconnected: {exc}",
            )

    return Response(
        generate(),
        mimetype=f"multipart/x-mixed-replace; boundary={CAMERA_STREAM_BOUNDARY}",
    )


if __name__ == "__main__":
    should_start_telemetry = not DEBUG_MODE or os.environ.get("WERKZEUG_RUN_MAIN") == "true"

    if should_start_telemetry:
        start_telemetry_thread()
        start_payload_thread()
        print(f"Web UI running on http://127.0.0.1:{WEB_PORT}")

    app.run(host="0.0.0.0", port=WEB_PORT, debug=DEBUG_MODE)
