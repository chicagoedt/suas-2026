# Plan: GStreamer Camera Feed for Sensor 1

## Summary

Replace the placeholder `Camera Feed` panel with a live GStreamer-backed webcam
stream.

- Use webcam device index `1`
- Stream at 640x480
- Deliver the feed to the browser as MJPEG through Flask
- Expose camera status for the UI and troubleshooting

## Key Changes

### Camera pipeline

- Start a GStreamer pipeline lazily when the browser requests the feed
- Use `gst-launch-1.0` with a webcam source plugin for the current platform
  (`avfvideosrc`, `v4l2src`, or `ksvideosrc`), with `nvarguscamerasrc` as a
  fallback if that is the only available source
- Encode with `jpegenc` and `multipartmux`
- Publish the multipart MJPEG stream on a local TCP socket and proxy it through
  Flask

### Web UI

- Replace the placeholder camera panel with a live `<img>` stream
- Add camera status text and a retrying empty-state overlay
- Retry the browser stream request if the feed is temporarily unavailable

### Runtime behavior

- Add a `/camera_status` endpoint for live pipeline state
- Keep clear error reporting if GStreamer or the camera plugin is unavailable
- Leave the rest of the dashboard unchanged

## Test Plan

- Confirm `app.py` still compiles
- Confirm `/camera_status` returns camera state
- Confirm the `Camera Feed` panel renders the live stream element
- Confirm the browser retries when the feed endpoint returns an error
- Confirm the UI surfaces camera errors when no supported webcam source plugin
  is available

## Assumptions

- Webcam device index `1` and 640x480 are fixed for this implementation
- Jetson `nvarguscamerasrc` is a fallback, not the preferred source
- MJPEG over Flask is sufficient for the current operator UI
