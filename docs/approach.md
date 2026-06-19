# Approach

> Documentation deliverable: approach, tools, assumptions, results. Filled in as
> the pipeline is implemented.

## Problem
Detect and track moving vehicles in nadir drone video (DJI Mavic 3T) and plot
their real-world paths (WGS84) on an interactive map, using the per-frame flight
telemetry embedded in the DJI SRT file.

## Pipeline
1. **Telemetry parse** — DJI SRT -> per-frame drone pose (lat, lon, alt, gimbal).
2. **Detection** — YOLO, vehicle classes only.
3. **Tracking** — ByteTrack -> stable IDs and pixel trajectories.
4. **Geo-referencing** — per-frame nadir projection: pixel -> ground offset
   (via GSD) -> rotate by gimbal yaw -> add to drone WGS84 position.
5. **Visualization** — folium/Leaflet map + optional annotated video.

## Key facts (from the provided data)
- Video: 1920x1080, 29.97 fps, 166 s, 4979 frames.
- Camera near-perfect nadir for the whole flight (gimbal pitch -89.9..-90 deg).
- Drone moves ~1.4 km, altitude 60..102 m, full yaw range -> per-frame transform.
- GSD ~4.5-7.6 cm/px -> pixel localization is far finer than the 1 m target.

## Assumptions & limitations
- Flat-terrain assumption (no DEM) — revisit if 1 m accuracy is *absolute*.
- Absolute accuracy is bounded by the drone's GNSS accuracy (RTK status TBC).
- Effective focal length / HFOV to be refined by empirical calibration.

## Results
_TBD — map screenshots, demo video, accuracy notes._
