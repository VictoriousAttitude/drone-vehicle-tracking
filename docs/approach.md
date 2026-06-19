# Approach

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

## Operating regime
- Designed for near-nadir survey flights (gimbal pitch close to -90 deg), where a
  flat-ground projection is well conditioned.
- The drone yaws freely, so the pixel->ground transform is recomputed per frame
  from the SRT pose rather than assumed constant.
- For the Mavic 3T wide camera (HFOV ~71.5 deg) at typical survey altitudes
  (~60-120 m), ground sampling distance is on the order of a few cm/px, so the
  limiting factor for absolute accuracy is the platform's GNSS, not pixels.

## Assumptions & limitations
- Flat-terrain assumption (no DEM) — revisit if sub-metre *absolute* accuracy is
  required over sloped terrain.
- Absolute accuracy is bounded by the drone's GNSS accuracy; without RTK/GCPs,
  sub-metre absolute is not achievable, while relative path accuracy is finer.
- Effective focal length / HFOV to be refined by empirical calibration against a
  known ground feature.

## Results
_TBD — map screenshots, demo video, accuracy notes._
