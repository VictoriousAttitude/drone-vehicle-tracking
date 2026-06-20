# Approach

## Problem
Detect and track moving vehicles in nadir drone video (DJI Mavic 3T) and plot
their real-world paths (WGS84) on an interactive map, using the per-frame flight
telemetry embedded in the DJI SRT file.

## Pipeline
1. **Telemetry parse** — DJI SRT -> per-frame drone pose (lat, lon, alt, gimbal).
2. **Detection** — YOLO, vehicle classes only (see *Detection model* below).
3. **Tracking** — ByteTrack -> stable IDs and pixel trajectories.
4. **Geo-referencing** — per-frame projection: pixel -> ground offset (ray cast,
   reduces to GSD * pixel_delta at nadir) -> rotate by gimbal yaw -> add to the
   drone's WGS84 position. Implemented per point with that point's own frame pose.
5. **Visualization** — folium/Leaflet map + optional annotated video.

## Detection model
COCO-pretrained YOLO is trained on ground-level imagery and does not generalise
to a nadir (top-down) viewpoint: on aerial test frames it returned no vehicles,
only spurious non-vehicle classes. Switching to VisDrone-finetuned weights and
raising the inference resolution (`imgsz` 1280, since vehicles are small at
survey altitude) restores reliable detection. Detection is decoupled behind the
`Detector` protocol, so swapping weights is a one-line config change.

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

## Visualization
The map (`outputs/map.html`, folium/Leaflet) carries both an OpenStreetMap and an
Esri satellite base layer (toggle via the layer control). Each vehicle is one
polyline; tracks are classified by net displacement (`pyproj` geodesic):
- **Moving** (displacement >= `moving_min_displacement_m`, default 3 m): solid
  coloured path with green start / red end markers and a popup carrying track id,
  class, net displacement and path length.
- **Stationary** (parked cars whose track is only GNSS/pixel jitter): faint grey
  dashed line, no markers.

This satisfies both the "moving cars" framing and the "paths of all detected
cars" deliverable without discarding any track.

## Results
_TBD — map screenshot and accuracy notes (run locally; outputs are gitignored to
keep source imagery and location out of the public repo)._
