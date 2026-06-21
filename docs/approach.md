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
5. **Smoothing** — centred moving average over each track's WGS84 coordinates to
   suppress GNSS/pixel jitter (see *Smoothing* below).
6. **Visualization** — folium/Leaflet map + optional annotated video.

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

## Smoothing
Geo-referenced tracks carry two independent noise sources — the drone's GNSS
(metres without RTK) and per-frame detection-box jitter — which make plotted
paths zig-zag and inflate path length and speed. A centred moving average
(`geo/smoothing.py`, window set by `processing.smoothing_window`) low-passes each
track's WGS84 coordinates before the GeoJSON and map are written. Smoothing is
done in **geo space**, never in pixel space: a parked car's pixels still travel
as the drone flies and yaws, so only the projected ground positions isolate the
true jitter. Latitude and longitude are filtered independently; the window is
symmetric and shrinks to a single point at each end, so the first and last fixes
stay anchored — **net displacement (and thus the moving/stationary split) is
preserved**, while interior jitter, path length and the jitter-inflated speed are
reduced. A constant-velocity Kalman/RTS smoother is the natural upgrade if a
motion model is wanted; the moving average needs no tuning, no extra dependency
and invents no positions.

## Visualization
The map (`outputs/map.html`, folium/Leaflet) carries both an OpenStreetMap and an
Esri satellite base layer (toggle via the layer control). Each vehicle is one
polyline; tracks are classified by net displacement (`pyproj` geodesic):
- **Moving** (displacement >= `moving_min_displacement_m`, default 3 m): solid
  coloured path with green start / red end markers and a popup carrying track id,
  class, net displacement, path length and **mean speed** (km/h).
- **Stationary** (parked cars whose track is only GNSS/pixel jitter): faint grey
  dashed line, no markers.

Mean speed is along-path geodesic distance over the elapsed SRT time
(`metrics.track_speed`, computed on the smoothed track) and is also written to
the GeoJSON `mean_speed_kmh` property. Smoothing removes most of the jitter that
would otherwise inflate it, but at low displacement the residual still dominates,
so it is meaningful for moving traffic and only indicative for near-stationary
tracks.

This satisfies both the "moving cars" framing and the "paths of all detected
cars" deliverable without discarding any track.

## Accuracy
Accuracy is validated without ground-control points or RTK — none are needed to
*bound* the error — by separating the two independent contributors and quantifying
each (`geo/error_budget.py`, `geo/metrics.py`).

**Relative (path-shape) accuracy** is set by the projection geometry alone. With
perfect telemetry the same ground feature imaged from two different drone poses
reprojects to the same WGS84 coordinate to sub-centimetre
(`test_same_ground_feature_reprojects_consistently_across_poses`), so geometry
adds no cross-frame error. In the field, `reprojection_scatter_m` turns this into
a GCP-free check: the spread of a *stationary* feature's reprojected positions
across frames is a direct, location-agnostic measurement of relative accuracy.

**Absolute accuracy** is bounded by an analytical error budget. Independent terms
combine in quadrature (`root_sum_square`):

| Term | Model | At 60 m / r=30 m | At 120 m / r=85 m |
|------|-------|------------------|-------------------|
| Gimbal tilt residual (0.1°) | `alt · tan(δ)` | ~0.10 m | ~0.21 m |
| Altitude error (1%) | `r · ε` | ~0.30 m | ~0.85 m |
| Focal/HFOV miscal. (1%) | `r · ε` | ~0.30 m | ~0.85 m |
| Heading error (0.5°) | `2r · sin(δ/2)` | ~0.26 m | ~0.74 m |
| **Geometry RSS** | | **~0.5 m** | **~1.4 m** |
| Platform GNSS (no RTK) | 1:1 | ~1–3 m | ~1–3 m |

The geometry is sub-metre across the operating envelope; the platform GNSS (1–3 m
horizontal for commercial drones without RTK, per manufacturer specs) dominates
and is the same magnitude regardless of altitude. **Conclusion:** sub-metre
*absolute* accuracy is not attainable from telemetry alone — it is GNSS-bound, and
would require RTK or ground-control points — whereas *relative* path accuracy is
finer, limited only by the geometry above. This is the factual basis for the
"up to ~1 m" target being a relative, not absolute, figure.

## Performance
Throughput is measured per stage rather than as a single opaque number, because
the per-stage split is what drives the cost/quality trade-off. `perf.py` wraps the
four swappable stages (decode, detect, track, project) in timing decorators and
runs the *real* pipeline through them — the dependency-injection seams added for
testing double as zero-overhead measurement points, so no benchmarking branch
leaks into `pipeline.run`. Run it on any clip:

```bash
dvt --video flight.MP4 --srt flight.SRT --benchmark
```

This prints frames processed, total wall time, throughput (fps) and the
decode/detect/track/project breakdown. Detection dominates wall time by a wide
margin; decode is secondary and tracking and geo-referencing are comparatively
negligible. The practical lever is `io.frame_stride`: processing every *n*-th
frame scales throughput roughly linearly while sampling vehicle positions more
coarsely — adequate while the inter-frame motion stays well below the track-
association gate. Absolute fps is hardware- and clip-dependent, so it is left to
the local run rather than quoted here.

## Results
_Map screenshot — run locally; outputs are gitignored to keep source imagery and
location out of the public repo. The accuracy figures above are reproducible from
any DJI SRT via `geo/error_budget.py` and `geo/metrics.reprojection_scatter_m`;
the performance breakdown via `dvt --benchmark`._
