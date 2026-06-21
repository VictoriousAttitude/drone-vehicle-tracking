# Approach

## Problem
Detect and track moving vehicles in nadir drone video (DJI Mavic 3T) and plot
their real-world paths (WGS84) on an interactive map, using the per-frame flight
telemetry embedded in the DJI SRT file.

## Pipeline
1. **Telemetry parse** — DJI SRT -> per-frame drone pose (lat, lon, alt, gimbal).
2. **Detection** — YOLO, vehicle classes only (see *Detection model* below).
3. **Tracking** — ByteTrack -> stable IDs and pixel trajectories; weak tracks are
   then dropped by length and mean detection confidence (see *Track quality*).
4. **Geo-referencing** — per-frame projection: pixel -> ground offset (ray cast,
   reduces to GSD * pixel_delta at nadir) -> rotate by gimbal yaw -> add to the
   drone's WGS84 position. Implemented per point with that point's own frame pose.
5. **Smoothing** — centred moving average over each track's WGS84 coordinates to
   suppress GNSS/pixel jitter (see *Smoothing* below).
6. **Visualization & export** — folium/Leaflet map, GeoJSON, optional annotated
   video, and optional Cursor-on-Target (CoT) XML for TAK (see *Export* below).

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

## Track quality
`min_track_length` already drops tracks too *short* to trust, but a track can
clear that gate while being built mostly from borderline detections just above
the per-frame `conf_threshold`. The detector confidence is carried end-to-end
onto each track point (verified to survive ByteTrack), so an aggregate gate is
available: tracks whose **mean** confidence falls below `min_track_confidence`
are dropped, and the mean is surfaced on the GeoJSON (`mean_confidence`) and the
map popup so weak tracks are visible rather than silently kept. Tracks that carry
no confidence (e.g. injected for testing) cannot be assessed and are kept, so the
filter degrades to a no-op rather than discarding data it cannot judge.

Heavier track-quality work — interpolating across detection gaps and re-ID across
full occlusions — is deliberately left out: both add real complexity (and, for
re-ID, an appearance model) for marginal gain on near-nadir survey footage where
vehicles are rarely occluded. They are the natural next step if needed.

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
  class, net displacement, path length, **mean speed** (km/h) and the
  self-reported position accuracy (see *Export* below).
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

## Export (CoT / TAK)
Beyond the map and GeoJSON, tracks export to **Cursor-on-Target** (`export/cot.py`,
written with `dvt --cot tracks.cot`) — the XML event format ingested by TAK
clients and servers (ATAK/WinTAK), so the output feeds a common operating picture
directly. Each track becomes one CoT `<event>` at its last known position: a
`<point>` (lat/lon), a `<track>` carrying course (last-segment forward azimuth)
and speed, a `<contact>` callsign and a `<remarks>` line with class and mean
confidence. Type defaults to `a-u-G` (MIL-STD-2525 atom / unknown affiliation /
Ground), since a detector cannot establish friend-or-foe.

**Self-reported accuracy** rides in the standard CoT `ce` (circular error, metres)
field and is also written to the GeoJSON `position_error_m` property and the map
popup, so every consumer sees the position's stated uncertainty. Rather than a
single configured constant, this is the **per-point geometry-aware** figure: the
error-budget terms below (tilt, altitude/focal scale, heading) are evaluated at
each point's own altitude and radial ground offset from nadir — derived from the
pixel geometry (`radius_px · GSD`) — and combined in quadrature with the GNSS
floor (`export.position_error_m`, default 3 m). So `ce` correctly grows toward the
frame edge and with altitude, and is reported per track as the conservative
**worst-case (maximum)** over its points. When a point has no telemetry the figure
degrades to the flat GNSS floor. It is honest by construction — the geometry is
sub-metre but the platform GNSS is not. `hae` and `le` are emitted as the CoT
"unknown" sentinel because ground elevation is not modelled under the flat-ground
projection.

The MISB analogue for moving-target tracks is **ST 0903 (VMTI)**, which is binary
KLV multiplexed into an ST 0601 video stream — a heavier, video-coupled artefact
(needs a KLV codec and a mux step). CoT is the practical, immediately-ingestible
format; VMTI is the natural next step if a MISB-compliant video feed is required.

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

This budget is not only an analytical table: `geo/error_budget.AccuracyModel`
evaluates the same terms **per point at runtime** (using that frame's altitude and
the pixel's radial offset from nadir) and the result is surfaced as the
self-reported accuracy on every track (CoT `ce`, GeoJSON, map) — see *Export*
above. The coefficients (gimbal-tilt residual, altitude/focal scale, heading and
the GNSS floor) live in the config's `accuracy` and `export` sections, so the
claim is reproducible and tunable rather than hard-coded.

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
