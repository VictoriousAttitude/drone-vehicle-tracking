# Drone Vehicle Tracking

Detect, track and **geo-reference** moving vehicles from nadir drone video, then
plot their real-world paths (WGS84) on an interactive map. Built around the
per-frame flight telemetry embedded in DJI `.SRT` files.

> Status: work in progress. Telemetry parsing and the geometry core are in
> place; detection/tracking/projection are being implemented.

## Demo

_TODO: map screenshot + short demo clip._

## How it works

```
SRT telemetry ─┐
               ├─> detect (YOLO) ─> track (ByteTrack) ─> geo-reference ─> map / video
video frames  ─┘
```

1. **Telemetry** — parse DJI SRT into per-frame drone pose (lat, lon, altitude, gimbal).
2. **Detection** — YOLO restricted to vehicle classes (`car`, `truck`, `bus`).
3. **Tracking** — ByteTrack assigns stable IDs -> per-vehicle pixel trajectories.
4. **Geo-referencing** — per-frame nadir projection maps each pixel to WGS84
   using the drone position, altitude and gimbal yaw.
5. **Visualization** — interactive folium/Leaflet map + optional annotated video.

See [`docs/approach.md`](docs/approach.md) for the method, assumptions and accuracy notes.

## Install

```bash
pip install -e ".[cv,dev]"   # core + CV/DL + dev tooling
```

The pure-Python core (telemetry, geometry, visualization) installs without the
heavy `cv` extra; `cv` adds OpenCV + Ultralytics for detection/tracking.

## Usage

```bash
dvt --video data/flight.MP4 --srt data/flight.SRT --config configs/default.yaml
```

## Project layout

```
src/drone_vehicle_tracking/
  telemetry/   # DJI SRT parsing + shared domain models
  detection/   # YOLO vehicle detector
  tracking/    # ByteTrack wrapper
  geo/         # camera model, GSD, nadir pixel->WGS84 projection
  visualization/  # folium map + annotated video
  pipeline.py  # orchestration
  cli.py       # entrypoint
tests/         # unit (pure logic) + integration (needs real data)
configs/       # all tunables
```

## Development

```bash
ruff check . && ruff format --check .
mypy
pytest -m "not integration"
```

## Data & privacy

Input video/telemetry are **not** committed (see `.gitignore`). Only a tiny
synthetic SRT fixture lives in the repo for tests.

## License

MIT — see [LICENSE](LICENSE).
