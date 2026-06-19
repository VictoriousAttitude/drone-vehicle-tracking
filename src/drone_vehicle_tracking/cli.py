"""Command-line entrypoint (installed as ``dvt``)."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect, track and geo-reference vehicles from nadir drone video."
    )
    parser.add_argument("--video", required=True, help="Path to the drone video file.")
    parser.add_argument("--srt", required=True, help="Path to the matching DJI SRT file.")
    parser.add_argument("--config", default="configs/default.yaml", help="Pipeline config.")
    args = parser.parse_args()

    from drone_vehicle_tracking.pipeline import run

    tracks = run(args.video, args.srt, args.config)
    print(f"Done: {len(tracks)} geo-referenced vehicle tracks.")


if __name__ == "__main__":
    main()
