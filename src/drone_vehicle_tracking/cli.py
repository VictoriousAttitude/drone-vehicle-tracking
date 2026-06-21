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
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run the pipeline and print a per-stage timing report instead of a track count.",
    )
    args = parser.parse_args()

    if args.benchmark:
        from drone_vehicle_tracking.perf import benchmark

        print(benchmark(args.video, args.srt, args.config).format_report())
        return

    from drone_vehicle_tracking.pipeline import run

    tracks = run(args.video, args.srt, args.config)
    print(f"Done: {len(tracks)} geo-referenced vehicle tracks.")


if __name__ == "__main__":  # pragma: no cover
    main()
