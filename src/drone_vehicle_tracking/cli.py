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
    parser.add_argument(
        "--overlay",
        help="Also write an annotated video (boxes + track IDs) to this path.",
    )
    parser.add_argument(
        "--cot",
        help="Also write Cursor-on-Target (CoT) XML events for TAK to this path.",
    )
    args = parser.parse_args()

    if args.benchmark:
        from drone_vehicle_tracking.perf import benchmark

        print(benchmark(args.video, args.srt, args.config).format_report())
        return

    from drone_vehicle_tracking.pipeline import run

    tracks = run(args.video, args.srt, args.config, cot_path=args.cot)
    print(f"Done: {len(tracks)} geo-referenced vehicle tracks.")
    if args.cot:
        print(f"Wrote CoT events: {args.cot}")

    if args.overlay:
        from drone_vehicle_tracking.visualization.video_overlay import render_overlay

        render_overlay(args.video, tracks, args.overlay)
        print(f"Wrote annotated video: {args.overlay}")


if __name__ == "__main__":  # pragma: no cover
    main()
