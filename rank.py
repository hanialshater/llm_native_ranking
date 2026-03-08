#!/usr/bin/env python3
"""CLI entry point: Run listwise ranking + RRF + global BT."""

import argparse
from ranking_pipeline.pipeline import run_rank


def main():
    parser = argparse.ArgumentParser(description="Rank essays via listwise LLM comparison")
    parser.add_argument(
        "--source",
        choices=["philosophize_this", "ig_nobel", "ebert"],
        default="philosophize_this",
    )
    parser.add_argument("--model", default=None, help="LLM model name")
    parser.add_argument("--db", default="ranking.db", help="Database path")
    parser.add_argument("--no-bt", action="store_true", help="Skip global BT ranking")
    parser.add_argument("--bt-passes", type=int, default=3, help="BT max passes")
    parser.add_argument("--bt-window", type=int, default=12, help="BT window size")
    parser.add_argument("--force", action="store_true",
                        help="Force re-rank all episodes, ignoring cached rankings")
    args = parser.parse_args()

    run_rank(
        source=args.source,
        model=args.model,
        db_path=args.db,
        run_global_bt=not args.no_bt,
        bt_window_size=args.bt_window,
        bt_max_passes=args.bt_passes,
        force=args.force,
    )


if __name__ == "__main__":
    main()
