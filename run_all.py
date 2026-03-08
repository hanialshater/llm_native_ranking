#!/usr/bin/env python3
"""CLI entry point: Run the full pipeline (scrape → rewrite → rank)."""

import argparse
from ranking_pipeline.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Full LLM-native ranking pipeline")
    parser.add_argument(
        "--source",
        choices=["philosophize_this", "ig_nobel", "ebert"],
        default="philosophize_this",
    )
    parser.add_argument("--n", type=int, default=10, help="Number of episodes/items")
    parser.add_argument("--model", default=None, help="LLM model name")
    parser.add_argument("--db", default="ranking.db", help="Database path")
    parser.add_argument("--no-bt", action="store_true", help="Skip global BT ranking")
    parser.add_argument("--bt-passes", type=int, default=3, help="BT max passes")
    parser.add_argument("--bt-window", type=int, default=12, help="BT window size")
    parser.add_argument("--force", action="store_true",
                        help="Force re-process all steps, ignoring cached results")
    args = parser.parse_args()

    run_pipeline(
        source=args.source,
        n_episodes=args.n,
        model=args.model,
        db_path=args.db,
        run_global_bt=not args.no_bt,
        bt_max_passes=args.bt_passes,
        bt_window_size=args.bt_window,
        force=args.force,
    )


if __name__ == "__main__":
    main()
