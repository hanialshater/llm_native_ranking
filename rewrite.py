#!/usr/bin/env python3
"""CLI entry point: Rewrite scraped content into mini-essays."""

import argparse
from ranking_pipeline.pipeline import run_rewrite


def main():
    parser = argparse.ArgumentParser(description="Rewrite episodes into mini-essays")
    parser.add_argument(
        "--source",
        choices=["philosophize_this", "ig_nobel", "ebert"],
        default="philosophize_this",
    )
    parser.add_argument("--model", default=None, help="LLM model name")
    parser.add_argument("--db", default="ranking.db", help="Database path")
    parser.add_argument("--force", action="store_true",
                        help="Force re-rewrite all episodes, replacing existing essays")
    args = parser.parse_args()

    run_rewrite(args.source, args.model, args.db, force=args.force)


if __name__ == "__main__":
    main()
