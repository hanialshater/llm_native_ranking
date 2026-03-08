#!/usr/bin/env python3
"""CLI entry point: Scrape content and store in SQLite."""

import argparse
from ranking_pipeline.pipeline import run_scrape


def main():
    parser = argparse.ArgumentParser(description="Scrape content sources")
    parser.add_argument(
        "--source",
        choices=["philosophize_this", "ig_nobel", "ebert"],
        default="philosophize_this",
    )
    parser.add_argument("--n", type=int, default=10, help="Number of episodes/items")
    parser.add_argument("--db", default="ranking.db", help="Database path")
    args = parser.parse_args()

    run_scrape(args.source, args.n, args.db)


if __name__ == "__main__":
    main()
