#!/usr/bin/env python3
"""CLI entry point: Score essays one-by-one with RAG context."""

import argparse
from ranking_pipeline.pipeline import run_rag_score


def main():
    parser = argparse.ArgumentParser(description="Score essays individually with RAG context")
    parser.add_argument(
        "--source",
        choices=["philosophize_this", "ig_nobel", "ebert"],
        default="philosophize_this",
    )
    parser.add_argument("--model", default=None, help="LLM model name")
    parser.add_argument("--db", default="ranking.db", help="Database path")
    parser.add_argument("--force", action="store_true",
                        help="Force re-score all essays, ignoring cached scores")
    args = parser.parse_args()

    run_rag_score(
        source=args.source,
        model=args.model,
        db_path=args.db,
        force=args.force,
    )


if __name__ == "__main__":
    main()
