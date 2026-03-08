#!/usr/bin/env python3
"""CLI entry point: Evaluate retrieval methods against gold-standard LLM ranking."""

import argparse
import json
from ranking_pipeline.db import get_connection, setup_db
from ranking_pipeline.eval import evaluate_queries
from ranking_pipeline.query import EXAMPLE_QUERIES


DEFAULT_EVAL_QUERIES = [
    "Something fun and surprising",
    "Practical advice I can use today",
    "Clear explanations of complex topics",
    "Memorable ideas that stick with you",
]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate BT vs RAG vs text-match retrieval against gold-standard LLM ranking"
    )
    parser.add_argument("--queries", nargs="+", default=None,
                        help="Custom queries to evaluate (default: built-in set)")
    parser.add_argument("--max-essays", type=int, default=50,
                        help="Max essays for gold ranking (keep small for cost)")
    parser.add_argument("--model", default=None, help="LLM model name")
    parser.add_argument("--db", default="ranking.db", help="Database path")
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    args = parser.parse_args()

    conn = get_connection(args.db)
    setup_db(conn)

    queries = args.queries or DEFAULT_EVAL_QUERIES

    print(f"Evaluating {len(queries)} queries against gold-standard ranking")
    print(f"Max essays per query: {args.max_essays}")
    print(f"Methods: bt, rag, text_match")

    results = evaluate_queries(
        conn, queries,
        model=args.model,
        max_essays=args.max_essays,
    )

    conn.close()

    if args.output and results:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
