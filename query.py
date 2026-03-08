#!/usr/bin/env python3
"""CLI entry point: Search ranked essays with natural language queries."""

import argparse
from ranking_pipeline.db import get_connection, setup_db
from ranking_pipeline.query import search, suggest_queries
from ranking_pipeline.rrf import USE_CASES


def format_results(weights, results):
    """Pretty-print query results."""
    print(f"\n{'='*60}")
    print("DIMENSION WEIGHTS (interpreted from your query)")
    print(f"{'='*60}")
    for dim, w in sorted(weights.items(), key=lambda x: -x[1]):
        bar = "#" * int(w * 10)
        print(f"  {dim:<12} {w:.1f}  {bar}")

    print(f"\n{'='*60}")
    print(f"TOP {len(results)} ESSAYS")
    print(f"{'='*60}")
    for i, r in enumerate(results, 1):
        title = r.get("title", "Untitled")
        source = r.get("source", "")
        ep_title = r.get("episode_title", "")
        score = r.get("score", 0)
        text = r.get("text", "")
        snippet = text[:200].replace("\n", " ")

        print(f"\n  #{i}  {title}  (score: {score:.3f})")
        if ep_title:
            print(f"       from: {ep_title} [{source}]")
        print(f"       {snippet}...")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Search ranked essays with natural language"
    )
    parser.add_argument("query", nargs="?", help="What kind of essay are you looking for?")
    parser.add_argument("--suggest", action="store_true", help="Show example queries")
    parser.add_argument("--use-case", choices=list(USE_CASES.keys()),
                        help="Use a preset use-case instead of NL query")
    parser.add_argument("--top", type=int, default=10, help="Number of results")
    parser.add_argument("--model", default=None, help="LLM model name")
    parser.add_argument("--db", default="ranking.db", help="Database path")
    args = parser.parse_args()

    if args.suggest:
        print("\nExample queries you can try:\n")
        for q in suggest_queries():
            print(f'  python query.py "{q}"')
        print()
        return

    if not args.query and not args.use_case:
        parser.print_help()
        print("\nTry: python query.py --suggest")
        return

    conn = get_connection(args.db)
    setup_db(conn)

    if args.use_case:
        weights = USE_CASES[args.use_case]
        from ranking_pipeline.query import get_top_essays
        results = get_top_essays(conn, weights, n=args.top)
    else:
        weights, results = search(conn, args.query, model=args.model, n=args.top)

    conn.close()

    if not results:
        print("\nNo ranked essays found. Run the pipeline first:")
        print("  python run_all.py --source philosophize_this")
        return

    format_results(weights, results)


if __name__ == "__main__":
    main()
