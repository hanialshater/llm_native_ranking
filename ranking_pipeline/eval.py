"""Evaluation framework: gold-standard LLM ranking vs retrieval methods.

For a small set of essays, concatenates all texts and asks the LLM to rank
them directly for a given query. Then compares BT, RAG, and text-match
retrieval methods against this gold standard at @1, @3, @5, @10.
"""

import json
from .llm import chat
from .db import get_connection, get_all_essays, get_essay_details
from .query import search


GOLD_RANK_PROMPT = """You are an expert essay curator. A reader wants: "{query}"

Below are {n} essays. Rank them from BEST to WORST for this reader's request.
Consider how well each essay matches what the reader is looking for.

{essays}

**Instructions:**
- Rank ALL essays from best match to worst match
- No ties — force strict ordering
- Return ONLY a JSON array of essay IDs in ranked order (best first):
  [42, 17, 3, ...]

Return only the JSON array, nothing else."""


def _strip_code_fences(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        end = -1 if lines[-1].strip().startswith("```") else len(lines)
        text = "\n".join(lines[1:end])
    return text.strip()


def gold_standard_ranking(essays, query, model=None):
    """
    Get gold-standard ranking by having LLM rank all essays at once for a query.

    Args:
        essays: List of dicts with 'id', 'title', 'text'.
        query: Natural language query.
        model: Model name.

    Returns:
        List of essay IDs in ranked order (best first).
    """
    formatted = "\n\n".join(
        f"**Essay {e['id']}** — {e.get('title', 'Untitled')}:\n{e['text']}"
        for e in essays
    )
    prompt = GOLD_RANK_PROMPT.format(query=query, n=len(essays), essays=formatted)
    text = chat(prompt, model=model, max_tokens=1000)
    text = _strip_code_fences(text)
    ranked_ids = json.loads(text)

    valid_ids = {e["id"] for e in essays}
    returned = set(ranked_ids)
    if returned != valid_ids:
        missing = valid_ids - returned
        extra = returned - valid_ids
        if extra:
            ranked_ids = [i for i in ranked_ids if i in valid_ids]
        if missing:
            ranked_ids.extend(missing)

    return ranked_ids


def precision_at_k(retrieved, gold, k):
    """
    Precision@K: fraction of top-K retrieved that appear in top-K gold.

    Args:
        retrieved: List of IDs in retrieval order.
        gold: List of IDs in gold-standard order.
        k: Cutoff.

    Returns:
        Float in [0, 1].
    """
    if k <= 0 or not retrieved or not gold:
        return 0.0
    gold_top_k = set(gold[:k])
    retrieved_top_k = retrieved[:k]
    hits = sum(1 for rid in retrieved_top_k if rid in gold_top_k)
    return hits / k


def evaluate_query(conn, query, model=None, max_essays=50, ks=(1, 3, 5, 10)):
    """
    Evaluate all retrieval methods against gold-standard LLM ranking for one query.

    Args:
        conn: DB connection.
        query: Natural language query.
        model: Model name.
        max_essays: Max essays to include in gold ranking (keep small for cost).
        ks: Tuple of K values for precision@K.

    Returns:
        Dict with gold ranking and per-method precision@K scores.
    """
    # Get all essays (limit for gold ranking feasibility)
    all_essays = get_all_essays(conn)
    if len(all_essays) > max_essays:
        all_essays = all_essays[:max_essays]

    if not all_essays:
        print("No essays found.")
        return {}

    essay_ids = [e["id"] for e in all_essays]
    details = get_essay_details(conn, essay_ids)
    detail_map = {d["id"]: d for d in details}

    # Build essay list with text for gold ranking
    essays_for_gold = []
    for e in all_essays:
        d = detail_map.get(e["id"])
        if d:
            essays_for_gold.append({
                "id": e["id"],
                "title": d.get("title", ""),
                "text": d.get("text", e.get("text", "")),
            })

    n = len(essays_for_gold)
    max_k = max(ks)
    print(f"\nEvaluating query: \"{query}\" ({n} essays)")

    # Gold standard
    print("  Computing gold-standard LLM ranking...")
    gold = gold_standard_ranking(essays_for_gold, query, model=model)

    # Run each retrieval method
    methods = ["bt", "rag", "text_match"]
    results = {"query": query, "n_essays": n, "gold_top10": gold[:10], "methods": {}}

    for method in methods:
        print(f"  Running {method} retrieval...")
        try:
            _, retrieved = search(conn, query, model=model, n=max_k, method=method)
            retrieved_ids = [r["id"] for r in retrieved]
        except Exception as e:
            print(f"    {method} failed: {e}")
            retrieved_ids = []

        precisions = {}
        for k in ks:
            p = precision_at_k(retrieved_ids, gold, k)
            precisions[f"P@{k}"] = p

        results["methods"][method] = {
            "retrieved_top10": retrieved_ids[:10],
            "precisions": precisions,
        }
        prec_str = ", ".join(f"P@{k}={precisions[f'P@{k}']:.2f}" for k in ks)
        print(f"    {method}: {prec_str}")

    return results


def evaluate_queries(conn, queries, model=None, max_essays=50, ks=(1, 3, 5, 10)):
    """
    Evaluate multiple queries and compute average precision@K per method.

    Args:
        conn: DB connection.
        queries: List of query strings.
        model: Model name.
        max_essays: Max essays for gold ranking.
        ks: K values for precision@K.

    Returns:
        Dict with per-query results and averaged metrics.
    """
    all_results = []
    for query in queries:
        result = evaluate_query(conn, query, model=model, max_essays=max_essays, ks=ks)
        if result:
            all_results.append(result)

    if not all_results:
        return {}

    # Average across queries
    methods = ["bt", "rag", "text_match"]
    averages = {}

    print(f"\n{'='*60}")
    print("AVERAGE RESULTS")
    print(f"{'='*60}")

    for method in methods:
        avg = {}
        for k in ks:
            key = f"P@{k}"
            values = [
                r["methods"][method]["precisions"][key]
                for r in all_results
                if method in r["methods"]
            ]
            avg[key] = sum(values) / len(values) if values else 0.0
        averages[method] = avg
        prec_str = ", ".join(f"P@{k}={avg[f'P@{k}']:.2f}" for k in ks)
        print(f"  {method:12s}: {prec_str}")

    return {
        "queries": all_results,
        "averages": averages,
    }
