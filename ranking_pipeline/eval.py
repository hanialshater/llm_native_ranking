"""Evaluation framework: gold-standard LLM ranking vs retrieval methods.

For a given set of essays, asks the LLM to rank them directly for a query
(gold standard). Then compares how each retrieval method's top-K overlaps
with the gold top-K, using Precision@K.

Key design: gold standard and all methods operate on the SAME essay subset
to ensure fair comparison.
"""

import json
from .llm import chat
from .db import get_all_essays, get_essay_details, get_bt_scores, get_rag_scores
from .query import interpret_query, score_essays, get_top_essays_text_match


GOLD_RANK_PROMPT = """You are an expert essay curator. A reader wants: "{query}"

Below are {n} essays. Rank them from BEST to WORST for this reader's request.
Consider how well each essay matches what the reader is looking for.

{essays}

**Instructions:**
- Rank ALL essays from best match to worst match
- No ties — force strict ordering
- Return ONLY a JSON array of essay IDs in ranked order (best first):
  [{example_ids}]

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
    example_ids = ", ".join(str(e["id"]) for e in essays[:3]) + ", ..."
    prompt = GOLD_RANK_PROMPT.format(
        query=query, n=len(essays), essays=formatted, example_ids=example_ids,
    )
    text = chat(prompt, model=model, max_tokens=2000)
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
        if missing or extra:
            print(f"    Gold warning: missing={len(missing)}, extra={len(extra)}, corrected")

    return ranked_ids


def precision_at_k(retrieved, gold, k):
    """Fraction of top-K retrieved that appear in top-K gold."""
    if k <= 0 or not retrieved or not gold:
        return 0.0
    gold_top_k = set(gold[:k])
    retrieved_top_k = retrieved[:k]
    hits = sum(1 for rid in retrieved_top_k if rid in gold_top_k)
    return hits / k


def _rank_subset_by_bt(conn, essay_ids, weights):
    """Rank a specific subset of essays using BT scores + weights."""
    bt = get_bt_scores(conn)
    if not bt:
        return []
    # Filter to only our subset
    bt_subset = {eid: scores for eid, scores in bt.items() if eid in essay_ids}
    if not bt_subset:
        return []
    composite = score_essays(bt_subset, weights)
    return sorted(composite, key=lambda i: -composite[i])


def _rank_subset_by_rag(conn, essay_ids, weights):
    """Rank a specific subset of essays using RAG scores + weights."""
    rag = get_rag_scores(conn)
    if not rag:
        return []
    rag_subset = {eid: scores for eid, scores in rag.items() if eid in essay_ids}
    if not rag_subset:
        return []
    composite = score_essays(rag_subset, weights)
    return sorted(composite, key=lambda i: -composite[i])


def _rank_subset_by_text(conn, essay_ids, query, essays_with_text):
    """Rank a specific subset of essays using text match."""
    query_words = set(query.lower().split())
    scored = {}
    for e in essays_with_text:
        if e["id"] not in essay_ids:
            continue
        text_lower = (e.get("text", "") or "").lower()
        title_lower = (e.get("title", "") or "").lower()
        combined = text_lower + " " + title_lower
        matches = sum(1 for w in query_words if w in combined)
        scored[e["id"]] = matches / max(len(query_words), 1)
    return sorted(scored, key=lambda i: -scored[i])


def evaluate_query(conn, query, model=None, max_essays=50, ks=(1, 3, 5, 10),
                   n_gold_runs=3):
    """
    Evaluate all retrieval methods against gold-standard LLM ranking.

    All methods are evaluated on the SAME essay subset for fair comparison.
    Gold standard is averaged over multiple runs to reduce LLM variance.

    Args:
        conn: DB connection.
        query: Natural language query.
        model: Model name.
        max_essays: Max essays to include (keeps gold ranking feasible).
        ks: Tuple of K values for precision@K.
        n_gold_runs: Number of gold ranking runs to average over.

    Returns:
        Dict with gold ranking and per-method precision@K scores.
    """
    # Get essay subset — same for gold AND all methods
    all_essays = get_all_essays(conn)
    if len(all_essays) > max_essays:
        all_essays = all_essays[:max_essays]

    if not all_essays:
        print("No essays found.")
        return {}

    essay_ids = [e["id"] for e in all_essays]
    essay_id_set = set(essay_ids)
    details = get_essay_details(conn, essay_ids)
    detail_map = {d["id"]: d for d in details}

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
    print(f"\nEvaluating query: \"{query}\" ({n} essays)")

    # Gold standard — run multiple times and aggregate via average rank
    print(f"  Computing gold-standard LLM ranking ({n_gold_runs} runs)...")
    avg_rank = {e["id"]: 0.0 for e in essays_for_gold}
    gold_runs = []
    for run in range(n_gold_runs):
        gold = gold_standard_ranking(essays_for_gold, query, model=model)
        gold_runs.append(gold)
        for rank_pos, eid in enumerate(gold):
            avg_rank[eid] += rank_pos
        print(f"    Gold run {run+1}/{n_gold_runs}: top-3 = {gold[:3]}")

    # Final gold = sorted by average rank position
    for eid in avg_rank:
        avg_rank[eid] /= n_gold_runs
    gold_final = sorted(avg_rank, key=lambda i: avg_rank[i])
    print(f"  Gold consensus top-5: {gold_final[:5]}")

    # Interpret query weights (shared by BT and RAG)
    weights = interpret_query(query, model=model)

    # Rank using each method — restricted to same essay subset
    methods_results = {}

    # BT
    bt_ranked = _rank_subset_by_bt(conn, essay_id_set, weights)
    methods_results["bt"] = bt_ranked

    # RAG
    rag_ranked = _rank_subset_by_rag(conn, essay_id_set, weights)
    methods_results["rag"] = rag_ranked

    # Text match
    text_ranked = _rank_subset_by_text(conn, essay_id_set, query, essays_for_gold)
    methods_results["text_match"] = text_ranked

    # Compute precision@K for each method
    results = {
        "query": query,
        "n_essays": n,
        "gold_top10": gold_final[:10],
        "weights": weights,
        "methods": {},
    }

    for method, retrieved_ids in methods_results.items():
        precisions = {}
        for k in ks:
            p = precision_at_k(retrieved_ids, gold_final, k)
            precisions[f"P@{k}"] = p

        results["methods"][method] = {
            "retrieved_top10": retrieved_ids[:10],
            "precisions": precisions,
        }
        prec_str = ", ".join(f"P@{k}={precisions[f'P@{k}']:.2f}" for k in ks)
        print(f"  {method:12s}: {prec_str}")

    return results


def evaluate_queries(conn, queries, model=None, max_essays=50, ks=(1, 3, 5, 10),
                     n_gold_runs=3):
    """
    Evaluate multiple queries and compute average precision@K per method.

    Args:
        conn: DB connection.
        queries: List of query strings.
        model: Model name.
        max_essays: Max essays for gold ranking.
        ks: K values for precision@K.
        n_gold_runs: Gold ranking runs per query for stability.

    Returns:
        Dict with per-query results and averaged metrics.
    """
    all_results = []
    for query in queries:
        result = evaluate_query(
            conn, query, model=model, max_essays=max_essays,
            ks=ks, n_gold_runs=n_gold_runs,
        )
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
