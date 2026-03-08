"""Natural language query interface for ranked essays."""

import json
from .llm import chat
from .ranker import DIMENSIONS
from .db import get_bt_scores, get_rrf_scores, get_essay_details


EXAMPLE_QUERIES = [
    "Something fun and mind-bending",
    "Short practical advice I can use today",
    "Surprising ideas that will stick with me",
    "Clear explanations of complex topics",
    "Essays I'd share with a smart friend",
    "Fun to read but also genuinely useful",
    "The most memorable and quotable pieces",
]

INTERPRET_PROMPT = """You are a search assistant for an essay collection.
The user describes what they want to read. Map their request to weights (0.0 to 3.0) across these dimensions:

{dimensions}

User request: "{query}"

Return ONLY a JSON object with dimension weights, e.g.:
{{"enjoyment": 2.0, "utility": 0.5, "clarity": 1.0, "surprise": 1.5, "stickiness": 2.0}}

Higher weight = more important for this request. Use 1.0 as neutral.
Return only the JSON object, nothing else."""


def interpret_query(query, model=None):
    """
    Use LLM to map a natural language query to dimension weights.

    Args:
        query: Natural language description of what the reader wants.
        model: Model name.

    Returns:
        Dict of {dimension: weight} with values 0.0-3.0.
    """
    dim_descriptions = "\n".join(
        f"- {key}: {desc.strip().splitlines()[0]}"
        for key, desc in DIMENSIONS.items()
    )
    prompt = INTERPRET_PROMPT.format(dimensions=dim_descriptions, query=query)
    text = chat(prompt, model=model, max_tokens=200)

    # Strip code fences if present
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        end = -1 if lines[-1].strip().startswith("```") else len(lines)
        text = "\n".join(lines[1:end]).strip()

    weights = json.loads(text)

    # Validate and clamp
    for dim in DIMENSIONS:
        if dim not in weights:
            weights[dim] = 1.0
        weights[dim] = max(0.0, min(3.0, float(weights[dim])))

    return weights


def score_essays(bt_scores, weights):
    """
    Compute weighted composite score for each essay from BT scores.

    Args:
        bt_scores: Dict of {essay_id: {dimension: bt_score}}.
        weights: Dict of {dimension: weight}.

    Returns:
        Dict of {essay_id: composite_score}.
    """
    composite = {}
    for eid, dim_scores in bt_scores.items():
        total = 0.0
        for dim, w in weights.items():
            total += w * dim_scores.get(dim, 0.0)
        composite[eid] = total
    return composite


def get_top_essays(conn, weights, n=10):
    """
    Query DB and return top N essays ranked by weighted BT scores.

    Falls back to RRF scores if no BT scores exist.

    Args:
        conn: DB connection.
        weights: Dict of {dimension: weight}.
        n: Number of results.

    Returns:
        List of dicts with essay details and score, sorted best first.
    """
    bt = get_bt_scores(conn)

    if bt:
        composite = score_essays(bt, weights)
    else:
        # Fallback: use default RRF scores
        composite = get_rrf_scores(conn, "default")

    if not composite:
        return []

    top_ids = sorted(composite, key=lambda i: -composite[i])[:n]
    details = get_essay_details(conn, top_ids)
    detail_map = {d["id"]: d for d in details}

    results = []
    for eid in top_ids:
        if eid in detail_map:
            entry = detail_map[eid]
            entry["score"] = composite[eid]
            results.append(entry)
    return results


def suggest_queries():
    """Return example queries to help users get started."""
    return EXAMPLE_QUERIES


def search(conn, query, model=None, n=10):
    """
    End-to-end: interpret a natural language query and return top essays.

    Args:
        conn: DB connection.
        query: Natural language description.
        model: Model name for query interpretation.
        n: Number of results.

    Returns:
        Tuple of (weights_dict, results_list).
    """
    weights = interpret_query(query, model=model)
    results = get_top_essays(conn, weights, n=n)
    return weights, results
