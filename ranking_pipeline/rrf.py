"""Reciprocal Rank Fusion (RRF) scoring and use-case weighting.

Zero LLM calls — pure math on existing rankings.
"""

USE_CASES = {
    "default": {"enjoyment": 1.0, "utility": 1.0, "clarity": 1.0, "surprise": 1.0, "stickiness": 1.0},
    "casual": {"enjoyment": 2.5, "utility": 0.5, "clarity": 1.5, "surprise": 1.5, "stickiness": 1.0},
    "practical": {"enjoyment": 0.5, "utility": 2.5, "clarity": 2.0, "surprise": 0.5, "stickiness": 1.0},
    "shareable": {"enjoyment": 1.5, "utility": 1.0, "clarity": 1.5, "surprise": 2.0, "stickiness": 2.5},
    "deep_dive": {"enjoyment": 1.0, "utility": 1.5, "clarity": 0.5, "surprise": 2.0, "stickiness": 2.0},
}


def rrf_scores(rankings, use_case="default", k=60):
    """
    Compute RRF scores from per-dimension rankings.

    Args:
        rankings: Dict of {dimension: [essay_id, ...]} (best first).
        use_case: Key into USE_CASES for weight vector.
        k: RRF smoothing constant (standard = 60).

    Returns:
        Dict of {essay_id: score}.
    """
    weights = USE_CASES[use_case]
    scores = {}
    for dim, ranking in rankings.items():
        w = weights.get(dim, 1.0)
        for pos, essay_id in enumerate(ranking):
            if essay_id not in scores:
                scores[essay_id] = 0.0
            scores[essay_id] += w / (k + pos + 1)
    return scores


def rank_by_usecase(rankings, use_case="default"):
    """
    Rank essays by a specific use-case weighting.

    Args:
        rankings: Dict of {dimension: [essay_id, ...]} (best first).
        use_case: Key into USE_CASES.

    Returns:
        List of essay IDs sorted by RRF score (best first).
    """
    scores = rrf_scores(rankings, use_case)
    return sorted(scores.keys(), key=lambda i: -scores[i])


def scores_for_all_usecases(rankings, k=60):
    """
    Compute RRF scores for all use cases.

    Returns:
        Dict of {use_case: {essay_id: score}}.
    """
    return {uc: rrf_scores(rankings, uc, k) for uc in USE_CASES}
