"""Reciprocal Rank Fusion (RRF) scoring and use-case weighting.

Zero LLM calls — pure math on existing rankings.
"""

USE_CASES = {
    "default": {"aporia": 1.0, "compression": 1.0, "defam": 1.0, "tension": 1.0, "force": 1.0},
    "substack": {"aporia": 1.5, "compression": 2.0, "defam": 1.5, "tension": 1.0, "force": 0.5},
    "beginner": {"aporia": 0.5, "compression": 1.5, "defam": 2.0, "tension": 1.0, "force": 2.0},
    "shareable": {"aporia": 0.5, "compression": 2.5, "defam": 1.5, "tension": 0.5, "force": 2.0},
    "provocative": {"aporia": 2.0, "compression": 1.0, "defam": 2.0, "tension": 2.0, "force": 0.5},
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
