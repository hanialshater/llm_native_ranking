"""Global Bradley-Terry ranking via choix.

Multi-pass shuffle → listwise LLM windows → pairwise observations → BT scores.
"""

import random

import choix
import numpy as np

from .ranker import rank_essays, DIMENSIONS


def listwise_to_pairwise(ranking):
    """
    Convert a ranked list [best, ..., worst] to adjacent pairwise observations.

    Only adjacent pairs — not all transitive pairs — to avoid winner-takes-all collapse.

    Args:
        ranking: List of IDs in ranked order (best first).

    Returns:
        List of (winner_id, loser_id) tuples.
    """
    pairs = []
    for i in range(len(ranking) - 1):
        pairs.append((ranking[i], ranking[i + 1]))
    return pairs


def run_ranking_pass(all_essays, dimension, window_size=12, model=None):
    """
    One full shuffle pass over all essays.

    Shuffles essays, partitions into windows, runs LLM listwise ranking per window,
    returns pairwise observations.

    Args:
        all_essays: List of {'id': int, 'text': str}.
        dimension: One of DIMENSIONS keys.
        window_size: Essays per LLM ranking call.
        model: Model name.

    Returns:
        List of (winner_id, loser_id) pairwise observations.
    """
    essay_ids = [e["id"] for e in all_essays]
    essay_map = {e["id"]: e for e in all_essays}

    shuffled = essay_ids.copy()
    random.shuffle(shuffled)

    all_pairs = []
    windows = [shuffled[i : i + window_size] for i in range(0, len(shuffled), window_size)]

    for win_idx, window_ids in enumerate(windows):
        if len(window_ids) < 3:
            continue
        window_essays = [essay_map[i] for i in window_ids]

        ranked_ids = rank_essays(window_essays, dimension, model)

        pairs = listwise_to_pairwise(ranked_ids)
        all_pairs.extend(pairs)

    return all_pairs


def compute_bt_scores(all_pairs, essay_ids):
    """
    Run choix BT (Luce Spectral Ranking) on accumulated pairwise observations.

    Args:
        all_pairs: List of (winner_id, loser_id) tuples.
        essay_ids: List of all essay IDs.

    Returns:
        Dict of {essay_id: bt_score}, higher = better.
    """
    id_to_idx = {eid: i for i, eid in enumerate(essay_ids)}
    idx_to_id = {i: eid for eid, i in id_to_idx.items()}
    n = len(essay_ids)

    choix_pairs = [
        (id_to_idx[w], id_to_idx[l])
        for w, l in all_pairs
        if w in id_to_idx and l in id_to_idx
    ]

    if not choix_pairs:
        return {eid: 0.0 for eid in essay_ids}

    params = choix.lsr_pairwise(n, choix_pairs, alpha=0.01)
    return {idx_to_id[i]: float(params[i]) for i in range(n)}


def check_convergence(scores_prev, scores_curr):
    """
    Spearman rank correlation between consecutive passes.

    Returns:
        Float in [-1, 1]. Values >= 0.98 indicate convergence.
    """
    ids = list(scores_curr.keys())
    prev_ranks = {
        eid: r
        for r, eid in enumerate(sorted(ids, key=lambda i: -scores_prev.get(i, 0)))
    }
    curr_ranks = {
        eid: r
        for r, eid in enumerate(sorted(ids, key=lambda i: -scores_curr[i]))
    }
    n = len(ids)
    if n < 2:
        return 1.0
    d2 = sum((prev_ranks[i] - curr_ranks[i]) ** 2 for i in ids)
    rho = 1 - (6 * d2) / (n * (n**2 - 1))
    return rho


def global_bt(
    all_essays,
    dimension,
    window_size=12,
    max_passes=6,
    convergence_tol=0.98,
    model=None,
):
    """
    Multi-pass global BT ranking for one dimension.

    Each pass: reshuffle → partition into windows → LLM listwise rank →
    accumulate pairwise observations → run choix BT → check convergence.

    Args:
        all_essays: List of {'id': int, 'text': str}.
        dimension: One of DIMENSIONS keys.
        window_size: Essays per LLM call (10-15 recommended).
        max_passes: Hard cap on passes.
        convergence_tol: Stop when rank correlation >= this.
        model: Model name.

    Returns:
        Dict of {essay_id: bt_score}.
    """
    essay_ids = [e["id"] for e in all_essays]
    all_pairs = []
    scores_prev = None
    scores_curr = None

    for pass_num in range(1, max_passes + 1):
        print(f"\n  Pass {pass_num}/{max_passes} — {len(all_essays)} essays, window={window_size}")
        new_pairs = run_ranking_pass(all_essays, dimension, window_size, model)
        all_pairs.extend(new_pairs)
        print(f"    Total pairwise observations: {len(all_pairs)}")

        scores_curr = compute_bt_scores(all_pairs, essay_ids)

        if scores_prev is not None:
            rho = check_convergence(scores_prev, scores_curr)
            print(f"    Rank correlation vs previous pass: {rho:.4f}")
            if rho >= convergence_tol:
                print(f"    Converged at pass {pass_num}")
                break
        else:
            print("    Pass 1 complete — need at least 2 passes to check convergence")

        scores_prev = scores_curr

    return scores_curr


def global_bt_all_dimensions(
    all_essays,
    window_size=12,
    max_passes=6,
    convergence_tol=0.98,
    model=None,
):
    """
    Run global BT for all 5 dimensions.

    Returns:
        Dict of {dimension: {essay_id: bt_score}}.
    """
    results = {}
    for dim in DIMENSIONS:
        print(f"\n{'='*50}\nDimension: {dim.upper()}\n{'='*50}")
        results[dim] = global_bt(
            all_essays, dim, window_size, max_passes, convergence_tol, model
        )
    return results
