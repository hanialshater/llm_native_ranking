"""Validation: compare ranking methods against human ground truth."""

from .ranker import DIMENSIONS


def spearman(rank_a, rank_b):
    """
    Spearman rank correlation between two rankings.

    Args:
        rank_a, rank_b: Dicts of {essay_id: rank} (1-based).

    Returns:
        Float in [-1, 1].
    """
    ids = list(rank_a.keys())
    n = len(ids)
    if n <= 1:
        return 1.0
    d2 = sum((rank_a[i] - rank_b[i]) ** 2 for i in ids)
    return 1 - (6 * d2) / (n * (n**2 - 1))


def rankings_to_ranks(ranking_list):
    """
    Convert ordered list to rank dict.

    Args:
        ranking_list: List of essay IDs, best first.

    Returns:
        Dict of {essay_id: rank} (1-based).
    """
    return {eid: r + 1 for r, eid in enumerate(ranking_list)}


def collect_human_ranking(essays, dimension):
    """
    Present essays to human in randomized order for ranking.

    Args:
        essays: List of dicts with 'id', 'title', 'text'.
        dimension: One of DIMENSIONS keys.

    Returns:
        List of essay IDs in ranked order (best first).
    """
    import random

    shuffled = essays.copy()
    random.shuffle(shuffled)

    print(f"\n{'='*60}")
    print(f"RANK BY: {DIMENSIONS[dimension]}")
    print(f"{'='*60}\n")
    for i, e in enumerate(shuffled):
        print(f"[{i+1}] {e.get('title', 'Untitled')}")
        print(f"    {e['text'][:200]}...")
        print()

    print("Enter essay numbers in order from best to worst")
    print("(e.g. '3 1 7 2 ...'):")
    order = list(map(int, input().split()))
    return [shuffled[i - 1]["id"] for i in order]


def validate_methods(human_rankings, llm_rankings):
    """
    Compare LLM listwise rankings against human ground truth.

    Args:
        human_rankings: Dict of {dimension: [essay_ids best-first]}.
        llm_rankings: Dict of {dimension: [essay_ids best-first]}.

    Returns:
        Dict of {dimension: spearman_correlation}.
    """
    results = {}
    for dim in DIMENSIONS:
        if dim in human_rankings and dim in llm_rankings:
            hr = rankings_to_ranks(human_rankings[dim])
            lr = rankings_to_ranks(llm_rankings[dim])
            results[dim] = spearman(hr, lr)
    return results


def compare_all_methods(human_rankings, methods):
    """
    Compare multiple methods against human ground truth.

    Args:
        human_rankings: Dict of {dimension: [essay_ids best-first]}.
        methods: Dict of {method_name: {dimension: [essay_ids best-first]}}.

    Returns:
        Dict of {method_name: {dimension: correlation}}.
    """
    results = {}
    for method_name, method_rankings in methods.items():
        results[method_name] = validate_methods(human_rankings, method_rankings)
    return results


def print_validation_report(results):
    """Pretty-print validation results."""
    print(f"\n{'='*70}")
    print("VALIDATION REPORT")
    print(f"{'='*70}")
    print(f"{'Method':<20} {'Aporia':>8} {'Compr.':>8} {'Defam':>8} {'Tension':>8} {'Force':>8} {'Mean':>8}")
    print("-" * 70)

    for method, dims in results.items():
        vals = [dims.get(d, 0) for d in ["aporia", "compression", "defam", "tension", "force"]]
        mean = sum(vals) / len(vals) if vals else 0
        print(
            f"{method:<20} {vals[0]:>8.3f} {vals[1]:>8.3f} {vals[2]:>8.3f} "
            f"{vals[3]:>8.3f} {vals[4]:>8.3f} {mean:>8.3f}"
        )
    print(f"{'='*70}")
