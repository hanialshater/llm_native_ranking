"""Full pipeline runner: scrape → rewrite → rank → score → fuse."""

import uuid

from .db import (
    get_connection,
    setup_db,
    insert_episode,
    insert_essay,
    insert_ranking,
    insert_rrf_score,
    insert_bt_score,
    get_all_essays,
    get_essays_for_episode,
    get_episodes,
)
from .scrapers.philosophize import scrape_all as scrape_philosophize
from .scrapers.ig_nobel import scrape_all as scrape_ig_nobel
from .scrapers.ebert import scrape_all as scrape_ebert
from .rewriter import rewrite_episode
from .ranker import rank_all_dimensions, DIMENSIONS
from .rrf import rrf_scores, USE_CASES
from .bt import global_bt_all_dimensions


def run_scrape(source="philosophize_this", n_episodes=10, db_path="ranking.db"):
    """Scrape content and store in DB."""
    conn = get_connection(db_path)
    setup_db(conn)

    print(f"Scraping {source} ({n_episodes} items)...")
    if source == "philosophize_this":
        raw = scrape_philosophize(n_episodes=n_episodes)
    elif source == "ig_nobel":
        raw = scrape_ig_nobel(n_papers=n_episodes)
    elif source == "ebert":
        raw = scrape_ebert(n_reviews=n_episodes)
    else:
        raise ValueError(f"Unknown source: {source}")

    for ep_data in raw:
        insert_episode(conn, source, ep_data)

    conn.close()
    print(f"Stored {len(raw)} episodes")
    return len(raw)


def run_rewrite(source="philosophize_this", model=None, db_path="ranking.db"):
    """Rewrite all episodes into mini-essays."""
    conn = get_connection(db_path)
    setup_db(conn)

    episodes = get_episodes(conn, source)
    total = 0

    for ep in episodes:
        # Skip if already has essays
        existing = get_essays_for_episode(conn, ep["id"])
        if existing:
            print(f"  Episode {ep['id']} already has {len(existing)} essays, skipping")
            continue
        if not ep["raw_text"]:
            print(f"  Episode {ep['id']} has no text, skipping")
            continue

        print(f"\n  Rewriting episode {ep['id']}: {ep['title'][:60]}")
        try:
            essays = rewrite_episode(ep["raw_text"], model=model)
            for e in essays:
                insert_essay(conn, ep["id"], e)
            print(f"    → {len(essays)} essays")
            total += len(essays)
        except Exception as e:
            print(f"    Failed: {e}")

    conn.close()
    print(f"\nTotal new essays: {total}")
    return total


def run_rank(source="philosophize_this", model=None, db_path="ranking.db",
             run_global_bt=True, bt_window_size=12, bt_max_passes=3):
    """Run listwise ranking + RRF + optional global BT."""
    conn = get_connection(db_path)
    setup_db(conn)
    session_id = str(uuid.uuid4())

    episodes = get_episodes(conn, source)
    all_essays = []

    # Within-episode ranking
    print("Within-episode listwise ranking...")
    for ep in episodes:
        ep_essays = get_essays_for_episode(conn, ep["id"])
        if len(ep_essays) < 3:
            continue

        print(f"\n  Episode {ep['id']}: {ep['title'][:40]} ({len(ep_essays)} essays)")
        essay_objs = [{"id": e["id"], "text": e["text"]} for e in ep_essays]
        all_essays.extend(essay_objs)
        rankings = rank_all_dimensions(essay_objs, model=model)

        # Store rankings
        for dim, ranked_ids in rankings.items():
            for rank_pos, essay_id in enumerate(ranked_ids):
                insert_ranking(conn, essay_id, session_id, dim, rank_pos + 1, "listwise_llm", model or "default")

        # RRF scores
        for use_case in USE_CASES:
            scores = rrf_scores(rankings, use_case)
            sorted_ids = sorted(scores.keys(), key=lambda i: -scores[i])
            for rank_pos, essay_id in enumerate(sorted_ids):
                insert_rrf_score(conn, essay_id, session_id, use_case, scores[essay_id], rank_pos + 1)

    # Global BT
    if run_global_bt and len(all_essays) >= 6:
        print(f"\nGlobal BT ranking ({len(all_essays)} essays)...")
        bt_results = global_bt_all_dimensions(
            all_essays,
            window_size=bt_window_size,
            max_passes=bt_max_passes,
            model=model,
        )
        for dim, bt_scores in bt_results.items():
            for essay_id, score in bt_scores.items():
                insert_bt_score(conn, essay_id, dim, score, bt_max_passes, session_id)
    else:
        print("\nSkipping global BT (not enough essays or disabled)")

    conn.close()
    print(f"\nRanking complete. Session: {session_id}")
    return session_id


def run_pipeline(source="philosophize_this", n_episodes=10, model=None,
                 db_path="ranking.db", run_global_bt=True, bt_max_passes=3):
    """Run the full pipeline end-to-end."""
    print(f"{'='*50}")
    print(f"STEP 1: SCRAPING")
    print(f"{'='*50}")
    run_scrape(source, n_episodes, db_path)

    print(f"\n{'='*50}")
    print(f"STEP 2: REWRITING")
    print(f"{'='*50}")
    run_rewrite(source, model, db_path)

    print(f"\n{'='*50}")
    print(f"STEP 3: RANKING")
    print(f"{'='*50}")
    session_id = run_rank(source, model, db_path, run_global_bt, bt_max_passes=bt_max_passes)

    return session_id
