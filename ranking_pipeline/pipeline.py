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
    get_essays_for_episode,
    get_episodes,
    episode_exists,
    get_rankings_for_episode,
    delete_essays_for_episode,
)
from .scrapers.philosophize import scrape_all as scrape_philosophize
from .scrapers.ig_nobel import scrape_all as scrape_ig_nobel
from .scrapers.ebert import scrape_all as scrape_ebert
from .rewriter import rewrite_episode
from .llm import run_parallel
from .ranker import rank_all_dimensions, DIMENSIONS
from .rrf import rrf_scores, USE_CASES
from .bt import global_bt_all_dimensions


def run_scrape(source="philosophize_this", n_episodes=10, db_path="ranking.db",
               force=False):
    """Scrape content and store in DB. Skips existing episodes unless force=True."""
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

    inserted = 0
    skipped = 0
    for ep_data in raw:
        ext_id = ep_data.get("external_id", "") or ep_data.get("url", "")
        ep_data["external_id"] = ext_id
        if not force and ext_id and episode_exists(conn, source, ext_id):
            skipped += 1
            continue
        insert_episode(conn, source, ep_data)
        inserted += 1

    conn.close()
    print(f"Stored {inserted} new episodes ({skipped} already existed, skipped)")
    return inserted


def run_rewrite(source="philosophize_this", model=None, db_path="ranking.db",
                force=False):
    """Rewrite all episodes into mini-essays (parallel). Skips already-done unless force=True."""
    conn = get_connection(db_path)
    setup_db(conn)

    episodes = get_episodes(conn, source)
    total_episodes = len(episodes)
    skipped = 0

    # Phase 1: collect eligible episodes (sequential DB ops)
    to_rewrite = []
    for idx, ep in enumerate(episodes, 1):
        prefix = f"[{idx}/{total_episodes}]"

        existing = get_essays_for_episode(conn, ep["id"])
        if existing and not force:
            skipped += 1
            print(f"  {prefix} Episode {ep['id']} already has {len(existing)} essays, skipping")
            continue

        if existing and force:
            print(f"  {prefix} Episode {ep['id']} — force mode, deleting {len(existing)} existing essays")
            delete_essays_for_episode(conn, ep["id"])

        if not ep["raw_text"]:
            print(f"  {prefix} Episode {ep['id']} has no text, skipping")
            skipped += 1
            continue

        title = ep["title"][:60] if ep["title"] else "Untitled"
        print(f"  {prefix} Queued: {title}")
        to_rewrite.append(ep)

    # Phase 2: parallel rewrite LLM calls
    total = 0
    failed = 0
    if to_rewrite:
        print(f"\n  Rewriting {len(to_rewrite)} episodes in parallel...")
        args_list = [(ep["raw_text"], model) for ep in to_rewrite]
        results = run_parallel(rewrite_episode, args_list, label="rewrites")

        # Insert results (sequential DB writes)
        for ep, essays in zip(to_rewrite, results):
            if essays:
                for e in essays:
                    insert_essay(conn, ep["id"], e)
                conn.commit()
                print(f"    Episode {ep['id']}: {len(essays)} essays")
                total += len(essays)
            else:
                failed += 1
                print(f"    Episode {ep['id']}: failed")

    conn.close()
    print(f"\nRewrite complete: {total} new essays, {skipped} skipped, {failed} failed")
    return total


def run_rank(source="philosophize_this", model=None, db_path="ranking.db",
             run_global_bt=True, bt_window_size=12, bt_max_passes=3,
             force=False):
    """Run listwise ranking + RRF + optional global BT. Skips ranked episodes unless force=True."""
    conn = get_connection(db_path)
    setup_db(conn)
    session_id = str(uuid.uuid4())

    episodes = get_episodes(conn, source)
    all_essays = []
    total_episodes = len(episodes)
    ranked_count = 0
    skipped = 0

    # Within-episode ranking
    print("Within-episode listwise ranking...")
    for idx, ep in enumerate(episodes, 1):
        prefix = f"[{idx}/{total_episodes}]"
        ep_essays = get_essays_for_episode(conn, ep["id"])
        if len(ep_essays) < 3:
            print(f"  {prefix} Episode {ep['id']} has only {len(ep_essays)} essays (<3), skipping ranking")
            continue

        # Skip if already ranked (check for any existing rankings)
        if not force and get_rankings_for_episode(conn, ep["id"]):
            skipped += 1
            essay_objs = [{"id": e["id"], "text": e["text"]} for e in ep_essays]
            all_essays.extend(essay_objs)
            print(f"  {prefix} Episode {ep['id']} already ranked, skipping (essays still collected for BT)")
            continue

        title = ep["title"][:40] if ep["title"] else "Untitled"
        print(f"\n  {prefix} Episode {ep['id']}: {title} ({len(ep_essays)} essays)")
        essay_objs = [{"id": e["id"], "text": e["text"]} for e in ep_essays]
        all_essays.extend(essay_objs)

        try:
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

            conn.commit()
            ranked_count += 1
        except Exception as e:
            print(f"    Failed ranking episode {ep['id']}: {e}")

    print(f"\nWithin-episode ranking: {ranked_count} ranked, {skipped} skipped")

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
        conn.commit()
    else:
        print("\nSkipping global BT (not enough essays or disabled)")

    conn.close()
    print(f"\nRanking complete. Session: {session_id}")
    return session_id


def run_pipeline(source="philosophize_this", n_episodes=10, model=None,
                 db_path="ranking.db", run_global_bt=True, bt_max_passes=3,
                 bt_window_size=12, force=False):
    """Run the full pipeline end-to-end."""
    print(f"{'='*50}")
    print(f"STEP 1: SCRAPING")
    print(f"{'='*50}")
    run_scrape(source, n_episodes, db_path, force=force)

    print(f"\n{'='*50}")
    print(f"STEP 2: REWRITING")
    print(f"{'='*50}")
    run_rewrite(source, model, db_path, force=force)

    print(f"\n{'='*50}")
    print(f"STEP 3: RANKING")
    print(f"{'='*50}")
    session_id = run_rank(source, model, db_path, run_global_bt,
                          bt_window_size=bt_window_size,
                          bt_max_passes=bt_max_passes, force=force)

    return session_id
