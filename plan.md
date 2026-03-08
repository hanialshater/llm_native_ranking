# Plan: Reader-Centric Dimensions + Natural Language Query Interface

## Overview
Two connected changes:
1. Replace academic dimensions with reader-centric ones
2. Add a natural language query interface that lets users describe what they want and get ranked essays back

## Part 1: New Dimensions

Replace the 5 dimensions in `ranker.py` with reader-centric ones:

| Key | Name | Description |
|-----|------|-------------|
| `enjoyment` | Enjoyment | Fun to read — voice, rhythm, wit, narrative pull |
| `utility` | Practical Utility | Actionable takeaways the reader can apply |
| `clarity` | Clarity | Easy to follow, clean structure, no re-reading needed |
| `surprise` | Surprise | Teaches something unexpected, reframes the familiar |
| `stickiness` | Stickiness | Stays with you — you'd share it or quote it later |

### Files to update:
- **`ranking_pipeline/ranker.py`** — Replace DIMENSIONS dict (source of truth)
- **`ranking_pipeline/rrf.py`** — Update USE_CASES weights to use new dimension keys
- **`ranking_pipeline/validate.py`** — Update hardcoded dimension list + column headers in `print_validation_report()`

## Part 2: Query Interface (`ranking_pipeline/query.py`)

New module with three capabilities:

### A. `interpret_query(user_input) -> dict[str, float]`
- Takes natural language like "fun and surprising", "practical stuff I can use", "something like that great essay about consciousness"
- Uses LLM to produce dimension weights (0.0-3.0 scale)
- Returns dict like `{"enjoyment": 2.5, "utility": 0.5, "clarity": 1.0, "surprise": 2.0, "stickiness": 1.5}`

### B. `get_top_essays(conn, weights, n=10) -> list[dict]`
- Queries DB for BT scores across dimensions
- Applies weighted combination of BT scores
- Returns top N essays with scores, titles, snippets
- Falls back to RRF scores if no BT scores exist

### C. `suggest_queries() -> list[str]`
- Returns example queries to help users get started
- e.g. "Something fun and mind-bending", "Short practical advice I can use today", "Essays that would make a great podcast episode"

## Part 3: CLI Entry Point (`query.py` at root)

```
python query.py "fun and surprising"
python query.py --suggest          # show example queries
python query.py --use-case substack  # still support preset use cases
python query.py --top 5            # limit results
```

### DB query functions needed in `db.py`:
- `get_bt_scores(conn) -> dict` — fetch all BT scores grouped by essay+dimension
- `get_essay_details(conn, essay_ids) -> list[dict]` — fetch essay text, title, episode info

## Order of operations:
1. Add new DB query helpers to `db.py`
2. Update DIMENSIONS in `ranker.py`
3. Update USE_CASES in `rrf.py`
4. Update `validate.py` hardcoded references
5. Create `ranking_pipeline/query.py` with interpret_query, get_top_essays, suggest_queries
6. Create `query.py` CLI entry point
7. Test the query interface
