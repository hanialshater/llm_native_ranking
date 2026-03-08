"""Listwise LLM ranking per dimension."""

import json
from .llm import chat, DEFAULT_MODEL


DIMENSIONS = {
    "aporia": """APORIA — Leaves you genuinely stuck, unable to resolve.
    The essay raises a question or tension the reader cannot immediately dissolve.
    High aporia = you put it down thinking. Low aporia = you nod and move on.""",
    "compression": """COMPRESSION — Collapses a big idea into a dense, portable unit.
    The essay says much in little. You could quote one sentence and it carries the whole.
    High compression = quotable, tweet-able, memorable. Low = requires context to land.""",
    "defam": """DEFAMILIARIZATION — Makes something familiar strange.
    The essay takes something you thought you understood and renders it suddenly odd.
    High defam = 'I never thought of it that way'. Low = confirms what you already know.""",
    "tension": """GENERATIVE TENSION — Opens new thought rather than closes it.
    Reading it produces further questions, connections, implications.
    High tension = you reach for a pen. Low = satisfying closure, nothing to add.""",
    "force": """STANDALONE FORCE — Hits hard without any context.
    A cold reader with no background in the topic would still feel the impact.
    High force = works as a tweet, a pull quote, an epigraph. Low = requires setup.""",
}

RANK_PROMPT = """You will rank the following essays by one specific dimension.

**Dimension:** {dimension_description}

**Essays:**
{essays}

**Instructions:**
- Rank ALL essays from best to worst on this dimension only
- No ties — force strict ordering
- Return ONLY a JSON array of essay numbers in ranked order (best first):
  [3, 7, 1, 12, ...]

Return only the JSON array, nothing else."""


def _strip_code_fences(text):
    """Strip markdown code fences from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line if it's ```)
        end = -1 if lines[-1].strip().startswith("```") else len(lines)
        text = "\n".join(lines[1:end])
    return text.strip()


def rank_essays(essays, dimension, model=None):
    """
    Rank essays on a single dimension via listwise LLM judgment.

    Args:
        essays: List of dicts with 'id' and 'text' keys.
        dimension: One of DIMENSIONS keys.
        model: Model name.

    Returns:
        List of essay IDs in ranked order (best first).
    """
    valid_ids = {e["id"] for e in essays}
    formatted = "\n\n".join(
        [f"**Essay {e['id']}:**\n{e['text']}" for e in essays]
    )
    dim_desc = DIMENSIONS[dimension]

    prompt = RANK_PROMPT.format(dimension_description=dim_desc, essays=formatted)
    text = chat(prompt, model=model, max_tokens=500)
    text = _strip_code_fences(text)

    ranked_ids = json.loads(text)

    # Validate: all returned IDs must be from the input set
    returned = set(ranked_ids)
    if returned != valid_ids:
        missing = valid_ids - returned
        extra = returned - valid_ids
        if extra:
            ranked_ids = [i for i in ranked_ids if i in valid_ids]
        if missing:
            ranked_ids.extend(missing)
        print(f"    Warning: LLM returned mismatched IDs (missing={missing}, extra={extra}), corrected")

    return ranked_ids


def rank_all_dimensions(essays, model=None):
    """
    Rank essays on all dimensions.

    Args:
        essays: List of dicts with 'id' and 'text' keys.
        model: Model name.

    Returns:
        Dict of {dimension: [essay_ids ranked best-first]}.
    """
    rankings = {}
    for dim in DIMENSIONS:
        print(f"  Ranking on: {dim}")
        rankings[dim] = rank_essays(essays, dim, model)
    return rankings
