"""Listwise LLM ranking per dimension."""

import json
from .llm import chat, run_parallel, DEFAULT_MODEL


DIMENSIONS = {
    "enjoyment": """ENJOYMENT — Fun to read.
    Voice, rhythm, wit, narrative pull. You keep reading because you want to, not because you should.
    High enjoyment = you'd read it on a Saturday morning for fun. Low = feels like homework.""",
    "utility": """PRACTICAL UTILITY — Gives you something you can use.
    Actionable takeaways, concrete frameworks, ideas you can apply today.
    High utility = you change how you do something after reading. Low = interesting but inert.""",
    "clarity": """CLARITY — Easy to follow without re-reading.
    Clean structure, logical flow, no ambiguity. The writer did the hard work so you don't have to.
    High clarity = you could explain it to a friend right after. Low = you'd need to re-read twice.""",
    "surprise": """SURPRISE — Teaches you something you didn't expect.
    Reframes the familiar, introduces a genuinely new angle, makes you see differently.
    High surprise = 'I never thought of it that way.' Low = confirms what you already know.""",
    "stickiness": """STICKINESS — Stays with you after you close the tab.
    You'd quote it, share it, or think about it days later.
    High stickiness = you text a friend the link. Low = forgotten by lunch.""",
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
    Rank essays on all dimensions (in parallel).

    Args:
        essays: List of dicts with 'id' and 'text' keys.
        model: Model name.

    Returns:
        Dict of {dimension: [essay_ids ranked best-first]}.
    """
    dims = list(DIMENSIONS.keys())
    print(f"  Ranking on {len(dims)} dimensions in parallel: {', '.join(dims)}")
    args_list = [(essays, dim, model) for dim in dims]
    results = run_parallel(rank_essays, args_list, label="dimensions")
    return {dim: result for dim, result in zip(dims, results) if result is not None}
