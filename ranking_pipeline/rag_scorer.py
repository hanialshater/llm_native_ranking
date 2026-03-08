"""RAG-based one-by-one essay scoring.

Scores each essay individually on all dimensions, using the original episode
text as retrieval-augmented context. Produces absolute 1-10 scores (not
relative rankings), so essays can be compared without seeing each other.
"""

import json
from .llm import chat, run_parallel, DEFAULT_MODEL
from .ranker import DIMENSIONS


SCORE_PROMPT = """You are an expert essay evaluator. Score this essay on ONE specific dimension.

**Dimension:** {dimension_description}

**Original source context (for reference — the essay was extracted from this):**
{context}

**Essay to score:**
Title: {title}
{essay_text}

**Instructions:**
- Score the essay from 1 to 10 on this dimension ONLY
- 1 = worst possible, 10 = best possible
- Use the full range. Most essays should land between 3-8. Reserve 9-10 for truly exceptional work.
- The source context helps you understand where the essay came from, but score the essay on its own merit
- Provide brief reasoning (1-2 sentences)

Return ONLY a JSON object:
{{"score": 7, "reasoning": "Brief explanation here"}}

Return only the JSON object, nothing else."""


def _strip_code_fences(text):
    """Strip markdown code fences from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        end = -1 if lines[-1].strip().startswith("```") else len(lines)
        text = "\n".join(lines[1:end])
    return text.strip()


def score_essay_dimension(essay_text, title, context, dimension, model=None):
    """
    Score a single essay on a single dimension with RAG context.

    Args:
        essay_text: The essay text to score.
        title: Essay title.
        context: Original episode/source text for RAG augmentation.
        dimension: One of DIMENSIONS keys.
        model: Model name.

    Returns:
        Dict with 'score' (float 1-10) and 'reasoning' (str).
    """
    dim_desc = DIMENSIONS[dimension]
    # Truncate context to avoid token limits (keep first 4KB)
    if context and len(context) > 4000:
        context = context[:4000] + "\n[... truncated ...]"

    prompt = SCORE_PROMPT.format(
        dimension_description=dim_desc,
        context=context or "(no source context available)",
        title=title or "Untitled",
        essay_text=essay_text,
    )
    text = chat(prompt, model=model, max_tokens=300)
    text = _strip_code_fences(text)
    result = json.loads(text)

    # Clamp score to 1-10
    score = max(1.0, min(10.0, float(result.get("score", 5))))
    reasoning = result.get("reasoning", "")
    return {"score": score, "reasoning": reasoning}


def score_essay_all_dimensions(essay_text, title, context, model=None):
    """
    Score a single essay on all 5 dimensions in parallel.

    Args:
        essay_text: The essay text.
        title: Essay title.
        context: Original episode text for RAG.
        model: Model name.

    Returns:
        Dict of {dimension: {score, reasoning}}.
    """
    dims = list(DIMENSIONS.keys())
    args_list = [(essay_text, title, context, dim, model) for dim in dims]
    results = run_parallel(score_essay_dimension, args_list, label="dimensions")
    return {
        dim: result
        for dim, result in zip(dims, results)
        if result is not None
    }


def score_essays_batch(essays_with_context, model=None):
    """
    Score a batch of essays, each on all dimensions.

    Each essay is scored independently (one-by-one, not compared to others).
    All dimension calls for all essays run in parallel.

    Args:
        essays_with_context: List of dicts with keys:
            - id: Essay ID
            - text: Essay text
            - title: Essay title
            - context: Original episode text (RAG augmentation)
        model: Model name.

    Returns:
        Dict of {essay_id: {dimension: {score, reasoning}}}.
    """
    # Build flat list of all (essay, dimension) scoring tasks
    all_tasks = []
    task_keys = []  # (essay_id, dimension)
    for essay in essays_with_context:
        for dim in DIMENSIONS:
            all_tasks.append((
                essay["text"],
                essay.get("title", ""),
                essay.get("context", ""),
                dim,
                model,
            ))
            task_keys.append((essay["id"], dim))

    print(f"  Scoring {len(essays_with_context)} essays × {len(DIMENSIONS)} dimensions = {len(all_tasks)} LLM calls")
    results = run_parallel(score_essay_dimension, all_tasks, label="RAG scores")

    # Reassemble into {essay_id: {dimension: result}}
    scores = {}
    for (eid, dim), result in zip(task_keys, results):
        if result is not None:
            if eid not in scores:
                scores[eid] = {}
            scores[eid][dim] = result

    return scores
