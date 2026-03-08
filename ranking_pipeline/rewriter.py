"""Rewrite raw transcripts/texts into standalone philosophical mini-essays."""

import json
from .llm import chat, run_parallel, DEFAULT_MODEL


REWRITE_PROMPT = """You are given a transcript or text. Extract the 10-15 most philosophically \
distinct claims, observations, or ideas. For each, write a standalone mini-essay that captures \
the idea fully — as long as the idea requires, no more. Do not compress or pad to a fixed length.
Do not reference the source text, episode number, or author.

Each mini-essay must:
- Stand alone — no "as discussed above" or "the author argues"
- Make one clear philosophical claim
- Be complete in itself — a cold reader should grasp it without context

Return a JSON array:
[
  {{"position": 1, "title": "Short title (5 words max)", "text": "essay..."}},
  ...
]

Return only valid JSON, no preamble."""


def rewrite_episode(raw_text, model=None):
    """
    Rewrite a raw transcript into standalone mini-essays.

    Args:
        raw_text: Raw transcript/article text.
        model: Model name (defaults to DeepSeek default).

    Returns:
        List of dicts with 'position', 'title', 'text'.
    """
    # Truncate very long texts to stay within context
    text_input = raw_text[:8000]
    prompt = f"{REWRITE_PROMPT}\n\n---\n\n{text_input}"

    text = chat(prompt, model=model)

    # Handle potential markdown code fences
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        end = -1 if lines[-1].strip().startswith("```") else len(lines)
        text = "\n".join(lines[1:end]).strip()

    essays = json.loads(text)

    # Validate structure: each item must have a 'text' field
    validated = []
    for e in essays:
        if isinstance(e, dict) and "text" in e:
            validated.append(e)
    if not validated:
        raise ValueError(f"LLM returned no valid essays (got {len(essays)} items)")
    return validated


def rewrite_batch(episodes_texts, model=None):
    """
    Rewrite multiple episodes (in parallel).

    Args:
        episodes_texts: List of raw text strings.
        model: Model name.

    Returns:
        List of lists of essay dicts.
    """
    print(f"  Rewriting {len(episodes_texts)} episodes in parallel...")
    args_list = [(text, model) for text in episodes_texts]
    results = run_parallel(rewrite_episode, args_list, label="rewrites")
    return [r if r is not None else [] for r in results]
