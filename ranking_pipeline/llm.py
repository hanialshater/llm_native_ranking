"""LLM client configuration for DeepSeek (OpenAI-compatible API)."""

import os
from openai import OpenAI


# DeepSeek defaults — override with env vars
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"  # DeepSeek-V3
REASONING_MODEL = "deepseek-reasoner"  # DeepSeek-R1


def get_client():
    """Get an OpenAI-compatible client configured for DeepSeek."""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL)
    if not api_key:
        raise ValueError(
            "Set DEEPSEEK_API_KEY environment variable. "
            "Get one at https://platform.deepseek.com/"
        )
    return OpenAI(api_key=api_key, base_url=base_url)


def chat(prompt, model=None, max_tokens=4000, temperature=0.0):
    """
    Single-turn chat completion.

    Args:
        prompt: User message string.
        model: Model name (defaults to DEFAULT_MODEL).
        max_tokens: Max response tokens.
        temperature: Sampling temperature.

    Returns:
        Response text string.
    """
    client = get_client()
    response = client.chat.completions.create(
        model=model or DEFAULT_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
