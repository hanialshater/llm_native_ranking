"""LLM client configuration for DeepSeek (OpenAI-compatible API)."""

import os
import time
from openai import OpenAI


# DeepSeek defaults — override with env vars
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"  # DeepSeek-V3
REASONING_MODEL = "deepseek-reasoner"  # DeepSeek-R1

# Retry settings
MAX_RETRIES = 4
INITIAL_BACKOFF = 2  # seconds


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
    Single-turn chat completion with retry and exponential backoff.

    Args:
        prompt: User message string.
        model: Model name (defaults to DEFAULT_MODEL).
        max_tokens: Max response tokens.
        temperature: Sampling temperature.

    Returns:
        Response text string.
    """
    client = get_client()
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model or DEFAULT_MODEL,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES:
                wait = INITIAL_BACKOFF * (2 ** attempt)
                print(f"    LLM call failed (attempt {attempt+1}/{MAX_RETRIES+1}): {e}")
                print(f"    Retrying in {wait}s...")
                time.sleep(wait)
    raise last_err
