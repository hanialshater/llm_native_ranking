"""LLM client configuration for DeepSeek (OpenAI-compatible API)."""

import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI


# DeepSeek defaults — override with env vars
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"  # DeepSeek-V3
REASONING_MODEL = "deepseek-reasoner"  # DeepSeek-R1

# Retry settings
MAX_RETRIES = 4
INITIAL_BACKOFF = 2  # seconds
REQUEST_TIMEOUT = 120  # seconds

# Concurrency settings
MAX_CONCURRENT = int(os.environ.get("LLM_MAX_CONCURRENT", "4"))

# Module-level client (reused across calls)
_client = None

# Semaphore to throttle concurrent LLM requests
_semaphore = threading.Semaphore(MAX_CONCURRENT)


def get_client():
    """Get an OpenAI-compatible client configured for DeepSeek (cached)."""
    global _client
    if _client is not None:
        return _client
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL)
    if not api_key:
        raise ValueError(
            "Set DEEPSEEK_API_KEY environment variable. "
            "Get one at https://platform.deepseek.com/"
        )
    _client = OpenAI(api_key=api_key, base_url=base_url, timeout=REQUEST_TIMEOUT)
    return _client


def chat(prompt, model=None, max_tokens=4000, temperature=0.0):
    """
    Single-turn chat completion with retry and exponential backoff.
    Throttled by a global semaphore to avoid API rate limits.

    Args:
        prompt: User message string.
        model: Model name (defaults to DEFAULT_MODEL).
        max_tokens: Max response tokens.
        temperature: Sampling temperature.

    Returns:
        Response text string.
    """
    with _semaphore:
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


def run_parallel(fn, args_list, label="tasks"):
    """
    Run a function across multiple inputs in parallel, throttled by the global semaphore.

    Args:
        fn: Callable to execute. Called as fn(*args) for each args tuple.
        args_list: List of argument tuples.
        label: Label for progress logging.

    Returns:
        List of results in the same order as args_list.
    """
    results = [None] * len(args_list)
    errors = []

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        future_to_idx = {
            executor.submit(fn, *args): idx
            for idx, args in enumerate(args_list)
        }
        done_count = 0
        total = len(args_list)
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            done_count += 1
            try:
                results[idx] = future.result()
            except Exception as e:
                errors.append((idx, e))
                print(f"    [{done_count}/{total}] {label} item {idx} failed: {e}")

    if errors:
        print(f"  {len(errors)}/{total} {label} failed")
    return results
