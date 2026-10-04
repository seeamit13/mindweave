"""
Thin wrapper around the Groq API used for both RAG answer generation
and artifact generation (reports/quizzes). Keeping this in one place makes
it trivial to swap models or add retry/error-handling logic in one spot.

Groq (https://groq.com) hosts open-weight models (Llama, etc.) behind an
OpenAI-compatible chat completions endpoint and is used here purely for
fast, low-cost generation - all retrieval/embeddings stay local regardless
of which LLM provider is configured.
"""

from __future__ import annotations

import groq

from . import config


class LLMError(Exception):
    pass


_client = None


def _get_client() -> "groq.Groq":
    global _client
    if _client is None:
        if not config.GROQ_API_KEY:
            raise LLMError(
                "GROQ_API_KEY is not set. Add it as an environment variable "
                "(locally in a .env file, or as a Hugging Face Space secret). "
                "Get a free key at https://console.groq.com/keys."
            )
        _client = groq.Groq(api_key=config.GROQ_API_KEY)
    return _client


def generate(system_prompt: str, user_prompt: str, max_tokens: int = None) -> str:
    """Single-turn generation call. Raises LLMError on failure so callers
    can surface a clean error message in the UI instead of crashing."""
    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            max_tokens=max_tokens or config.LLM_MAX_TOKENS,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (response.choices[0].message.content or "").strip()
    except groq.APIError as e:
        raise LLMError(f"LLM request failed: {e}") from e
    except Exception as e:
        raise LLMError(f"Unexpected error calling the LLM: {e}") from e
