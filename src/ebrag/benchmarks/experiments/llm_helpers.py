"""Shared LLM-completion helper for the C4 experiment scripts.

Holds ``make_ollama_generate``, used by the scripts that need an LLM completion call
(for LLM-writing the long answer-bearing passages and the corrected HotpotQA-style
refutations).
"""

from __future__ import annotations

from typing import Callable

from ebrag.common import get_logger
from ebrag.common.config import get_settings

logger = get_logger(__name__)


def make_ollama_generate(
    model: str, max_tokens: int = 512, temperature: float = 0.7
) -> Callable[[str], str]:
    """Return a ``str -> str`` completion fn over the configured OpenAI-compatible backend.

    ``max_tokens`` is generous because reasoning models (gpt-oss) spend tokens on a hidden
    channel before emitting ``content``.
    """
    from openai import OpenAI

    client = OpenAI(**get_settings().llm.openai_client_kwargs())

    def generate(prompt: str) -> str:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001 - degrade, don't crash the run
            logger.error("ollama_generate_failed", model=model, error=str(e))
            return ""

    return generate
