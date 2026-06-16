"""
Curated model sweeps for multi-model ablation (Ollama Cloud).

The agenda's headline claims (counter-evidence burial, conflict-stratified
miscalibration) must hold *across* generators, not for one model. This module pins a
default single-model and a curated ablation set spanning families, sizes, and
reasoning vs non-reasoning models. Names are validated against the live
``client.models.list()`` on 2026-05-27; re-check before a final run.

The ``reasoning`` flag matters operationally: reasoning models (e.g. gpt-oss) emit
chain-of-thought in a separate ``message.reasoning`` field and need a generous
``max_tokens`` or ``content`` comes back empty (see ``memory: ollama-cloud-setup``).
"""

from __future__ import annotations

from pydantic import BaseModel


class ModelSpec(BaseModel):
    """A model in the ablation sweep, with metadata for analysis and serving."""

    name: str  # Ollama Cloud model id
    family: str
    size_b: float | None  # approx parameter count in billions (None if unknown)
    reasoning: bool = False  # emits a separate reasoning channel -> needs big max_tokens


#: Default generation model for single-model runs and the paper's headline tables.
DEFAULT_MODEL = "gpt-oss:120b"

#: Cheap model for fast local iteration / smoke tests.
SMOKE_MODEL = "gpt-oss:20b"

#: Curated ablation sweep: 10 models, ~8 families, 4B -> 671B, reasoning + non-reasoning.
DEFAULT_SWEEP: list[ModelSpec] = [
    ModelSpec(name="gpt-oss:20b", family="gpt-oss", size_b=20, reasoning=True),
    ModelSpec(name="gpt-oss:120b", family="gpt-oss", size_b=120, reasoning=True),
    ModelSpec(name="gemma3:4b", family="gemma", size_b=4),
    ModelSpec(name="gemma3:12b", family="gemma", size_b=12),
    ModelSpec(name="gemma3:27b", family="gemma", size_b=27),
    ModelSpec(name="qwen3-next:80b", family="qwen", size_b=80),
    ModelSpec(name="deepseek-v3.1:671b", family="deepseek", size_b=671, reasoning=True),
    ModelSpec(name="glm-4.6", family="glm", size_b=None),
    ModelSpec(name="ministral-3:8b", family="mistral", size_b=8),
    ModelSpec(name="nemotron-3-nano:30b", family="nemotron", size_b=30),
]


def sweep_names(sweep: list[ModelSpec] | None = None) -> list[str]:
    """Model ids in a sweep (defaults to :data:`DEFAULT_SWEEP`)."""
    return [m.name for m in (sweep or DEFAULT_SWEEP)]


def get_spec(name: str, sweep: list[ModelSpec] | None = None) -> ModelSpec | None:
    """Look up a :class:`ModelSpec` by model id."""
    for m in sweep or DEFAULT_SWEEP:
        if m.name == name:
            return m
    return None


def is_reasoning_model(name: str) -> bool:
    """Best-effort: does this model emit a separate reasoning channel?

    Falls back to a name heuristic for models outside the curated sweep so the harness
    can default to a generous ``max_tokens``.
    """
    spec = get_spec(name)
    if spec is not None:
        return spec.reasoning
    lowered = name.lower()
    return any(tag in lowered for tag in ("gpt-oss", "thinking", "-r1", "reason"))
