"""
LLM-as-judge entailment / conflict detection.

A training-free, torch-free alternative to the cross-encoder NLI in
:mod:`ebrag.dialectic.conflict`. It asks an instruction-tuned LLM (via the configured
OpenAI-compatible backend, e.g. Ollama Cloud) to label the relationship between two
texts as entailment / contradiction / neutral.

It is used by the C4 experiments to (i) compare an LLM-as-judge against the DeBERTa
cross-encoder, and (ii) provide a CPU-only pair classifier that runs without
``torch`` / ``transformers``.

``check_pair`` is signature-compatible with
:meth:`ebrag.dialectic.conflict.ConflictDetector.check_pair`. The LLM call is injectable
(pass ``client``) and the response parsing is a pure, unit-tested function.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

from ebrag.common import get_logger, get_settings
from ebrag.dialectic.models import EntailmentLabel

logger = get_logger(__name__)


_PROMPT_TEMPLATE = """You are a strict natural-language-inference annotator.
Given a PREMISE and a HYPOTHESIS, decide the relationship of the HYPOTHESIS to the PREMISE:
- "entailment": the premise supports / implies the hypothesis.
- "contradiction": the premise contradicts the hypothesis.
- "neutral": neither supported nor contradicted (unrelated or insufficient).

Respond with ONLY a JSON object, no extra text:
{{"label": "entailment|contradiction|neutral", "confidence": <0.0-1.0>, "rationale": "<short>"}}

PREMISE:
{premise}

HYPOTHESIS:
{hypothesis}
"""


class EntailmentJudgement(BaseModel):
    """Structured output of the LLM entailment judge."""

    label: EntailmentLabel = EntailmentLabel.NEUTRAL
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""


_LABEL_ALIASES = {
    "entailment": EntailmentLabel.ENTAILMENT,
    "entail": EntailmentLabel.ENTAILMENT,
    "entails": EntailmentLabel.ENTAILMENT,
    "support": EntailmentLabel.ENTAILMENT,
    "supported": EntailmentLabel.ENTAILMENT,
    "contradiction": EntailmentLabel.CONTRADICTION,
    "contradict": EntailmentLabel.CONTRADICTION,
    "contradicts": EntailmentLabel.CONTRADICTION,
    "refute": EntailmentLabel.CONTRADICTION,
    "refutes": EntailmentLabel.CONTRADICTION,
    "neutral": EntailmentLabel.NEUTRAL,
    "unrelated": EntailmentLabel.NEUTRAL,
}


def parse_judgement(raw: str) -> EntailmentJudgement:
    """Parse a model response into an :class:`EntailmentJudgement`.

    Robust to (a) clean JSON, (b) JSON embedded in surrounding prose, and (c) no JSON at
    all (falls back to scanning for a label keyword). Unknown / unparseable input yields
    a low-confidence ``neutral`` judgement rather than raising.
    """
    if not raw or not raw.strip():
        return EntailmentJudgement()

    # 1) Try to locate and parse a JSON object.
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            label = _coerce_label(str(data.get("label", "")))
            if label is not None:
                conf = data.get("confidence", 0.5)
                try:
                    conf = float(conf)
                except (TypeError, ValueError):
                    conf = 0.5
                conf = min(1.0, max(0.0, conf))
                return EntailmentJudgement(
                    label=label,
                    confidence=conf,
                    rationale=str(data.get("rationale", "")),
                )
        except (json.JSONDecodeError, ValueError):
            pass

    # 2) Fallback: scan prose for a label keyword.
    lowered = raw.lower()
    for alias, label in _LABEL_ALIASES.items():
        if re.search(rf"\b{alias}\b", lowered):
            return EntailmentJudgement(label=label, confidence=0.5, rationale=raw[:200])

    return EntailmentJudgement()


def _coerce_label(value: str) -> EntailmentLabel | None:
    return _LABEL_ALIASES.get(value.strip().lower())


_ANSWER_CONFLICT_PROMPT = """You judge whether two passages give CONFLICTING ANSWERS to a specific question.

Count it as a conflict ONLY IF both passages actually address the question and assert
different or incompatible answers to it. If a passage is about a different entity/sense, a
different topic, or simply does not answer the question, it is NOT a conflict.

Respond with ONLY a JSON object, no extra text:
{{"label": "conflict|no_conflict", "confidence": <0.0-1.0>, "rationale": "<short>"}}

QUESTION:
{question}

PASSAGE A:
{passage_a}

PASSAGE B:
{passage_b}
"""

_CONFLICT_WORDS = ("conflict", "contradiction", "conflicting", "incompatible")
_NO_CONFLICT_WORDS = ("no_conflict", "no-conflict", "no conflict", "not_conflict", "none", "neutral")


def parse_answer_conflict(raw: str) -> EntailmentJudgement:
    """Parse an answer-conflict response into CONTRADICTION (conflict) or NEUTRAL (none).

    Robust to JSON and to prose; checks the no-conflict phrasing before the conflict
    phrasing so "no_conflict" is never misread as "conflict".
    """
    if not raw or not raw.strip():
        return EntailmentJudgement()

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            label_raw = str(data.get("label", "")).strip().lower()
            conf = data.get("confidence", 0.5)
            try:
                conf = min(1.0, max(0.0, float(conf)))
            except (TypeError, ValueError):
                conf = 0.5
            rationale = str(data.get("rationale", ""))
            if label_raw in _NO_CONFLICT_WORDS:
                return EntailmentJudgement(
                    label=EntailmentLabel.NEUTRAL, confidence=conf, rationale=rationale
                )
            if label_raw in _CONFLICT_WORDS:
                return EntailmentJudgement(
                    label=EntailmentLabel.CONTRADICTION, confidence=conf, rationale=rationale
                )
        except (json.JSONDecodeError, ValueError):
            pass

    lowered = raw.lower()
    if any(w in lowered for w in _NO_CONFLICT_WORDS) or "not a conflict" in lowered:
        return EntailmentJudgement(label=EntailmentLabel.NEUTRAL, confidence=0.5, rationale=raw[:200])
    if any(w in lowered for w in _CONFLICT_WORDS):
        return EntailmentJudgement(label=EntailmentLabel.CONTRADICTION, confidence=0.5, rationale=raw[:200])
    return EntailmentJudgement()


class LLMEntailmentJudge:
    """LLM-backed entailment/conflict classifier (OpenAI-compatible backend)."""

    def __init__(
        self,
        model: str | None = None,
        client: Any = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> None:
        self.settings = get_settings()
        self.model = model or self.settings.llm.model
        # Reasoning models (gpt-oss etc.) spend tokens on a hidden channel, so keep the
        # budget generous or `content` returns empty.
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = client
        logger.info("llm_entailment_judge_created", model=self.model)

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(**self.settings.llm.openai_client_kwargs())
        return self._client

    def _complete(self, prompt: str) -> str:
        """Single completion; returns the message content (empty string on failure)."""
        try:
            resp = self._get_client().chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001 - judge must degrade, not crash a sweep
            logger.error("llm_judge_completion_failed", model=self.model, error=str(e))
            return ""

    def classify(self, premise: str, hypothesis: str) -> EntailmentJudgement:
        """Classify the NLI relationship of ``hypothesis`` to ``premise``."""
        prompt = _PROMPT_TEMPLATE.format(premise=premise, hypothesis=hypothesis)
        return parse_judgement(self._complete(prompt))

    def check_pair(self, text_a: str, text_b: str) -> tuple[EntailmentLabel, float]:
        """Drop-in replacement for ``ConflictDetector.check_pair``.

        Returns ``(label, confidence)`` for the relationship of ``text_b`` to ``text_a``.
        """
        j = self.classify(text_a, text_b)
        return j.label, j.confidence

    def verify_claim(self, claim: str, evidence: str) -> EntailmentJudgement:
        """Faithfulness check: is ``claim`` entailed by ``evidence``?"""
        return self.classify(premise=evidence, hypothesis=claim)

    def judge_answer_conflict(
        self, question: str, passage_a: str, passage_b: str
    ) -> EntailmentJudgement:
        """Answer-relevance conflict: do A and B give conflicting answers to ``question``?

        Returns CONTRADICTION when they assert incompatible answers to the question, else
        NEUTRAL. This is the C4 "answer-relevant" framing that should suppress the
        same-name-different-referent false positives of bare pairwise NLI.
        """
        prompt = _ANSWER_CONFLICT_PROMPT.format(
            question=question, passage_a=passage_a, passage_b=passage_b
        )
        return parse_answer_conflict(self._complete(prompt))
