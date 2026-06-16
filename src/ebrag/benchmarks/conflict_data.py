"""
Controlled conflict datasets for the C1 / C2 / C4 experiments.

A :class:`ConflictQuestion` bundles a question with a pool of **stance-labelled
passages** (supporting / refuting / neutral / distractor) and a ``conflict_label`` saying
whether genuine disagreement is present. The labelled passages give the *gold* sets that
:func:`ebrag.benchmarks.conflict_metrics.stance_recall` consumes, so we can measure
whether a retriever buries counter-evidence.

We get conflict items two ways:

- **Synthetic injection** (:class:`SyntheticConflictInjector`): take a normal QA item with
  its supporting passages and add a controlled *refuting* passage. The ``templated``
  strategy is deterministic (no API key); the ``llm`` strategy writes a fluent false
  passage via an injectable ``generate_fn`` (e.g. Ollama).
- **Native conflict datasets** (FEVER REFUTES, AmbigDocs, RAMDocs): wired in the loaders
  when the pilot installs ``datasets``; they populate the same model.

This keeps the gold counter-evidence labels explicit and controllable, which the C1
measurement needs.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel, Field


class Stance(str, Enum):
    """The role a passage plays w.r.t. the question's correct answer."""

    SUPPORTING = "supporting"  # supports the correct answer
    REFUTING = "refuting"  # asserts/argues a different (incorrect or minority) answer
    NEUTRAL = "neutral"  # on-topic but non-committal
    DISTRACTOR = "distractor"  # off-topic noise


class StancePassage(BaseModel):
    """A retrievable passage with a gold stance label."""

    id: str
    text: str
    stance: Stance
    source_id: str = ""


class ConflictQuestion(BaseModel):
    """A question plus a pool of stance-labelled passages and a conflict flag."""

    id: str
    question: str
    answer: str | list[str]
    passages: list[StancePassage] = Field(default_factory=list)
    #: True when the passage pool contains genuine disagreement (>=1 refuting passage).
    conflict_label: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def passage_ids(self) -> list[str]:
        return [p.id for p in self.passages]

    @property
    def passage_texts(self) -> list[str]:
        return [p.text for p in self.passages]

    @property
    def supporting_ids(self) -> list[str]:
        return [p.id for p in self.passages if p.stance == Stance.SUPPORTING]

    @property
    def counter_evidence_ids(self) -> list[str]:
        """Gold refuting/minority passages -- the counter-evidence stance_recall tracks."""
        return [p.id for p in self.passages if p.stance == Stance.REFUTING]


def _first_answer(answer: str | list[str]) -> str:
    return answer[0] if isinstance(answer, list) else answer


class SyntheticConflictInjector:
    """Turn standard QA items into controlled :class:`ConflictQuestion` s.

    Args:
        generate_fn: Optional ``str -> str`` callable used by the ``llm`` strategy to
            write a refuting passage (e.g. a wrapper around an Ollama completion). The
            ``templated`` strategy ignores it and stays deterministic.
    """

    def __init__(self, generate_fn: Callable[[str], str] | None = None) -> None:
        self._generate_fn = generate_fn

    def build(
        self,
        question_id: str,
        question: str,
        answer: str | list[str],
        supporting_texts: list[str],
        distractor_texts: list[str] | None = None,
        false_answer: str | None = None,
        strategy: str = "templated",
    ) -> ConflictQuestion:
        """Build a conflicting item: supporting + distractor passages + one refuting passage."""
        passages = self._stance_passages(question_id, supporting_texts, distractor_texts)
        refutation = self._make_refutation(question, answer, false_answer, strategy)
        passages.append(
            StancePassage(
                id=f"{question_id}-ctr-0",
                text=refutation,
                stance=Stance.REFUTING,
                source_id="synthetic-refutation",
            )
        )
        return ConflictQuestion(
            id=question_id,
            question=question,
            answer=answer,
            passages=passages,
            conflict_label=True,
            metadata={"strategy": strategy, "false_answer": false_answer},
        )

    def build_unanimous(
        self,
        question_id: str,
        question: str,
        answer: str | list[str],
        supporting_texts: list[str],
        distractor_texts: list[str] | None = None,
    ) -> ConflictQuestion:
        """Build a non-conflicting control item (no refuting passage).

        Needed as the unanimous stratum for EDHS and conflict-stratified calibration.
        """
        passages = self._stance_passages(question_id, supporting_texts, distractor_texts)
        return ConflictQuestion(
            id=question_id,
            question=question,
            answer=answer,
            passages=passages,
            conflict_label=False,
            metadata={"strategy": "none"},
        )

    @staticmethod
    def _stance_passages(
        question_id: str,
        supporting_texts: list[str],
        distractor_texts: list[str] | None,
    ) -> list[StancePassage]:
        passages = [
            StancePassage(id=f"{question_id}-sup-{i}", text=t, stance=Stance.SUPPORTING)
            for i, t in enumerate(supporting_texts)
        ]
        passages += [
            StancePassage(id=f"{question_id}-dist-{i}", text=t, stance=Stance.DISTRACTOR)
            for i, t in enumerate(distractor_texts or [])
        ]
        return passages

    def _make_refutation(
        self,
        question: str,
        answer: str | list[str],
        false_answer: str | None,
        strategy: str,
    ) -> str:
        ans = _first_answer(answer)
        if strategy == "templated":
            if false_answer:
                return (
                    f'According to some sources, the answer to "{question}" is '
                    f"{false_answer}, not {ans}."
                )
            return (
                f'Contrary to other reports, it is not the case that the answer to '
                f'"{question}" is {ans}.'
            )
        if strategy == "llm":
            if self._generate_fn is None:
                raise ValueError("the 'llm' strategy requires a generate_fn")
            prompt = (
                "Write a single short, fluent passage (2-3 sentences) that plausibly but "
                "INCORRECTLY argues against the correct answer below, asserting a different "
                "answer. Do not mention that it is incorrect.\n\n"
                f'Question: {question}\nCorrect answer: {ans}\n\nPassage:'
            )
            return self._generate_fn(prompt).strip()
        raise ValueError(f"unknown strategy: {strategy!r}")
