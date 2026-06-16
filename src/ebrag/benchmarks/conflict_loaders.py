"""
Loaders that turn natural QA datasets into :class:`ConflictQuestion` s for C1.

HotpotQA (distractor setting) is ideal as a first natural corpus: each question ships
with 10 real Wikipedia paragraphs, 2 of which are gold *supporting* (named in
``supporting_facts``) and the rest distractors. We add one *counter-evidence* passage by
LLM-generated refutation (natural vocabulary, so it does not trivially echo the question
the way the templated synthetic refutations do) — this lets retrieval-side burial actually
manifest while keeping the gold counter-evidence label explicit.

The dataset transform (``hotpot_example_to_texts``) is pure and unit-tested; the loader
itself lazily imports ``datasets`` and streams (no full download).
"""

from __future__ import annotations

from typing import Any, Callable, Iterator

from ebrag.benchmarks.conflict_data import ConflictQuestion, SyntheticConflictInjector
from ebrag.common import get_logger

logger = get_logger(__name__)


def hotpot_example_to_texts(example: dict[str, Any]) -> tuple[str, str, list[str], list[str]]:
    """Split a HotpotQA distractor example into (question, answer, supporting, distractor).

    Pure: takes the HF example dict, returns the question, gold answer, the gold supporting
    paragraphs (titles named in ``supporting_facts``), and the distractor paragraphs.
    """
    question = example["question"]
    answer = example["answer"]
    ctx = example["context"]
    titles: list[str] = ctx["title"]
    sentences: list[list[str]] = ctx["sentences"]
    sup_titles = set(example["supporting_facts"]["title"])

    supporting: list[str] = []
    distractor: list[str] = []
    for title, sents in zip(titles, sentences):
        paragraph = " ".join(sents).strip()
        if not paragraph:
            continue
        (supporting if title in sup_titles else distractor).append(paragraph)
    return question, answer, supporting, distractor


def load_hotpotqa_conflict(
    n: int,
    generate_fn: Callable[[str], str] | None = None,
    split: str = "validation",
    refute_strategy: str = "llm",
) -> list[ConflictQuestion]:
    """Stream ``n`` HotpotQA questions and build ConflictQuestions with a refuting passage.

    Args:
        n: number of questions.
        generate_fn: ``str -> str`` used to write the natural refutation (required when
            ``refute_strategy="llm"``). Use ``make_ollama_generate`` from the C1 driver.
        split: HotpotQA split.
        refute_strategy: "llm" (natural refutation) or "templated" (deterministic).
    """
    from datasets import load_dataset

    ds = load_dataset("hotpot_qa", "distractor", split=split, streaming=True)
    injector = SyntheticConflictInjector(generate_fn=generate_fn)

    out: list[ConflictQuestion] = []
    for i, example in enumerate(_take(ds, n)):
        question, answer, supporting, distractor = hotpot_example_to_texts(example)
        if not supporting:
            continue
        qid = str(example.get("id", f"hotpot-{i}"))
        cq = injector.build(
            question_id=qid,
            question=question,
            answer=answer,
            supporting_texts=supporting,
            distractor_texts=distractor,
            strategy=refute_strategy,
        )
        out.append(cq)
    logger.info("hotpotqa_conflict_loaded", n=len(out), split=split, strategy=refute_strategy)
    return out


def _take(iterable: Iterator[Any], n: int) -> Iterator[Any]:
    for i, item in enumerate(iterable):
        if i >= n:
            break
        yield item


# --------------------------------------------------------------------------- #
# FEVER (copenlu/fever_gold_evidence) — real human-labelled (claim, evidence, label)
# --------------------------------------------------------------------------- #


def extract_evidence_text(evidence_field: Any) -> str:
    """Pull the sentence text out of a FEVER ``evidence`` field (pure).

    The field is a list of ``[article, sent_id, sentence_text]`` triples; we concatenate
    the sentence texts (index 2). Handles missing/odd shapes by returning "".
    """
    if not evidence_field:
        return ""
    sents: list[str] = []
    for item in evidence_field:
        if isinstance(item, list) and len(item) >= 3 and item[2]:
            sents.append(str(item[2]))
    return " ".join(sents).strip()


def load_fever_evidence_pairs(
    n_per_class: int = 50, split: str = "validation"
) -> list[dict]:
    """Load human-labelled (claim, evidence, label) pairs from copenlu/fever_gold_evidence.

    Returns up to ``n_per_class`` of each of REFUTES and SUPPORTS (NOT ENOUGH INFO skipped),
    each as ``{claim, evidence, gold_conflict, category}`` where ``gold_conflict`` is True
    iff label is REFUTES. VERIFIABLE-only.
    """
    from datasets import load_dataset

    ds = load_dataset("copenlu/fever_gold_evidence", split=split, streaming=True)
    out: list[dict] = []
    seen = {"REFUTES": 0, "SUPPORTS": 0}
    for ex in ds:
        if seen["REFUTES"] >= n_per_class and seen["SUPPORTS"] >= n_per_class:
            break
        if ex.get("verifiable") != "VERIFIABLE":
            continue
        label = ex.get("label")
        if label not in ("REFUTES", "SUPPORTS") or seen[label] >= n_per_class:
            continue
        evidence = extract_evidence_text(ex.get("evidence"))
        if not evidence:
            continue
        seen[label] += 1
        out.append({
            "claim": ex["claim"],
            "evidence": evidence,
            "gold_conflict": label == "REFUTES",
            "category": label.lower(),
            "id": ex.get("id", f"fever-{len(out)}"),
        })
    logger.info("fever_pairs_loaded", refutes=seen["REFUTES"], supports=seen["SUPPORTS"])
    return out
