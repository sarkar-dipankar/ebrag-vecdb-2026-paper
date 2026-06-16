"""
C4, properly scoped: conflict detection on LONG, answer-bearing passage pairs.

The realism step (`c4_realism`) showed pairwise conflict is ill-posed when a passage is a
multi-hop evidence fragment rather than an answer. So here BOTH passages assert a specific
answer to the same single-hop factoid question (3-4 sentence LLM-written passages), and we
test whether the detector ladder + answer-relevance finding survive realistic *length* when
conflict is well-defined:

- ``conflict``    : passage asserting answer A vs passage asserting a different answer B (gold conflict).
- ``agreement``   : two passages both asserting A (gold no-conflict).
- ``topic_shift`` : passage asserting A vs an answer-bearing passage about a DIFFERENT subject (gold no-conflict).

Detectors: cross-encoder NLI, LLM bare, LLM answer-relevant, with bootstrap CIs.

Run:  python -m ebrag.benchmarks.experiments.c4_answer_bearing [judge_model] [gen_model]
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from ebrag.benchmarks.experiments.llm_helpers import make_ollama_generate
from ebrag.benchmarks.experiments.c1_neural_burial import TRIPLES
from ebrag.benchmarks.experiments.c4_conflict_detection import score_detections
from ebrag.benchmarks.experiments.c4_detector_comparison import bootstrap_ci
from ebrag.benchmarks.sweeps import DEFAULT_MODEL
from ebrag.common import get_logger
from ebrag.dialectic.models import EntailmentLabel

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logger = get_logger(__name__)


def assemble_answer_bearing_pairs(items: list[dict]) -> list[dict]:
    """Build conflict/agreement/topic_shift pairs from per-question generated passages (pure).

    Each item: {question, answer_a, answer_b, p_a, p_b, p_a2}. topic_shift reuses the next
    item's answer-A passage as a guaranteed off-topic (different-subject) answer-bearing text.
    """
    pairs: list[dict] = []
    n = len(items)
    for i, it in enumerate(items):
        q, p_a, p_b, p_a2 = it["question"], it["p_a"], it["p_b"], it["p_a2"]
        if p_a.strip() and p_b.strip():
            pairs.append({"question": q, "premise": p_a, "hypothesis": p_b,
                          "gold_conflict": True, "category": "conflict"})
        if p_a.strip() and p_a2.strip():
            pairs.append({"question": q, "premise": p_a, "hypothesis": p_a2,
                          "gold_conflict": False, "category": "agreement"})
        other = items[(i + 1) % n] if n > 1 else None
        if other and p_a.strip() and other["p_a"].strip():
            pairs.append({"question": q, "premise": p_a, "hypothesis": other["p_a"],
                          "gold_conflict": False, "category": "topic_shift"})
    return pairs


def _gen_passage(complete_fn: Callable[[str], str], question: str, answer: str, vary: bool) -> str:
    extra = " Use different wording and framing from a typical statement." if vary else ""
    prompt = (
        "Write a coherent 3-4 sentence encyclopedia-style passage that clearly answers the "
        f"question, asserting that the answer is '{answer}'. State the answer explicitly and "
        f"do not mention any alternative.{extra}\n\nQuestion: {question}\nPassage:"
    )
    return complete_fn(prompt).strip()


def build_items(gen_model: str, triples: list[tuple[str, str, str]] | None = None) -> list[dict]:
    triples = triples or TRIPLES
    gen = make_ollama_generate(gen_model, max_tokens=512, temperature=0.7)
    items: list[dict] = []
    for question, ans_a, ans_b in triples:
        items.append({
            "question": question, "answer_a": ans_a, "answer_b": ans_b,
            "p_a": _gen_passage(gen, question, ans_a, vary=False),
            "p_b": _gen_passage(gen, question, ans_b, vary=False),
            "p_a2": _gen_passage(gen, question, ans_a, vary=True),
        })
    return items


def run(judge_model: str = DEFAULT_MODEL, gen_model: str = "gemma3:27b",
        max_workers: int = 6) -> dict:
    items = build_items(gen_model)
    pairs = assemble_answer_bearing_pairs(items)

    from ebrag.dialectic.conflict import ConflictDetector
    from ebrag.dialectic.llm_judge import LLMEntailmentJudge

    det = ConflictDetector()
    judge = LLMEntailmentJudge(model=judge_model, max_tokens=1024)
    judge._get_client()

    ce = [det.check_pair(p["premise"], p["hypothesis"])[0] for p in pairs]

    def task(args):
        i, mode = args
        p = pairs[i]
        if mode == "bare":
            j = judge.classify(p["premise"], p["hypothesis"])
        else:
            j = judge.judge_answer_conflict(p["question"], p["premise"], p["hypothesis"])
        return (i, mode), j.label

    jobs = [(i, m) for i in range(len(pairs)) for m in ("bare", "answer_relevant")]
    preds: dict = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for key, label in ex.map(task, jobs):
            preds[key] = label

    def records_for(get_label) -> list[dict]:
        return [{"category": p["category"], "gold_conflict": p["gold_conflict"],
                 "predicted_conflict": get_label(i) == EntailmentLabel.CONTRADICTION}
                for i, p in enumerate(pairs)]

    detectors = {
        "cross_encoder_nli": records_for(lambda i: ce[i]),
        "llm_bare": records_for(lambda i: preds[(i, "bare")]),
        "llm_answer_relevant": records_for(lambda i: preds[(i, "answer_relevant")]),
    }
    by_detector = {}
    for name, recs in detectors.items():
        m = score_detections(recs)
        m["precision_ci"] = bootstrap_ci(recs, lambda r: score_detections(r)["precision"])
        m["recall_ci"] = bootstrap_ci(recs, lambda r: score_detections(r)["recall"])
        by_detector[name] = {"metrics": m, "records": recs}

    return {
        "experiment": "c4_answer_bearing",
        "judge_model": judge_model, "gen_model": gen_model,
        "n_pairs": len(pairs),
        "category_counts": {c: sum(1 for p in pairs if p["category"] == c)
                            for c in ("conflict", "agreement", "topic_shift")},
        "by_detector": by_detector,
    }


def main() -> None:
    judge_model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    gen_model = sys.argv[2] if len(sys.argv) > 2 else "gemma3:27b"
    results = run(judge_model=judge_model, gen_model=gen_model)

    out_dir = Path("benchmarks/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "c4_answer_bearing.json"
    out_path.write_text(json.dumps(results, indent=2))

    print(f"C4 answer-bearing: {results['n_pairs']} pairs {results['category_counts']}, judge={judge_model}\n")
    hdr = f"{'detector':20s} | P (CI)            R (CI)            F1    per-cat acc"
    print(hdr); print("-" * len(hdr))
    for name, d in results["by_detector"].items():
        m = d["metrics"]; pci = m["precision_ci"]; rci = m["recall_ci"]
        pca = {c: round(a, 2) for c, a in m["per_category_accuracy"].items()}
        print(f"{name:20s} | {m['precision']:.2f}[{pci[0]:.2f},{pci[1]:.2f}]  "
              f"{m['recall']:.2f}[{rci[0]:.2f},{rci[1]:.2f}]  {m['f1']:.2f}  {pca}")
    print(f"\n  -> wrote {out_path}")


if __name__ == "__main__":
    main()
