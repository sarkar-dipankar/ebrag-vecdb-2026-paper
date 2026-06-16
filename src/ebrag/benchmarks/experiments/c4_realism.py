"""
C4 realism step: does the detector ladder hold on long, real, multi-claim passage pairs?

Curated probes (Pilots 0-4) are single clean sentences. Here we build realistic pairs from
HotpotQA-distractor (real multi-sentence Wikipedia paragraphs):

- ``conflict``    : a gold supporting paragraph vs an LLM-generated long refutation that
                    asserts a different answer (gold = conflict).
- ``topic_shift`` : a gold supporting paragraph vs a real distractor paragraph that does not
                    answer the question (gold = no conflict).
- ``agreement``   : two gold supporting paragraphs for the same question (gold = no conflict).

We then run the three detectors (cross-encoder NLI, LLM bare, LLM answer-relevant) with
bootstrap CIs, as in `c4_detector_comparison`, to see whether the ladder survives realistic
retrieved text.

Run:  python -m ebrag.benchmarks.experiments.c4_realism [n] [judge_model] [gen_model]
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from ebrag.benchmarks.experiments.llm_helpers import make_ollama_generate
from ebrag.benchmarks.experiments.c4_conflict_detection import score_detections
from ebrag.benchmarks.experiments.c4_detector_comparison import bootstrap_ci
from ebrag.benchmarks.sweeps import DEFAULT_MODEL
from ebrag.common import get_logger
from ebrag.dialectic.models import EntailmentLabel

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logger = get_logger(__name__)


def assemble_pairs(
    qid: str,
    question: str,
    supporting: list[str],
    distractor: list[str],
    refutation: str,
) -> list[dict]:
    """Build realistic labelled pairs from one question's paragraphs (pure)."""
    pairs: list[dict] = []
    if not supporting:
        return pairs
    s0 = supporting[0]
    if refutation.strip():
        pairs.append({"qid": qid, "question": question, "premise": s0,
                      "hypothesis": refutation, "gold_conflict": True, "category": "conflict"})
    if distractor:
        pairs.append({"qid": qid, "question": question, "premise": s0,
                      "hypothesis": distractor[0], "gold_conflict": False, "category": "topic_shift"})
    if len(supporting) >= 2:
        pairs.append({"qid": qid, "question": question, "premise": s0,
                      "hypothesis": supporting[1], "gold_conflict": False, "category": "agreement"})
    return pairs


def make_refutation_fn(complete_fn: Callable[[str], str]) -> Callable[[str, str], str]:
    """Return (question, answer) -> a long, fluent passage asserting a DIFFERENT answer."""

    def refute(question: str, answer: str) -> str:
        prompt = (
            "Write a coherent 3-4 sentence passage, in the style of an encyclopedia excerpt, "
            "that confidently asserts an INCORRECT answer to the question -- a different answer "
            f"from the correct one. Do not mention that it is incorrect.\n\nQuestion: {question}\n"
            f"Correct answer (do NOT assert this one): {answer}\nPassage:"
        )
        return complete_fn(prompt).strip()

    return refute


def make_boolean_refutation_fn(complete_fn: Callable[[str], str]) -> Callable[[str, str], str]:
    """Reliable conflict gold for yes/no questions: assert the OPPOSITE answer.

    Restricting to boolean questions removes the failure mode of the open refutation
    generator (inventing a different entity instead of conflicting on the answer): the
    opposite yes/no verdict is, by construction, a genuine answer-level conflict.
    """

    def refute(question: str, answer: str) -> str:
        opposite = "no" if answer.strip().lower() == "yes" else "yes"
        prompt = (
            "Write a coherent 2-3 sentence passage, in an encyclopedic tone, that argues the "
            f"answer to the following yes/no question is clearly '{opposite}'. State the "
            f"'{opposite}' conclusion explicitly and give a plausible (even if fabricated) "
            f"justification. Do not hedge.\n\nQuestion: {question}\nPassage:"
        )
        return complete_fn(prompt).strip()

    return refute


def build_realism_pairs(n: int, gen_model: str, reliable: bool = False) -> list[dict]:
    """Build realism pairs. ``reliable`` restricts to yes/no questions with opposite-answer
    refutations, giving trustworthy conflict gold."""
    from ebrag.benchmarks.conflict_loaders import hotpot_example_to_texts
    from datasets import load_dataset

    complete = make_ollama_generate(gen_model, max_tokens=512, temperature=0.7)
    refute = make_boolean_refutation_fn(complete) if reliable else make_refutation_fn(complete)
    ds = load_dataset("hotpot_qa", "distractor", split="validation", streaming=True)

    pairs: list[dict] = []
    seen = 0
    for example in ds:
        if seen >= n:
            break
        question, answer, supporting, distractor = hotpot_example_to_texts(example)
        if not supporting:
            continue
        ans = answer if isinstance(answer, str) else answer[0]
        if reliable and ans.strip().lower() not in ("yes", "no"):
            continue  # only boolean questions get trustworthy opposite-answer gold
        seen += 1
        refutation = refute(question, ans)
        qid = str(example.get("id", f"hotpot-{seen}"))
        pairs.extend(assemble_pairs(qid, question, supporting, distractor, refutation))
    logger.info("realism_pairs_built", questions=seen, pairs=len(pairs), reliable=reliable)
    return pairs


def run(n: int = 20, judge_model: str = DEFAULT_MODEL, gen_model: str = "gemma3:27b",
        reliable: bool = False, max_workers: int = 6) -> dict:
    pairs = build_realism_pairs(n, gen_model, reliable=reliable)

    from ebrag.dialectic.conflict import ConflictDetector
    from ebrag.dialectic.llm_judge import LLMEntailmentJudge

    det = ConflictDetector()
    judge = LLMEntailmentJudge(model=judge_model, max_tokens=1024)
    judge._get_client()

    # Cross-encoder (sequential).
    ce = [det.check_pair(p["premise"], p["hypothesis"])[0] for p in pairs]

    # LLM detectors (parallel): bare + answer-relevant.
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
        out = []
        for i, p in enumerate(pairs):
            label = get_label(i)
            out.append({"category": p["category"], "gold_conflict": p["gold_conflict"],
                        "predicted_conflict": label == EntailmentLabel.CONTRADICTION})
        return out

    detectors = {
        "cross_encoder_nli": records_for(lambda i: ce[i]),
        "llm_bare": records_for(lambda i: preds[(i, "bare")]),
        "llm_answer_relevant": records_for(lambda i: preds[(i, "answer_relevant")]),
    }

    by_detector = {}
    for name, recs in detectors.items():
        m = score_detections(recs)
        m["precision_ci"] = bootstrap_ci(recs, lambda r: score_detections(r)["precision"])
        by_detector[name] = {"metrics": m, "records": recs}

    return {
        "experiment": "c4_realism",
        "reliable_conflict_gold": reliable,
        "judge_model": judge_model, "gen_model": gen_model,
        "n_pairs": len(pairs),
        "category_counts": {c: sum(1 for p in pairs if p["category"] == c)
                            for c in ("conflict", "topic_shift", "agreement")},
        "by_detector": by_detector,
    }


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    judge_model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
    gen_model = sys.argv[3] if len(sys.argv) > 3 else "gemma3:27b"
    reliable = len(sys.argv) > 4 and sys.argv[4] == "reliable"
    results = run(n=n, judge_model=judge_model, gen_model=gen_model, reliable=reliable)

    out_dir = Path("benchmarks/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / ("c4_realism_reliable.json" if reliable else "c4_realism.json")
    out_path.write_text(json.dumps(results, indent=2))

    tag = "reliable" if reliable else "standard"
    print(f"C4 realism ({tag} gold): {results['n_pairs']} pairs {results['category_counts']}, judge={judge_model}\n")
    hdr = f"{'detector':20s} | P (95% CI)        F1    per-category acc"
    print(hdr); print("-" * len(hdr))
    for name, d in results["by_detector"].items():
        m = d["metrics"]; pci = m["precision_ci"]
        pca = {c: round(a, 2) for c, a in m["per_category_accuracy"].items()}
        print(f"{name:20s} | {m['precision']:.2f} [{pci[0]:.2f},{pci[1]:.2f}]  {m['f1']:.2f}  {pca}")
    print(f"\n  -> wrote {out_path}")


if __name__ == "__main__":
    main()
