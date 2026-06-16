"""
C4 pilot: does the LLM judge detect *answer-relevant* conflict?

Runs :class:`ebrag.dialectic.llm_judge.LLMEntailmentJudge` over a curated probe set of
passage pairs with gold conflict labels, spanning the failure modes the C4 thesis
predicts:

- explicit contradictions (should be caught),
- agreement / paraphrase (should NOT be flagged),
- topic shift (on-topic-ish but different entity -> false-positive trap),
- numeric / temporal / implicit conflicts (subtle -> false-negative trap).

Reports contradiction-detection precision/recall/F1, per-category accuracy, the
topic-shift false-positive rate, and the subtle-conflict false-negative rate. The scorer
is pure and unit-tested; the judging step calls the configured model (Ollama Cloud).

Run:  python -m ebrag.benchmarks.experiments.c4_conflict_detection [model ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ebrag.benchmarks.sweeps import DEFAULT_MODEL
from ebrag.common import get_logger
from ebrag.dialectic.llm_judge import LLMEntailmentJudge
from ebrag.dialectic.models import EntailmentLabel

logger = get_logger(__name__)


# (premise, hypothesis, gold_is_conflict, category)
ProbePair = tuple[str, str, bool, str]

#: Categories whose gold label is "conflict" but that are subtle (FN trap).
SUBTLE_CONFLICT_CATEGORIES = frozenset({"numeric", "temporal", "implicit"})


def build_probe_pairs() -> list[ProbePair]:
    """Curated NLI probe pairs with gold answer-relevant-conflict labels."""
    return [
        # --- explicit contradictions (gold: conflict) ---
        ("The capital of Australia is Canberra.",
         "The capital of Australia is Sydney.", True, "explicit"),
        ("Mount Everest is the tallest mountain on Earth.",
         "K2 is the tallest mountain on Earth.", True, "explicit"),
        ("Albert Einstein developed the theory of general relativity.",
         "Isaac Newton developed the theory of general relativity.", True, "explicit"),
        # --- numeric / temporal / implicit conflicts (gold: conflict, subtle) ---
        ("Water boils at 100 degrees Celsius at sea level.",
         "Water boils at 90 degrees Celsius at sea level.", True, "numeric"),
        ("The Eiffel Tower is 330 metres tall.",
         "The Eiffel Tower stands 300 metres tall.", True, "numeric"),
        ("The treaty was signed in 1919.",
         "The treaty was signed in 1920.", True, "temporal"),
        ("The company reported a profit in the most recent quarter.",
         "The company posted a loss in the most recent quarter.", True, "implicit"),
        # --- agreement / paraphrase (gold: no conflict) ---
        ("Albert Einstein developed general relativity.",
         "General relativity was formulated by Einstein.", False, "agreement"),
        ("Gold's chemical symbol is Au.",
         "On the periodic table, gold is denoted Au.", False, "agreement"),
        ("The Pacific is the largest ocean on Earth.",
         "No ocean is larger than the Pacific.", False, "agreement"),
        # --- topic shift (gold: no conflict; false-positive trap) ---
        ("The capital of Australia is Canberra.",
         "Sydney is the most populous city in Australia.", False, "topic_shift"),
        ("Mitochondria produce ATP in cells.",
         "Chloroplasts are found in plant cells.", False, "topic_shift"),
        ("Photosynthesis occurs in plants.",
         "The French Revolution began in 1789.", False, "topic_shift"),
        # --- hard no-conflict traps (gold: no conflict; over-eager detectors flag these) ---
        ("Paris is the capital of France.",
         "Paris is a small town in Texas, USA.", False, "sense_trap"),
        ("A marathon is 42.195 kilometres long.",
         "A marathon is about 26.2 miles long.", False, "unit_agreement"),
        ("It is not the case that the Earth is flat.",
         "The Earth is round.", False, "negation_agreement"),
        ("The novel was published in the 1920s.",
         "The novel was published in 1925.", False, "scope_agreement"),
    ]


def score_detections(records: list[dict]) -> dict:
    """Score contradiction-detection quality against gold (pure; no I/O).

    Each record needs ``gold_conflict`` (bool), ``predicted_conflict`` (bool), and
    ``category`` (str).
    """
    tp = fp = fn = tn = 0
    by_cat: dict[str, dict[str, int]] = {}
    for r in records:
        gold = bool(r["gold_conflict"])
        pred = bool(r["predicted_conflict"])
        cat = r["category"]
        cat_stats = by_cat.setdefault(cat, {"n": 0, "correct": 0})
        cat_stats["n"] += 1
        cat_stats["correct"] += int(gold == pred)
        if gold and pred:
            tp += 1
        elif not gold and pred:
            fp += 1
        elif gold and not pred:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # Topic-shift false-positive rate: gold no-conflict topic_shift flagged as conflict.
    ts = [r for r in records if r["category"] == "topic_shift"]
    ts_fp = sum(1 for r in ts if r["predicted_conflict"]) / len(ts) if ts else None

    # Subtle-conflict false-negative rate: gold conflict in subtle categories missed.
    subtle = [r for r in records if r["category"] in SUBTLE_CONFLICT_CATEGORIES]
    subtle_fn = (
        sum(1 for r in subtle if not r["predicted_conflict"]) / len(subtle)
        if subtle
        else None
    )

    return {
        "n": len(records),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "per_category_accuracy": {
            c: s["correct"] / s["n"] for c, s in sorted(by_cat.items())
        },
        "topic_shift_false_positive_rate": ts_fp,
        "subtle_conflict_false_negative_rate": subtle_fn,
    }


def run_for_model(model: str, pairs: list[ProbePair] | None = None) -> dict:
    """Run the LLM judge over the probe pairs for one model and score the result."""
    pairs = pairs or build_probe_pairs()
    judge = LLMEntailmentJudge(model=model)
    records: list[dict] = []
    for premise, hypothesis, gold_conflict, category in pairs:
        j = judge.classify(premise, hypothesis)
        predicted_conflict = j.label == EntailmentLabel.CONTRADICTION
        records.append(
            {
                "premise": premise,
                "hypothesis": hypothesis,
                "category": category,
                "gold_conflict": gold_conflict,
                "predicted_label": j.label.value,
                "predicted_conflict": predicted_conflict,
                "confidence": j.confidence,
            }
        )
    return {"model": model, "metrics": score_detections(records), "records": records}


def run_pilot(models: list[str] | None = None) -> dict:
    models = models or [DEFAULT_MODEL]
    return {
        "experiment": "c4_conflict_detection_pilot",
        "by_model": {m: run_for_model(m) for m in models},
    }


def main() -> None:
    models = sys.argv[1:] or [DEFAULT_MODEL]
    results = run_pilot(models)

    out_dir = Path("benchmarks/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "c4_pilot.json"
    out_path.write_text(json.dumps(results, indent=2))

    for model, res in results["by_model"].items():
        m = res["metrics"]
        print(f"\n[{model}]  n={m['n']}")
        print(f"  contradiction P/R/F1 = {m['precision']:.2f}/{m['recall']:.2f}/{m['f1']:.2f}")
        print(f"  topic-shift FP rate      = {m['topic_shift_false_positive_rate']}")
        print(f"  subtle-conflict FN rate  = {m['subtle_conflict_false_negative_rate']}")
        print(f"  per-category accuracy    = {m['per_category_accuracy']}")
    print(f"\n  -> wrote {out_path}")


if __name__ == "__main__":
    main()
