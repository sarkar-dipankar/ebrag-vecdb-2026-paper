"""
C4 core experiment: does answer-relevance framing fix pairwise-NLI false positives?

Pilot 0 (``c4_conflict_detection``) found that LLM judges are near-perfect on isolated
NLI pairs EXCEPT same-name-different-referent "sense traps", where they report a spurious
contradiction. The C4 thesis says this is because pairwise judging lacks the *question*
that says which referent is relevant. This experiment tests that directly.

For each probe we judge the same passage pair two ways and score both against the
answer-relevant gold:

- ``bare``: pairwise NLI -- "does B contradict A?" (``classify``)
- ``answer_relevant``: "do A and B give conflicting answers to the QUESTION?"
  (``judge_answer_conflict``)

Hypothesis: ``answer_relevant`` sharply reduces sense-trap false positives (higher
precision) without losing recall on genuine conflicts. Runs across a model list to test
family/size invariance. Reuses the tested scorer in ``c4_conflict_detection`` and the
expanded probe set in ``c4_probes``.

Run:  python -m ebrag.benchmarks.experiments.c4_answer_relevance [model ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ebrag.benchmarks.experiments.c4_conflict_detection import score_detections
from ebrag.benchmarks.experiments.c4_probes import ContextProbe, build_probe_set
from ebrag.benchmarks.sweeps import DEFAULT_MODEL
from ebrag.common import get_logger
from ebrag.dialectic.llm_judge import LLMEntailmentJudge
from ebrag.dialectic.models import EntailmentLabel

logger = get_logger(__name__)


def build_context_probes() -> list[ContextProbe]:
    """The expanded, deterministic answer-relevance probe set (see ``c4_probes``)."""
    return build_probe_set()


def run_for_model(model: str, probes: list[ContextProbe] | None = None) -> dict:
    """Judge each probe in both conditions and score against the answer-relevant gold."""
    probes = probes or build_context_probes()
    judge = LLMEntailmentJudge(model=model)
    bare_records: list[dict] = []
    ctx_records: list[dict] = []

    for premise, hypothesis, question, gold, category in probes:
        bare = judge.classify(premise, hypothesis)
        ctx = judge.judge_answer_conflict(question, premise, hypothesis)
        common = {"premise": premise, "hypothesis": hypothesis, "question": question,
                  "category": category, "gold_conflict": gold}
        bare_records.append({**common,
                             "predicted_label": bare.label.value,
                             "predicted_conflict": bare.label == EntailmentLabel.CONTRADICTION})
        ctx_records.append({**common,
                            "predicted_label": ctx.label.value,
                            "predicted_conflict": ctx.label == EntailmentLabel.CONTRADICTION})

    return {
        "model": model,
        "bare": {"metrics": score_detections(bare_records), "records": bare_records},
        "answer_relevant": {"metrics": score_detections(ctx_records), "records": ctx_records},
    }


def run_pilot(models: list[str] | None = None) -> dict:
    models = models or [DEFAULT_MODEL]
    return {
        "experiment": "c4_answer_relevance",
        "hypothesis": "answer-relevance framing reduces sense-trap false positives vs bare pairwise NLI",
        "by_model": {m: run_for_model(m) for m in models},
    }


def _fmt(metrics: dict) -> str:
    return (
        f"P/R/F1={metrics['precision']:.2f}/{metrics['recall']:.2f}/{metrics['f1']:.2f} "
        f"sense_trap_acc={metrics['per_category_accuracy'].get('sense_trap')} "
        f"FP={metrics['confusion']['fp']}"
    )


def main() -> None:
    models = sys.argv[1:] or [DEFAULT_MODEL]
    results = run_pilot(models)

    out_dir = Path("benchmarks/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "c4_answer_relevance.json"
    out_path.write_text(json.dumps(results, indent=2))

    for model, res in results["by_model"].items():
        print(f"\n[{model}]")
        print(f"  bare            {_fmt(res['bare']['metrics'])}")
        print(f"  answer_relevant {_fmt(res['answer_relevant']['metrics'])}")
    print(f"\n  -> wrote {out_path}")


if __name__ == "__main__":
    main()
