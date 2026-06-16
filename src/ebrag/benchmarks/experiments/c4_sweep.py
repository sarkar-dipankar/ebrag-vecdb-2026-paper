"""
C4 multi-model sweep: the answer-relevance effect across the model zoo.

Scales `c4_answer_relevance` to the full `DEFAULT_SWEEP` (or any model list) and runs the
judging calls concurrently (700+ calls would be too slow sequentially). For each model it
reports bare vs answer-relevant detection metrics, and aggregates the **precision lift**
and **sense-trap false-positive reduction** against model size/family to chart how the
fix's effectiveness scales.

Run:  python -m ebrag.benchmarks.experiments.c4_sweep [model ...]   # default: DEFAULT_SWEEP
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ebrag.benchmarks.experiments.c4_answer_relevance import build_context_probes
from ebrag.benchmarks.experiments.c4_conflict_detection import score_detections
from ebrag.benchmarks.experiments.c4_detector_comparison import bootstrap_ci
from ebrag.benchmarks.sweeps import DEFAULT_SWEEP, get_spec, sweep_names
from ebrag.common import get_logger
from ebrag.dialectic.llm_judge import LLMEntailmentJudge
from ebrag.dialectic.models import EntailmentLabel

logger = get_logger(__name__)


def _judge_one(judge: LLMEntailmentJudge, probe: tuple, condition: str):
    premise, hypothesis, question, _gold, _cat = probe
    if condition == "bare":
        return judge.classify(premise, hypothesis)
    return judge.judge_answer_conflict(question, premise, hypothesis)


def run_sweep(
    models: list[str] | None = None,
    probes: list[tuple] | None = None,
    max_workers: int = 8,
    max_tokens: int = 1024,
) -> dict:
    """Run bare + answer-relevant judging for every model concurrently and score.

    ``max_tokens`` is kept generous so reasoning models (which spend tokens on a hidden
    channel) still return parseable ``content``. ``max_workers`` is modest to stay under
    the backend's concurrent-request limit (429s degrade calls to neutral).
    """
    models = models or sweep_names()
    probes = probes or build_context_probes()

    # Pre-build one judge per model and force client init before threading.
    judges: dict[str, LLMEntailmentJudge] = {}
    for m in models:
        j = LLMEntailmentJudge(model=m, max_tokens=max_tokens)
        j._get_client()
        judges[m] = j

    tasks = [
        (m, i, cond)
        for m in models
        for i in range(len(probes))
        for cond in ("bare", "answer_relevant")
    ]

    def run_task(task: tuple[str, int, str]):
        m, i, cond = task
        judgement = _judge_one(judges[m], probes[i], cond)
        return task, judgement.label, judgement.confidence

    predictions: dict[tuple[str, int, str], tuple] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for task, label, conf in ex.map(run_task, tasks):
            predictions[task] = (label, conf)

    by_model: dict[str, dict] = {}
    for m in models:
        conditions: dict[str, dict] = {}
        for cond in ("bare", "answer_relevant"):
            records = []
            for i, (premise, hyp, q, gold, cat) in enumerate(probes):
                label, conf = predictions[(m, i, cond)]
                records.append({
                    "category": cat,
                    "gold_conflict": gold,
                    "predicted_conflict": label == EntailmentLabel.CONTRADICTION,
                    "predicted_label": label.value,
                    "confidence": conf,
                })
            metrics = score_detections(records)
            metrics["precision_ci"] = bootstrap_ci(
                records, lambda r: score_detections(r)["precision"]
            )
            conditions[cond] = {"metrics": metrics, "records": records}
        by_model[m] = conditions

    return {
        "experiment": "c4_sweep",
        "n_probes": len(probes),
        "models": models,
        "by_model": by_model,
        "summary": _summarize(by_model),
    }


def _summarize(by_model: dict) -> dict:
    """Aggregate the answer-relevance effect across models."""
    rows = []
    for model, conds in by_model.items():
        bare = conds["bare"]["metrics"]
        ar = conds["answer_relevant"]["metrics"]
        spec = get_spec(model)
        rows.append({
            "model": model,
            "family": spec.family if spec else None,
            "size_b": spec.size_b if spec else None,
            "bare_precision": bare["precision"],
            "bare_precision_ci": bare.get("precision_ci"),
            "ar_precision": ar["precision"],
            "ar_precision_ci": ar.get("precision_ci"),
            "precision_lift": ar["precision"] - bare["precision"],
            "bare_f1": bare["f1"],
            "ar_f1": ar["f1"],
            "bare_sense_trap_acc": bare["per_category_accuracy"].get("sense_trap"),
            "ar_sense_trap_acc": ar["per_category_accuracy"].get("sense_trap"),
            "bare_fp": bare["confusion"]["fp"],
            "ar_fp": ar["confusion"]["fp"],
        })
    n = len(rows) or 1
    return {
        "per_model": rows,
        "mean_precision_lift": sum(r["precision_lift"] for r in rows) / n,
        "mean_bare_fp": sum(r["bare_fp"] for r in rows) / n,
        "mean_ar_fp": sum(r["ar_fp"] for r in rows) / n,
        "models_improved": sum(1 for r in rows if r["precision_lift"] > 0),
        "n_models": len(rows),
    }


def main() -> None:
    models = sys.argv[1:] or sweep_names()
    results = run_sweep(models)

    out_dir = Path("benchmarks/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "c4_sweep.json"
    out_path.write_text(json.dumps(results, indent=2))

    s = results["summary"]
    print(f"C4 sweep: {results['n_probes']} probes x {s['n_models']} models\n")
    hdr = f"{'model':22s} {'fam':9s} {'size':>5s} | bareP  arP   lift | bFP arFP | stAcc b->ar"
    print(hdr)
    print("-" * len(hdr))
    for r in s["per_model"]:
        size = f"{r['size_b']:.0f}" if r["size_b"] else "-"
        bci = r.get("bare_precision_ci") or (0, 0)
        aci = r.get("ar_precision_ci") or (0, 0)
        print(
            f"{r['model']:22s} {str(r['family']):9s} {size:>5s} | "
            f"{r['bare_precision']:.2f}[{bci[0]:.2f},{bci[1]:.2f}] -> "
            f"{r['ar_precision']:.2f}[{aci[0]:.2f},{aci[1]:.2f}]  {r['precision_lift']:+.2f} | "
            f"FP {r['bare_fp']:>2d}->{r['ar_fp']:<2d}"
        )
    print(
        f"\nmean precision lift = {s['mean_precision_lift']:+.3f} | "
        f"mean FP {s['mean_bare_fp']:.1f} -> {s['mean_ar_fp']:.1f} | "
        f"improved: {s['models_improved']}/{s['n_models']}"
    )
    print(f"  -> wrote {out_path}")


if __name__ == "__main__":
    main()
