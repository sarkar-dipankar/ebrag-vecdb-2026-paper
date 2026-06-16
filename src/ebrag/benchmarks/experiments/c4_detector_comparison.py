"""
C4 hardening: cross-encoder NLI vs LLM-judge (bare and answer-relevant).

Compares three conflict detectors on the C4 probe set, with bootstrap confidence
intervals:

- ``cross_encoder_nli``    : DeBERTa-MNLI pairwise NLI (``dialectic.conflict.ConflictDetector``).
                              A pure pairwise detector -- there is no way to condition it on
                              the question, so it is stuck with the sense-trap failure.
- ``llm_bare``             : LLM judge, pairwise (``LLMEntailmentJudge.classify``).
- ``llm_answer_relevant``  : LLM judge conditioned on the question (``judge_answer_conflict``).

The C4 story: the classic cross-encoder NLI cannot escape the same-name-different-referent
false positive; only the answer-relevant LLM framing can. Reports precision/recall/F1,
per-category accuracy, and bootstrap CIs.

Run:  python -m ebrag.benchmarks.experiments.c4_detector_comparison [model]
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ebrag.benchmarks.experiments.c4_answer_relevance import build_context_probes
from ebrag.benchmarks.experiments.c4_conflict_detection import score_detections
from ebrag.benchmarks.sweeps import DEFAULT_MODEL
from ebrag.common import get_logger
from ebrag.dialectic.models import EntailmentLabel

# Model is cached; avoid the online safetensors-conversion attempt.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logger = get_logger(__name__)


def bootstrap_ci(records, fn, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI for ``fn(records) -> float`` (pure, seeded)."""
    n = len(records)
    if n == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(fn([records[i] for i in idx]))
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))


def _with_cis(records: list[dict]) -> dict:
    m = score_detections(records)
    m["precision_ci"] = bootstrap_ci(records, lambda r: score_detections(r)["precision"])
    m["f1_ci"] = bootstrap_ci(records, lambda r: score_detections(r)["f1"])
    m["sense_trap_acc_ci"] = bootstrap_ci(
        records, lambda r: score_detections(r)["per_category_accuracy"].get("sense_trap", 0.0)
    )
    return m


def _record(premise, hyp, q, gold, cat, label: EntailmentLabel, conf: float) -> dict:
    return {
        "premise": premise, "hypothesis": hyp, "question": q, "category": cat,
        "gold_conflict": gold,
        "predicted_label": label.value,
        "predicted_conflict": label == EntailmentLabel.CONTRADICTION,
        "confidence": conf,
    }


def run(model: str = DEFAULT_MODEL, max_workers: int = 6) -> dict:
    probes = build_context_probes()

    # 1) Cross-encoder NLI (sequential; one shared torch model).
    from ebrag.dialectic.conflict import ConflictDetector

    det = ConflictDetector()
    ce_records = []
    for premise, hyp, q, gold, cat in probes:
        label, score = det.check_pair(premise, hyp)
        ce_records.append(_record(premise, hyp, q, gold, cat, label, score))

    # 2) LLM detectors (parallel): bare pairwise + answer-relevant.
    from ebrag.dialectic.llm_judge import LLMEntailmentJudge

    judge = LLMEntailmentJudge(model=model, max_tokens=1024)
    judge._get_client()

    def task(args):
        i, mode = args
        premise, hyp, q, gold, cat = probes[i]
        if mode == "bare":
            j = judge.classify(premise, hyp)
        else:
            j = judge.judge_answer_conflict(q, premise, hyp)
        return (i, mode), j.label, j.confidence

    jobs = [(i, m) for i in range(len(probes)) for m in ("bare", "answer_relevant")]
    preds: dict = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for key, label, conf in ex.map(task, jobs):
            preds[key] = (label, conf)

    bare_records, ar_records = [], []
    for i, (premise, hyp, q, gold, cat) in enumerate(probes):
        lb, cb = preds[(i, "bare")]
        la, ca = preds[(i, "answer_relevant")]
        bare_records.append(_record(premise, hyp, q, gold, cat, lb, cb))
        ar_records.append(_record(premise, hyp, q, gold, cat, la, ca))

    detectors = {
        "cross_encoder_nli": ce_records,
        "llm_bare": bare_records,
        "llm_answer_relevant": ar_records,
    }
    return {
        "experiment": "c4_detector_comparison",
        "model": model,
        "n_probes": len(probes),
        "by_detector": {name: {"metrics": _with_cis(recs), "records": recs}
                        for name, recs in detectors.items()},
    }


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    results = run(model=model)

    out_dir = Path("benchmarks/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "c4_detector_comparison.json"
    out_path.write_text(json.dumps(results, indent=2))

    print(f"C4 detector comparison: {results['n_probes']} probes, LLM={model}\n")
    hdr = f"{'detector':20s} | P (95% CI)         F1     sense_trap_acc (95% CI)"
    print(hdr); print("-" * len(hdr))
    for name, d in results["by_detector"].items():
        m = d["metrics"]
        pci = m["precision_ci"]; sci = m["sense_trap_acc_ci"]
        print(
            f"{name:20s} | {m['precision']:.2f} [{pci[0]:.2f},{pci[1]:.2f}]  "
            f"{m['f1']:.2f}   {m['per_category_accuracy'].get('sense_trap', 0.0):.2f} "
            f"[{sci[0]:.2f},{sci[1]:.2f}]"
        )
    print(f"\n  -> wrote {out_path}")


if __name__ == "__main__":
    main()
