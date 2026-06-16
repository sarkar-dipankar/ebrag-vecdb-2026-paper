"""
C4 external validation on real human-labelled FEVER (claim, evidence) pairs.

Pilot 6 showed the answer-bearing finding on LLM-generated long passages. This run repeats
the 3-detector comparison on a real human-labelled corpus:
``copenlu/fever_gold_evidence`` --- VERIFIABLE claims with bundled evidence sentences and
SUPPORTS / REFUTES gold labels. Premise = evidence sentence(s), hypothesis = claim. The
answer-relevant judge is framed around the claim ("Is this claim true: <claim>?"), with
passage A = evidence, passage B = claim.

Maps:
- REFUTES  -> gold_conflict = True  (evidence contradicts claim)
- SUPPORTS -> gold_conflict = False (evidence supports claim)

Reports precision / recall / F1 with bootstrap CIs; expected to match Pilot 6's ordering
if the synthetic answer-bearing finding generalises to real data.

Run:  python -m ebrag.benchmarks.experiments.c4_fever_validation [n_per_class] [judge_model]
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ebrag.benchmarks.conflict_loaders import load_fever_evidence_pairs
from ebrag.benchmarks.experiments.c4_conflict_detection import score_detections
from ebrag.benchmarks.experiments.c4_detector_comparison import bootstrap_ci
from ebrag.benchmarks.sweeps import DEFAULT_MODEL
from ebrag.common import get_logger
from ebrag.dialectic.models import EntailmentLabel

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logger = get_logger(__name__)


def run(n_per_class: int = 50, judge_model: str = DEFAULT_MODEL,
        max_workers: int = 6) -> dict:
    # Loader needs to reach the (cached) dataset; allow hub for this step only.
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    pairs = load_fever_evidence_pairs(n_per_class=n_per_class)
    # Restore offline for the cached cross-encoder model load.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from ebrag.dialectic.conflict import ConflictDetector
    from ebrag.dialectic.llm_judge import LLMEntailmentJudge

    det = ConflictDetector()
    judge = LLMEntailmentJudge(model=judge_model, max_tokens=1024)
    judge._get_client()

    ce = [det.check_pair(p["evidence"], p["claim"])[0] for p in pairs]

    def task(args):
        i, mode = args
        p = pairs[i]
        if mode == "bare":
            j = judge.classify(p["evidence"], p["claim"])
        else:
            q = f"Is this claim true: \"{p['claim']}\"?"
            j = judge.judge_answer_conflict(q, p["evidence"], p["claim"])
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
        "experiment": "c4_fever_validation",
        "judge_model": judge_model,
        "n_pairs": len(pairs),
        "category_counts": {c: sum(1 for p in pairs if p["category"] == c)
                            for c in ("refutes", "supports")},
        "by_detector": by_detector,
    }


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    judge_model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
    results = run(n_per_class=n, judge_model=judge_model)

    out_dir = Path("benchmarks/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "c4_fever_validation.json"
    out_path.write_text(json.dumps(results, indent=2))

    print(f"C4 FEVER validation: {results['n_pairs']} pairs {results['category_counts']}, judge={judge_model}\n")
    hdr = f"{'detector':20s} | P (CI)            R (CI)            F1"
    print(hdr); print("-" * len(hdr))
    for name, d in results["by_detector"].items():
        m = d["metrics"]; pci = m["precision_ci"]; rci = m["recall_ci"]
        print(f"{name:20s} | {m['precision']:.2f}[{pci[0]:.2f},{pci[1]:.2f}]  "
              f"{m['recall']:.2f}[{rci[0]:.2f},{rci[1]:.2f}]  {m['f1']:.2f}")
    print(f"\n  -> wrote {out_path}")


if __name__ == "__main__":
    main()
