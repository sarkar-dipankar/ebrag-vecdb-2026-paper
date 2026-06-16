"""
C4 ablation: is the small-model answer-relevance regression a FORMAT or a CONTENT failure?

Pilot 2 found gemma3:4b/12b regress under the answer-relevance prompt (over-flagging
conflict). This isolates the cause by judging the same probes under three prompt variants
and capturing the raw model output:

- ``json``    : the production prompt (JSON {"label": "conflict|no_conflict"}).
- ``plain``   : ask for a single word -- removes the JSON-formatting burden.
- ``fewshot`` : JSON prompt + 2 worked examples -- gives the instruction by demonstration.

If a small model recovers under ``plain`` it was a format-following failure; if it recovers
under ``fewshot`` it was instruction complexity; if neither helps it is a genuine capability
limit. Raw outputs on the sense-trap false positives are saved for inspection.

Run:  python -m ebrag.benchmarks.experiments.c4_prompt_robustness
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ebrag.benchmarks.experiments.c4_conflict_detection import score_detections
from ebrag.benchmarks.experiments.c4_probes import build_probe_set
from ebrag.benchmarks.experiments.llm_helpers import make_ollama_generate
from ebrag.common import get_logger
from ebrag.dialectic.llm_judge import _ANSWER_CONFLICT_PROMPT, parse_answer_conflict
from ebrag.dialectic.models import EntailmentLabel

logger = get_logger(__name__)

DEFAULT_ABLATION_MODELS = ["gemma3:4b", "gemma3:12b", "gpt-oss:20b"]


def _prompt_json(q: str, a: str, b: str) -> str:
    return _ANSWER_CONFLICT_PROMPT.format(question=q, passage_a=a, passage_b=b)


def _prompt_plain(q: str, a: str, b: str) -> str:
    return (
        "Do passage A and passage B give CONFLICTING ANSWERS to the question? If a passage "
        "is about a different thing or does not answer the question, it is NOT a conflict.\n"
        "Reply with ONLY one word: conflict OR no_conflict.\n\n"
        f"QUESTION: {q}\nA: {a}\nB: {b}\nAnswer:"
    )


def _prompt_fewshot(q: str, a: str, b: str) -> str:
    examples = (
        'Example 1 -> QUESTION: What is the capital of France? '
        'A: "Paris is the capital of France." B: "Paris is a town in Texas." '
        'Answer: {"label": "no_conflict"}  (B is a different Paris)\n'
        'Example 2 -> QUESTION: What is the capital of Australia? '
        'A: "The capital of Australia is Canberra." B: "The capital of Australia is Sydney." '
        'Answer: {"label": "conflict"}\n\n'
    )
    return examples + _ANSWER_CONFLICT_PROMPT.format(question=q, passage_a=a, passage_b=b)


PROMPT_VARIANTS = {"json": _prompt_json, "plain": _prompt_plain, "fewshot": _prompt_fewshot}


def run(models: list[str] | None = None, variants: list[str] | None = None,
        max_workers: int = 8) -> dict:
    models = models or DEFAULT_ABLATION_MODELS
    variants = variants or list(PROMPT_VARIANTS)
    probes = build_probe_set()

    fns = {m: make_ollama_generate(m, max_tokens=512, temperature=0.0) for m in models}

    tasks = [(m, v, i) for m in models for v in variants for i in range(len(probes))]

    def run_task(task):
        m, v, i = task
        premise, hyp, q, _gold, _cat = probes[i]
        raw = fns[m](PROMPT_VARIANTS[v](q, premise, hyp))
        return task, raw, parse_answer_conflict(raw).label

    raw_by_task: dict = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for task, raw, label in ex.map(run_task, tasks):
            raw_by_task[task] = (raw, label)

    by_model: dict = {}
    for m in models:
        by_variant: dict = {}
        for v in variants:
            records, fp_samples = [], []
            for i, (premise, hyp, q, gold, cat) in enumerate(probes):
                raw, label = raw_by_task[(m, v, i)]
                pred_conflict = label == EntailmentLabel.CONTRADICTION
                records.append({"category": cat, "gold_conflict": gold,
                                "predicted_conflict": pred_conflict})
                if cat == "sense_trap" and pred_conflict and len(fp_samples) < 3:
                    fp_samples.append({"premise": premise, "hypothesis": hyp,
                                       "raw": raw[:200]})
            by_variant[v] = {"metrics": score_detections(records), "sense_trap_fp_samples": fp_samples}
        by_model[m] = by_variant

    return {"experiment": "c4_prompt_robustness", "models": models, "variants": variants,
            "n_probes": len(probes), "by_model": by_model}


def main() -> None:
    models = sys.argv[1:] or DEFAULT_ABLATION_MODELS
    results = run(models)

    out_dir = Path("benchmarks/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "c4_prompt_robustness.json"
    out_path.write_text(json.dumps(results, indent=2))

    print(f"C4 prompt robustness: {results['n_probes']} probes\n")
    hdr = f"{'model':16s} {'variant':9s} | precision  F1    sense_trap_acc  FP"
    print(hdr); print("-" * len(hdr))
    for m, bv in results["by_model"].items():
        for v, d in bv.items():
            mt = d["metrics"]
            print(f"{m:16s} {v:9s} | {mt['precision']:.2f}      {mt['f1']:.2f}  "
                  f"{mt['per_category_accuracy'].get('sense_trap', 0.0):.2f}          {mt['confusion']['fp']}")
    print(f"\n  -> wrote {out_path}")


if __name__ == "__main__":
    main()
