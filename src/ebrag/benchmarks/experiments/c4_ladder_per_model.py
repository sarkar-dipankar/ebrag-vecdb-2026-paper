"""
Per-model detector ladder: does answer-relevance beat the cross-encoder model-by-model?

The cross-encoder NLI is model-fixed (DeBERTa-MNLI); its number is constant across the LLM
sweep. So this is a pure combination of two existing artefacts:

- ``c4_sweep.json``                  (per-LLM bare + answer-relevant precision/F1 at n=63)
- ``c4_detector_comparison.json``    (cross-encoder reference at n=63)

For each LLM model we line up (cross-encoder, bare, answer-relevant) precision/F1 and check
whether the ladder ``answer_relevant >= bare >= cross_encoder`` holds. The expected
nuance: the LLM-bare beats the cross-encoder for capable models on this probe set, but
answer-relevance drops below the cross-encoder for the small-Gemma regressors --- the same
capacity caveat we saw in Pilot 2, now visible against the cross-encoder reference.

Pure (no LLM/network); operates on cached JSON.

Run:  python -m ebrag.benchmarks.experiments.c4_ladder_per_model
"""

from __future__ import annotations

import json
from pathlib import Path

REPORTS = Path("benchmarks/reports")


def analyze(sweep: dict, ladder: dict) -> dict:
    """Combine the LLM sweep with the fixed cross-encoder reference into a per-model view."""
    ce_metrics = ladder["by_detector"]["cross_encoder_nli"]["metrics"]
    ce_p = ce_metrics["precision"]
    ce_f1 = ce_metrics["f1"]

    rows: list[dict] = []
    for model_name, model_block in sweep["by_model"].items():
        bare = model_block["bare"]["metrics"]
        ar = model_block["answer_relevant"]["metrics"]
        rows.append({
            "model": model_name,
            "cross_encoder_p": ce_p,
            "cross_encoder_f1": ce_f1,
            "bare_p": bare["precision"],
            "bare_f1": bare["f1"],
            "ar_p": ar["precision"],
            "ar_f1": ar["f1"],
            "bare_beats_ce": bare["precision"] > ce_p,
            "ar_beats_ce": ar["precision"] > ce_p,
            "ladder_holds": ar["precision"] >= bare["precision"] >= ce_p,
        })

    summary = {
        "n_models": len(rows),
        "bare_beats_ce": sum(1 for r in rows if r["bare_beats_ce"]),
        "ar_beats_ce": sum(1 for r in rows if r["ar_beats_ce"]),
        "ladder_holds": sum(1 for r in rows if r["ladder_holds"]),
        "cross_encoder_precision": ce_p,
        "cross_encoder_f1": ce_f1,
    }
    return {"experiment": "c4_ladder_per_model", "per_model": rows, "summary": summary}


def main() -> None:
    sweep = json.loads((REPORTS / "c4_sweep.json").read_text())
    ladder = json.loads((REPORTS / "c4_detector_comparison.json").read_text())
    out = analyze(sweep, ladder)
    out_path = REPORTS / "c4_ladder_per_model.json"
    out_path.write_text(json.dumps(out, indent=2))

    s = out["summary"]
    print(f"Per-model ladder (cross-encoder ref: P={s['cross_encoder_precision']:.2f}, F1={s['cross_encoder_f1']:.2f})")
    print(f"  bare > cross-encoder:  {s['bare_beats_ce']}/{s['n_models']}")
    print(f"  ar   > cross-encoder:  {s['ar_beats_ce']}/{s['n_models']}")
    print(f"  full ladder holds:     {s['ladder_holds']}/{s['n_models']}")
    print(f"\n  {'model':22s}  CE P   bare P   ar P    ladder?")
    print(f"  {'-'*22}  -----  ------  ------  -------")
    for r in out["per_model"]:
        mark = "OK" if r["ladder_holds"] else "no"
        print(f"  {r['model']:22s}  {r['cross_encoder_p']:.2f}   {r['bare_p']:.2f}    {r['ar_p']:.2f}    {mark}")
    print(f"\n  -> wrote {out_path}")


if __name__ == "__main__":
    main()
