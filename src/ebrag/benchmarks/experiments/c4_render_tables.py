"""
Render the C4 result JSON artifacts into LaTeX table fragments for the paper.

Each renderer returns a complete ``\\begin{table} ... \\end{table}`` string with caption +
label, ready to ``\\input{}`` from ``main.tex``. The renderer functions are pure (take a
dict, return a str) and unit-tested; ``main()`` reads the JSON artifacts from
``benchmarks/reports/`` and writes the ``.tex`` fragments into the paper's ``results/``.

Run:  python -m ebrag.benchmarks.experiments.c4_render_tables [out_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPORTS = Path("reports")
PAPER_RESULTS = Path("paper/results")


def _tex(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def _ci(ci, fmt: str = "{:.2f}") -> str:
    if not ci:
        return ""
    lo, hi = ci
    return f"[{fmt.format(lo)},{fmt.format(hi)}]"


def render_detector_ladder(data: dict) -> str:
    """3-detector ladder (Pilot 3 / Pilot 7)."""
    n = data.get("n_probes")
    rows = []
    short = {
        "cross_encoder_nli": "cross-enc NLI",
        "llm_bare": "LLM bare",
        "llm_answer_relevant": "LLM answer-rel",
    }
    for name in ("cross_encoder_nli", "llm_bare", "llm_answer_relevant"):
        m = data["by_detector"][name]["metrics"]
        p, pci = m["precision"], m.get("precision_ci")
        f1 = m["f1"]
        st = m["per_category_accuracy"].get("sense_trap", 0.0)
        sci = m.get("sense_trap_acc_ci")
        rows.append(f"{short[name]} & {p:.2f} {_ci(pci)} & {f1:.2f} & {st:.2f} {_ci(sci)} \\\\")
    body = "\n".join(rows)
    return f"""\\begin{{table}}[t]
\\centering
\\small
\\begin{{tabular}}{{lccc}}
\\toprule
Detector & Precision (95\\% CI) & F1 & Sense-trap acc (95\\% CI) \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\caption{{Detector ladder on the curated probe set ($n={n}$). The cross-encoder is structurally
unable to be question-conditioned; only the answer-relevant LLM judge closes the sense-trap gap.}}
\\label{{tab:detector_ladder}}
\\end{{table}}
"""


def render_sweep(data: dict) -> str:
    """10-model sweep with per-model precision CIs (Pilot 2)."""
    s = data["summary"]
    n_probes = data.get("n_probes")
    rows = []
    for r in s["per_model"]:
        size = f"{int(r['size_b'])}" if r.get("size_b") else "--"
        bci = r.get("bare_precision_ci") or (0, 0)
        aci = r.get("ar_precision_ci") or (0, 0)
        lift = r["precision_lift"]
        rows.append(
            f"\\texttt{{{_tex(r['model'])}}} & {_tex(str(r['family']))} & {size} & "
            f"{r['bare_precision']:.2f} {_ci(bci)} & {r['ar_precision']:.2f} {_ci(aci)} & "
            f"{lift:+.2f} & {r['bare_fp']}$\\rightarrow${r['ar_fp']} \\\\"
        )
    body = "\n".join(rows)
    return f"""\\begin{{table*}}[t]
\\centering
\\small
\\begin{{tabular}}{{llrcccc}}
\\toprule
Model & Family & Size (B) & bare P (95\\% CI) & answer-rel P (95\\% CI) & Lift & FP b$\\rightarrow$ar \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\caption{{Answer-relevance precision lift across the 10-model sweep
({s['models_improved']}/{s['n_models']} improve; mean lift {s['mean_precision_lift']:+.3f}).
The four most capable models reach perfect precision; the two small Gemma models regress
(non-overlapping CIs). $n={n_probes}$ probes.}}
\\label{{tab:sweep}}
\\end{{table*}}
"""


def _three_row_pr(data: dict) -> list[str]:
    rows = []
    for name in ("cross_encoder_nli", "llm_bare", "llm_answer_relevant"):
        m = data["by_detector"][name]["metrics"]
        p, pci = m["precision"], m.get("precision_ci")
        r, rci = m["recall"], m.get("recall_ci")
        f1 = m["f1"]
        rows.append(
            f"{_tex(name)} & {p:.2f} {_ci(pci)} & {r:.2f} {_ci(rci)} & {f1:.2f} \\\\"
        )
    return rows


def render_answer_bearing(data: dict) -> str:
    """Pilot 6 — long answer-bearing pairs (synthetic)."""
    n = data.get("n_pairs")
    cc = data.get("category_counts", {})
    body = "\n".join(_three_row_pr(data))
    return f"""\\begin{{table}}[t]
\\centering
\\small
\\begin{{tabular}}{{lccc}}
\\toprule
Detector & Precision (95\\% CI) & Recall (95\\% CI) & F1 \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\caption{{Long answer-bearing passage pairs ($n={n}$:
{cc.get('conflict', 0)} conflict / {cc.get('agreement', 0)} agreement /
{cc.get('topic_shift', 0)} topic-shift, all 3--4 sentence LLM-written passages asserting a
specific answer). LLM judges dominate F1; the cross-encoder regains perfect precision but
length caps recall.}}
\\label{{tab:answer_bearing}}
\\end{{table}}
"""


def render_fever(data: dict) -> str:
    """Pilot 7 — real human-labelled FEVER pairs."""
    n = data.get("n_pairs")
    body = "\n".join(_three_row_pr(data))
    return f"""\\begin{{table}}[t]
\\centering
\\small
\\begin{{tabular}}{{lccc}}
\\toprule
Detector & Precision (95\\% CI) & Recall (95\\% CI) & F1 \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\caption{{External validation on real human-labelled FEVER pairs
($n={n}$; \\texttt{{copenlu/fever\\_gold\\_evidence}}). On short claim--evidence NLI, the
cross-encoder's native distribution, it is competitive; LLM judges retain perfect precision
(no false positives on real data) but are more conservative on recall.}}
\\label{{tab:fever}}
\\end{{table}}
"""


def render_prompt_robustness(data: dict) -> str:
    """Pilot 4 — small-model regression: format vs capability."""
    n = data.get("n_probes")
    rows = []
    for m, by_v in data["by_model"].items():
        for v, d in by_v.items():
            mt = d["metrics"]
            st = mt["per_category_accuracy"].get("sense_trap", 0.0)
            rows.append(
                f"\\texttt{{{_tex(m)}}} & {v} & {mt['precision']:.2f} & {mt['f1']:.2f} & {st:.2f} & {mt['confusion']['fp']} \\\\"
            )
    body = "\n".join(rows)
    return f"""\\begin{{table*}}[t]
\\centering
\\small
\\begin{{tabular}}{{llcccc}}
\\toprule
Model & Variant & Precision & F1 & Sense-trap acc & FP \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\caption{{Prompt-robustness ablation ($n={n}$). The small-Gemma regression is a
\\emph{{capability}} limit, not a format artefact: gemma3:4b emits clean parseable JSON
that verbalises the right distinction and still mislabels (manual inspection); switching to
a single-word format only shifts its threshold (FP $25\\!\\to\\!11$); few-shot
demonstration partially rescues gemma3:12b but not gemma3:4b.}}
\\label{{tab:prompt_robustness}}
\\end{{table*}}
"""


def render_ladder_per_model(data: dict) -> str:
    """Per-model detector ladder (cross-encoder vs LLM-bare vs LLM-answer-relevant)."""
    s = data["summary"]
    ce_p = s["cross_encoder_precision"]
    rows = []
    for r in data["per_model"]:
        mark = r"\checkmark" if r["ladder_holds"] else r"$\times$"
        rows.append(
            f"\\texttt{{{_tex(r['model'])}}} & "
            f"{r['cross_encoder_p']:.2f} & {r['bare_p']:.2f} & {r['ar_p']:.2f} & {mark} \\\\"
        )
    body = "\n".join(rows)
    return f"""\\begin{{table*}}[t]
\\centering
\\small
\\begin{{tabular}}{{lcccc}}
\\toprule
LLM model & cross-enc P & LLM-bare P & LLM-ar P & ladder \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\caption{{Per-model ladder against the fixed cross-encoder reference
(DeBERTa-MNLI, $P={ce_p:.2f}$ on $n=63$). Bare LLM judging beats the cross-encoder for
\\textbf{{{s['bare_beats_ce']}/{s['n_models']}}} models; answer-relevance beats it for
\\textbf{{{s['ar_beats_ce']}/{s['n_models']}}}; the full ladder
(answer-rel $\\geq$ bare $\\geq$ cross-enc) holds for {s['ladder_holds']}/{s['n_models']}.
The two small Gemma regressors drop \\emph{{below}} the cross-encoder under answer-relevance,
making them the only models where a cross-encoder would beat the LLM judge.}}
\\label{{tab:ladder_per_model}}
\\end{{table*}}
"""


def render_retrieval_realism(data: dict) -> str:
    """Vector-retrieval realism: top-k dense retrieval feeding the 3 detectors."""
    n_claims = data.get("n_claims")
    n_corpus = data.get("n_corpus_passages")
    k = data.get("k")
    r = data.get("retrieval", {})
    recall_at_k = r.get("recall_at_k", 0.0)
    recall_at_1 = r.get("recall_at_1", 0.0)
    short = {
        "cross_encoder_nli": "cross-enc NLI",
        "llm_bare": "LLM bare",
        "llm_answer_relevant": "LLM answer-rel",
    }
    rows = []
    for name in ("cross_encoder_nli", "llm_bare", "llm_answer_relevant"):
        m = data["by_detector"][name]["metrics"]
        rate, rci = m["conflict_flag_rate"], m.get("conflict_flag_rate_ci")
        pg = m["precision_on_gold"]
        lat = m["mean_latency_s"]
        rows.append(
            f"{short[name]} & {rate:.2f} {_ci(rci)} & {pg:.2f} & {lat:.2f} \\\\"
        )
    body = "\n".join(rows)
    return f"""\\begin{{table}}[t]
\\centering
\\small
\\begin{{tabular}}{{lccc}}
\\toprule
Detector & Flag rate (95\\% CI) & Prec@gold & Latency (s) \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\caption{{Vector-retrieval realism. Top-$k$ dense retrieval ($k={k}$, exact cosine over
an embedded corpus of $n={n_corpus}$ FEVER evidence sentences, all-MiniLM-L6-v2
embeddings) for $n={n_claims}$ FEVER claims; gold-evidence recall@$k$ is
{recall_at_k:.2f} (recall@1 ${recall_at_1:.2f}$). Each retrieved (claim, passage) pair is
scored by the three detectors. The cross-encoder's higher flag rate at comparable
precision-on-gold reflects the same precision/recall trade seen on synthetic and FEVER
pairs (\\S\\ref{{sec:answerbearing}}, \\S\\ref{{sec:fever}}). Per-detector mean latency
shows the cost ladder a vector-DB practitioner faces in deployment.}}
\\label{{tab:retrieval_realism}}
\\end{{table}}
"""


RENDERERS = {
    "c4_detector_comparison.json": ("c4_detector_ladder.tex", render_detector_ladder),
    "c4_sweep.json": ("c4_sweep.tex", render_sweep),
    "c4_answer_bearing.json": ("c4_answer_bearing.tex", render_answer_bearing),
    "c4_fever_validation.json": ("c4_fever.tex", render_fever),
    "c4_prompt_robustness.json": ("c4_prompt_robustness.tex", render_prompt_robustness),
    "c4_ladder_per_model.json": ("c4_ladder_per_model.tex", render_ladder_per_model),
    "c4_retrieval_realism.json": ("c4_retrieval_realism.tex", render_retrieval_realism),
}


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else PAPER_RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    for src_name, (dst_name, fn) in RENDERERS.items():
        src = REPORTS / src_name
        if not src.exists():
            print(f"  skip (missing): {src}")
            continue
        data = json.loads(src.read_text())
        (out_dir / dst_name).write_text(fn(data))
        print(f"  rendered {src.name} -> {(out_dir / dst_name).relative_to(out_dir.parent.parent)}")


if __name__ == "__main__":
    main()
