"""Tests for the C4 LaTeX table renderers (pure)."""

from ebrag.benchmarks.experiments.c4_render_tables import (
    _tex,
    render_answer_bearing,
    render_detector_ladder,
    render_fever,
    render_ladder_per_model,
    render_prompt_robustness,
    render_retrieval_realism,
    render_sweep,
)


def _detector_block(p: float, f1: float, st: float, ci_p=(0.5, 0.9), ci_s=(0.4, 0.95),
                    recall: float | None = None, ci_r=(0.5, 0.8)) -> dict:
    m = {
        "precision": p, "precision_ci": ci_p,
        "f1": f1, "per_category_accuracy": {"sense_trap": st},
        "sense_trap_acc_ci": ci_s,
    }
    if recall is not None:
        m["recall"] = recall
        m["recall_ci"] = ci_r
    return {"metrics": m}


class TestTexEscape:
    def test_underscore(self) -> None:
        assert _tex("cross_encoder_nli") == r"cross\_encoder\_nli"

    def test_percent_ampersand(self) -> None:
        assert _tex("50% & rising") == r"50\% \& rising"


class TestRenderDetectorLadder:
    def test_includes_three_detectors_and_caption(self) -> None:
        data = {"n_probes": 42, "by_detector": {
            "cross_encoder_nli": _detector_block(0.68, 0.81, 0.67),
            "llm_bare": _detector_block(0.90, 0.95, 0.75),
            "llm_answer_relevant": _detector_block(1.00, 1.00, 1.00),
        }}
        tex = render_detector_ladder(data)
        assert r"\begin{table}" in tex and r"\end{table}" in tex
        assert r"\label{tab:detector_ladder}" in tex
        assert "n=42" in tex
        # Display names are shortened in the rendered output (for column-width fit)
        for name in ("cross-enc NLI", "LLM bare", "LLM answer-rel"):
            assert name in tex
        assert "0.68" in tex and "1.00" in tex


class TestRenderSweep:
    def test_rows_and_summary_in_caption(self) -> None:
        data = {"n_probes": 63, "summary": {
            "per_model": [
                {"model": "gpt-oss:20b", "family": "gpt-oss", "size_b": 20,
                 "bare_precision": 0.84, "bare_precision_ci": [0.70, 0.96],
                 "ar_precision": 1.00, "ar_precision_ci": [1.00, 1.00],
                 "precision_lift": 0.16, "bare_fp": 5, "ar_fp": 0},
            ],
            "mean_precision_lift": 0.011, "models_improved": 8, "n_models": 10,
            "mean_bare_fp": 4.9, "mean_ar_fp": 5.8,
        }}
        tex = render_sweep(data)
        assert r"\label{tab:sweep}" in tex
        assert "gpt-oss:20b" in tex
        assert "8/10 improve" in tex
        assert "+0.16" in tex  # precision lift formatting


class TestRenderAnswerBearingAndFever:
    def _ab_data(self) -> dict:
        return {"n_pairs": 42, "category_counts": {"conflict": 14, "agreement": 14, "topic_shift": 14},
                "by_detector": {
                    "cross_encoder_nli": _detector_block(1.0, 0.67, 0.5, recall=0.5),
                    "llm_bare": _detector_block(0.92, 0.89, 0.86, recall=0.86),
                    "llm_answer_relevant": _detector_block(1.0, 0.88, 0.79, recall=0.79),
                }}

    def test_answer_bearing_includes_pr_and_n(self) -> None:
        tex = render_answer_bearing(self._ab_data())
        assert r"\label{tab:answer_bearing}" in tex
        assert "n=42" in tex
        assert "Recall" in tex and "Precision" in tex

    def test_fever(self) -> None:
        data = self._ab_data()
        data["n_pairs"] = 100
        tex = render_fever(data)
        assert r"\label{tab:fever}" in tex
        assert "n=100" in tex
        assert "fever" in tex.lower()


class TestRenderLadderPerModel:
    def test_rows_and_counts_in_caption(self) -> None:
        data = {"summary": {"n_models": 2, "bare_beats_ce": 2, "ar_beats_ce": 1,
                            "ladder_holds": 1, "cross_encoder_precision": 0.68,
                            "cross_encoder_f1": 0.81},
                "per_model": [
                    {"model": "gpt-oss:120b", "cross_encoder_p": 0.68,
                     "bare_p": 0.90, "ar_p": 1.00, "ladder_holds": True},
                    {"model": "gemma3:4b", "cross_encoder_p": 0.68,
                     "bare_p": 0.96, "ar_p": 0.51, "ladder_holds": False},
                ]}
        tex = render_ladder_per_model(data)
        assert r"\label{tab:ladder_per_model}" in tex
        assert "gpt-oss:120b" in tex and "gemma3:4b" in tex
        assert "2/2" in tex and "1/2" in tex  # counts in caption
        assert r"\checkmark" in tex


class TestRenderPromptRobustness:
    def test_rows_per_model_variant(self) -> None:
        data = {"n_probes": 63, "by_model": {
            "gemma3:4b": {
                "json": {"metrics": {"precision": 0.51, "f1": 0.68,
                                     "per_category_accuracy": {"sense_trap": 0.0},
                                     "confusion": {"fp": 25}}},
                "plain": {"metrics": {"precision": 0.67, "f1": 0.75,
                                      "per_category_accuracy": {"sense_trap": 0.58},
                                      "confusion": {"fp": 11}}},
            }
        }}
        tex = render_prompt_robustness(data)
        assert "gemma3:4b" in tex
        assert "json" in tex and "plain" in tex
        assert "0.51" in tex and "0.67" in tex
        assert r"\label{tab:prompt_robustness}" in tex


class TestRenderRetrievalRealism:
    def test_renders_all_three_detectors_and_caption(self) -> None:
        data = {
            "n_claims": 100, "n_corpus_passages": 100, "k": 5,
            "retrieval": {"recall_at_k": 0.96, "recall_at_1": 0.84,
                          "model": "sentence-transformers/all-MiniLM-L6-v2",
                          "index": "exact cosine over L2-normalised embeddings"},
            "by_detector": {
                "cross_encoder_nli": {"metrics": {
                    "conflict_flag_rate": 0.34, "conflict_flag_rate_ci": (0.30, 0.38),
                    "precision_on_gold": 0.86, "mean_latency_s": 0.18,
                    "n_pairs": 500, "n_gold_pairs": 100}},
                "llm_bare": {"metrics": {
                    "conflict_flag_rate": 0.10, "conflict_flag_rate_ci": (0.07, 0.13),
                    "precision_on_gold": 0.90, "mean_latency_s": 2.86,
                    "n_pairs": 500, "n_gold_pairs": 100}},
                "llm_answer_relevant": {"metrics": {
                    "conflict_flag_rate": 0.09, "conflict_flag_rate_ci": (0.06, 0.12),
                    "precision_on_gold": 0.93, "mean_latency_s": 3.83,
                    "n_pairs": 500, "n_gold_pairs": 100}},
            },
        }
        tex = render_retrieval_realism(data)
        assert r"\begin{table}" in tex and r"\end{table}" in tex
        assert r"\label{tab:retrieval_realism}" in tex
        for name in ("cross-enc NLI", "LLM bare", "LLM answer-rel"):
            assert name in tex
        # Flag-rate row values render with CI.
        assert "0.34" in tex and "0.10" in tex
        assert "0.96" in tex  # recall@k in caption
        assert "$k=5$" in tex
