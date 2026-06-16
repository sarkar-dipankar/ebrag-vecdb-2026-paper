# An Input-Regime Audit of Conflict Detection for Retrieval-Augmented Generation

Reproducibility artefacts for the paper submitted to **VecDB @ VLDB 2026** (The 2nd
Workshop on Vector Databases).

- **Paper:** [paper/main.pdf](paper/main.pdf)
- **Author:** Dipankar Sarkar, Skelf Research — `dipankar@skelfresearch.com`
- **Venue:** [VecDB@VLDB2026](https://vecdb-ws.github.io/vldb2026/)

## What this repository contains

```
.
├── paper/
│   ├── main.pdf            compiled paper (6 pp, acmart sigconf, VLDB workshop block)
│   ├── main.tex            paper source
│   ├── references.bib      bibliography
│   └── results/            rendered LaTeX tables (auto-generated from /reports)
├── reports/                JSON artefacts, one per experiment
│   ├── c4_detector_comparison.json    detector ladder on curated probes
│   ├── c4_sweep.json                  10-model bare-vs-answer-relevant sweep
│   ├── c4_ladder_per_model.json       per-model cross-encoder/LLM comparison
│   ├── c4_prompt_robustness.json      JSON / plain / few-shot prompt ablation
│   ├── c4_answer_bearing.json         long answer-bearing passage pairs
│   ├── c4_fever_validation.json       external validation on 100 FEVER pairs
│   ├── c4_realism_reliable.json       HotpotQA long multi-claim regime
│   └── c4_retrieval_realism.json      top-k dense retrieval realism (§11)
├── src/ebrag/              minimal runnable Python slice
│   ├── common/             logging + config
│   ├── chat/llm.py         OpenAI-compatible client (used against Ollama Cloud)
│   ├── dialectic/          conflict detector + LLM-as-judge entailment
│   ├── retrieval/dense.py  USearch dense index (small-corpus path uses exact cosine)
│   └── benchmarks/
│       ├── conflict_loaders.py    FEVER, HotpotQA-conflict loaders
│       ├── sweeps.py              the 10-model DEFAULT_SWEEP definition
│       └── experiments/c4_*.py    the 9 experiment scripts + table renderer
├── tests/                  unit tests for the table renderers
├── requirements.txt        runtime dependencies
└── LICENSE                 MIT
```

## The five experiments behind the paper

| Section | Script | JSON artefact | Rendered table |
|---|---|---|---|
| §5 Sense trap & answer-relevance | `c4_detector_comparison.py` | `c4_detector_comparison.json` | `c4_detector_ladder.tex` |
| §6 Cross-model scaling (10 models) | `c4_sweep.py` | `c4_sweep.json` | `c4_sweep.tex` |
| §6 Per-model ladder | `c4_ladder_per_model.py` | `c4_ladder_per_model.json` | `c4_ladder_per_model.tex` |
| §7 Capability not format | `c4_prompt_robustness.py` | `c4_prompt_robustness.json` | `c4_prompt_robustness.tex` |
| §8 Realism (HotpotQA long fragments) | `c4_realism.py` | `c4_realism_reliable.json` | (inline in body) |
| §9 Long answer-bearing pairs | `c4_answer_bearing.py` | `c4_answer_bearing.json` | `c4_answer_bearing.tex` |
| §10 FEVER external validation | `c4_fever_validation.py` | `c4_fever_validation.json` | `c4_fever.tex` |
| §11 Vector-retrieval realism | `c4_retrieval_realism.py` | `c4_retrieval_realism.json` | `c4_retrieval_realism.tex` |

## Reproducing the results

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ./src
```

### 2. Configure the LLM backend

The experiments call an OpenAI-compatible chat endpoint. We used Ollama Cloud during
the paper run. Set:

```bash
export OPENAI_API_KEY=...           # your Ollama Cloud key
export OPENAI_BASE_URL=https://ollama.com/v1
```

The default judge model is `gpt-oss:20b` (see `src/ebrag/benchmarks/sweeps.py`).
Reasoning models (gpt-oss family) need a generous `max_tokens` so the hidden reasoning
channel does not eat the visible content; the scripts default to 1024.

### 3. Run an experiment

```bash
# §10 external validation (~5 min, 100 FEVER pairs):
python -m ebrag.benchmarks.experiments.c4_fever_validation 50 gpt-oss:20b

# §11 vector-retrieval realism (~10 min, 100 claims × top-5 retrieved):
python -m ebrag.benchmarks.experiments.c4_retrieval_realism 50 gpt-oss:20b

# §6 cross-model sweep (long; hits multiple model families):
python -m ebrag.benchmarks.experiments.c4_sweep 63
```

Each script writes a JSON artefact to `benchmarks/reports/`.

### 4. Re-render the LaTeX tables

```bash
python -m ebrag.benchmarks.experiments.c4_render_tables paper/results/
```

This regenerates the `c4_*.tex` files under `paper/results/` from the JSON artefacts.
The paper's `main.tex` `\input`s them directly.

### 5. Rebuild the PDF

```bash
cd paper && latexmk -pdf main.tex
```

(uses the acmart sigconf class with VLDB workshop metadata; see `main.tex` preamble.)

## Verifying the headline numbers without running anything

Every reported number is in `reports/*.json`. For example, the **§11 retrieval realism**
finding (cross-encoder over-fires on retrieved pairs while LLM judges stay conservative):

```bash
python - <<'PY'
import json
d = json.load(open("reports/c4_retrieval_realism.json"))
print(f"corpus={d['n_corpus_passages']}, k={d['k']}, recall@k={d['retrieval']['recall_at_k']:.2f}")
for name, b in d["by_detector"].items():
    m = b["metrics"]
    print(f"  {name:22s} flag={m['conflict_flag_rate']:.2f} "
          f"prec@gold={m['precision_on_gold']:.2f} latency={m['mean_latency_s']:.2f}s")
PY
```

Expected output:
```
corpus=100, k=5, recall@k=0.87
  cross_encoder_nli      flag=0.43 prec@gold=0.85 latency=0.16s
  llm_bare               flag=0.09 prec@gold=0.93 latency=4.05s
  llm_answer_relevant    flag=0.08 prec@gold=0.89 latency=5.21s
```

## Tests

The LaTeX table renderers are pure (dict in, str out) and unit-tested:

```bash
pytest tests/test_c4_render_tables.py -q
```

(9 tests, runs in ~1s.)

## Citation

If you use these artefacts, please cite the paper:

```bibtex
@inproceedings{sarkar2026inputregime,
  title     = {An Input-Regime Audit of Conflict Detection for Retrieval-Augmented Generation},
  author    = {Sarkar, Dipankar},
  booktitle = {VLDB 2026 Workshop: The 2nd Workshop on Vector Databases (VecDB)},
  year      = {2026}
}
```

## License

MIT. See [LICENSE](LICENSE).
