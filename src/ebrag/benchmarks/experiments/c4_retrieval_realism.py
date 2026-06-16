"""
C4 vector-retrieval realism check.

The other C4 experiments evaluate the three detectors on curated or pre-paired data. This
one closes the deployment loop: build a small dense index over FEVER evidence sentences,
retrieve top-k passages for each claim from the index (exact cosine on an L2-normalised
embedding matrix --- the natural reference for the small index sizes used here), and run
all three detectors on the (claim, retrieved-passage) pairs that the vector DB actually
returns.

Pipeline (matches the practitioner-facing diagram in the paper):

    FEVER claim  --(USearch dense top-k)-->  k retrieved evidence sentences
                                                    |
              for each retrieved passage  ---> detector(claim, passage)

Detectors compared (same three as the rest of C4):
- cross_encoder_nli (DeBERTa-base-MNLI)
- llm_bare (LLM-as-judge, pairwise)
- llm_answer_relevant (LLM-as-judge, question-conditioned on the claim)

Reported metrics per detector:
- conflict_flag_rate: fraction of retrieved pairs flagged conflict
- precision_on_gold: when the gold evidence is among the retrieved k, fraction of those
  gold pairs whose detector verdict matches the FEVER label (REFUTES -> conflict;
  SUPPORTS -> no conflict)
- mean_latency_s: wall-clock per detector call (relevant to the cost/latency choice the
  paper's pipeline framing calls out)

This is a small, focused experiment. It does not retrain anything; it adds a real
retrieval step in front of the same three detectors evaluated everywhere else in C4.

Run:  python -m ebrag.benchmarks.experiments.c4_retrieval_realism [n_claims] [judge_model]
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ebrag.benchmarks.conflict_loaders import load_fever_evidence_pairs
from ebrag.benchmarks.experiments.c4_detector_comparison import bootstrap_ci
from ebrag.benchmarks.sweeps import DEFAULT_MODEL
from ebrag.common import get_logger
from ebrag.dialectic.models import EntailmentLabel

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logger = get_logger(__name__)


def _embed(texts: list[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    """Embed a list of strings into a (n, d) float32 matrix.

    Uses the default sentence-transformers model that HybridRetriever uses.
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.astype(np.float32)


def _build_index(passages: list[str], embeddings: np.ndarray) -> dict:
    """Build a small-corpus dense index.

    The C4 corpus is on the order of 100 passages, so we keep an L2-normalised matrix
    and do exact cosine search at query time. This is the natural reference for a
    vector-DB top-k call on a small index, gives deterministic top-k, and avoids the
    HNSW approximation entirely (which is also more honest for a paper experiment).
    """
    # L2-normalise rows so cosine becomes a dot product.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normed = embeddings / norms
    return {"passages": passages, "vectors": normed}


def _retrieve(idx: dict, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Top-k cosine search against the corpus matrix."""
    q = query_vec.astype(np.float32)
    qn = q / max(float(np.linalg.norm(q)), 1e-12)
    sims = idx["vectors"] @ qn  # (n_passages,) cosine similarities
    # argpartition is O(n); we only need the top-k.
    top = np.argpartition(-sims, kth=min(k, len(sims) - 1))[:k]
    top = top[np.argsort(-sims[top])]
    return [(int(i), float(sims[i])) for i in top]


def run(n_per_class: int = 50, judge_model: str = DEFAULT_MODEL,
        k: int = 5, max_workers: int = 6) -> dict:
    """Load FEVER, build the dense index, retrieve top-k, run 3 detectors on retrieved pairs."""
    # Loader needs the hub for the dataset; restore offline for the cached cross-encoder.
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    pairs = load_fever_evidence_pairs(n_per_class=n_per_class)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # Build the corpus from all distinct evidence sentences; remember which pair each
    # passage came from for the gold-recall@k check.
    corpus: list[str] = []
    gold_passage_idx_for_claim: list[int] = []  # per pair: index into corpus
    for p in pairs:
        gold_passage_idx_for_claim.append(len(corpus))
        corpus.append(p["evidence"])

    logger.info("c4_retrieval_realism_corpus_built", n_passages=len(corpus))

    # Embed corpus + claims (the claims are the queries the vector DB sees).
    claims = [p["claim"] for p in pairs]
    corpus_emb = _embed(corpus)
    claim_emb = _embed(claims)
    idx = _build_index(corpus, corpus_emb)

    # Retrieve top-k for each claim.
    retrievals: list[list[tuple[int, float]]] = []
    for q in claim_emb:
        retrievals.append(_retrieve(idx, q, k=k))

    # Detectors --- the same three used everywhere else in C4.
    from ebrag.dialectic.conflict import ConflictDetector
    from ebrag.dialectic.llm_judge import LLMEntailmentJudge
    det = ConflictDetector()
    judge = LLMEntailmentJudge(model=judge_model, max_tokens=1024)
    judge._get_client()

    # For each (claim, retrieved_passage), record detector verdicts + latency.
    # The "gold" verdict is well-defined only when the retrieved passage IS the gold
    # evidence for that claim (otherwise the claim/passage relationship has no FEVER
    # label). We compute conflict_flag_rate on all retrieved pairs and
    # precision_on_gold on the subset where retrieved_passage == gold_evidence.

    records_ce: list[dict] = []
    records_bare: list[dict] = []
    records_ar: list[dict] = []
    lat_ce: list[float] = []
    lat_bare: list[float] = []
    lat_ar: list[float] = []

    # Cross-encoder is sync; LLM calls go through ThreadPoolExecutor.
    for i, p in enumerate(pairs):
        for passage_idx, sim in retrievals[i]:
            passage_text = corpus[passage_idx]
            is_gold = (passage_idx == gold_passage_idx_for_claim[i])
            t0 = time.time()
            ce_label, _ = det.check_pair(passage_text, p["claim"])
            lat_ce.append(time.time() - t0)
            records_ce.append({
                "claim_idx": i, "passage_idx": passage_idx, "is_gold": is_gold,
                "gold_conflict": bool(p["gold_conflict"]) if is_gold else None,
                "category": p["category"] if is_gold else "non_gold",
                "predicted_conflict": ce_label == EntailmentLabel.CONTRADICTION,
                "similarity": sim,
            })

    # LLM jobs: (i, passage_idx, mode) for each retrieved pair x {bare, answer_relevant}.
    jobs = []
    for i, _ in enumerate(pairs):
        for passage_idx, _sim in retrievals[i]:
            jobs.append((i, passage_idx, "bare"))
            jobs.append((i, passage_idx, "answer_relevant"))

    def llm_task(args):
        i, passage_idx, mode = args
        p = pairs[i]
        passage_text = corpus[passage_idx]
        t0 = time.time()
        if mode == "bare":
            j = judge.classify(passage_text, p["claim"])
        else:
            q = f"Is this claim true: \"{p['claim']}\"?"
            j = judge.judge_answer_conflict(q, passage_text, p["claim"])
        return (i, passage_idx, mode), j.label, time.time() - t0

    preds: dict = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for key, label, dt in ex.map(llm_task, jobs):
            preds[key] = label
            (lat_ar if key[2] == "answer_relevant" else lat_bare).append(dt)

    for i, p in enumerate(pairs):
        for passage_idx, sim in retrievals[i]:
            is_gold = (passage_idx == gold_passage_idx_for_claim[i])
            common = {
                "claim_idx": i, "passage_idx": passage_idx, "is_gold": is_gold,
                "gold_conflict": bool(p["gold_conflict"]) if is_gold else None,
                "category": p["category"] if is_gold else "non_gold",
                "similarity": sim,
            }
            records_bare.append({**common,
                "predicted_conflict": preds[(i, passage_idx, "bare")] == EntailmentLabel.CONTRADICTION})
            records_ar.append({**common,
                "predicted_conflict": preds[(i, passage_idx, "answer_relevant")] == EntailmentLabel.CONTRADICTION})

    # Retrieval-level metrics (sanity): how often does top-k include the gold passage?
    gold_at_k = sum(1 for i in range(len(pairs))
                    if any(p_idx == gold_passage_idx_for_claim[i]
                           for p_idx, _ in retrievals[i]))
    gold_at_1 = sum(1 for i in range(len(pairs))
                    if retrievals[i] and retrievals[i][0][0] == gold_passage_idx_for_claim[i])

    def summarise(records: list[dict], latencies: list[float]) -> dict:
        n = len(records)
        flagged = sum(1 for r in records if r["predicted_conflict"])
        gold_records = [r for r in records if r["is_gold"]]
        gold_match = sum(1 for r in gold_records
                         if r["predicted_conflict"] == bool(r["gold_conflict"]))
        # Bootstrap CI on the conflict-flag rate.
        rate_ci = bootstrap_ci(records,
                               lambda rs: sum(1 for r in rs if r["predicted_conflict"]) / max(1, len(rs)))
        return {
            "n_pairs": n,
            "n_gold_pairs": len(gold_records),
            "conflict_flag_rate": flagged / max(1, n),
            "conflict_flag_rate_ci": rate_ci,
            "precision_on_gold": gold_match / max(1, len(gold_records)),
            "mean_latency_s": sum(latencies) / max(1, len(latencies)),
        }

    by_detector = {
        "cross_encoder_nli": {"metrics": summarise(records_ce, lat_ce), "records": records_ce},
        "llm_bare":          {"metrics": summarise(records_bare, lat_bare), "records": records_bare},
        "llm_answer_relevant": {"metrics": summarise(records_ar, lat_ar), "records": records_ar},
    }

    return {
        "experiment": "c4_retrieval_realism",
        "judge_model": judge_model,
        "n_claims": len(pairs),
        "n_corpus_passages": len(corpus),
        "k": k,
        "retrieval": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "index": "exact cosine over L2-normalised embeddings",
            "recall_at_k": gold_at_k / max(1, len(pairs)),
            "recall_at_1": gold_at_1 / max(1, len(pairs)),
        },
        "by_detector": by_detector,
    }


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    judge_model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
    results = run(n_per_class=n, judge_model=judge_model)

    out_dir = Path("benchmarks/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "c4_retrieval_realism.json"
    out_path.write_text(json.dumps(results, indent=2))

    print(f"\nC4 retrieval realism: {results['n_claims']} claims, "
          f"|corpus|={results['n_corpus_passages']}, k={results['k']}, "
          f"recall@k={results['retrieval']['recall_at_k']:.2f}, judge={judge_model}\n")
    hdr = f"{'detector':22s} | flag-rate (CI)       prec@gold   latency(s)"
    print(hdr); print("-" * len(hdr))
    for name, d in results["by_detector"].items():
        m = d["metrics"]; ci = m["conflict_flag_rate_ci"]
        print(f"{name:22s} | {m['conflict_flag_rate']:.2f} [{ci[0]:.2f},{ci[1]:.2f}]   "
              f"{m['precision_on_gold']:.2f}        {m['mean_latency_s']:.3f}")
    print(f"\n  -> wrote {out_path}")


if __name__ == "__main__":
    main()
