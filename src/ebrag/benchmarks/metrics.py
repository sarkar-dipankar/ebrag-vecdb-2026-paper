"""
Evaluation metrics for benchmarks.

Implements standard QA and retrieval metrics including:
- Exact Match (EM)
- Token-level F1
- ROUGE scores
- Retrieval recall/precision
- NDCG and MRR
"""

import re
import string
from collections import Counter
from typing import Sequence

import numpy as np

from ebrag.common import get_logger

logger = get_logger(__name__)


def normalize_answer(text: str) -> str:
    """
    Normalize answer text for comparison.

    Applies standard normalization:
    - Lowercase
    - Remove articles (a, an, the)
    - Remove punctuation
    - Collapse whitespace
    """
    # Lowercase
    text = text.lower()

    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Collapse whitespace
    text = " ".join(text.split())

    return text.strip()


def tokenize(text: str) -> list[str]:
    """Tokenize text into words."""
    return normalize_answer(text).split()


def exact_match(prediction: str, gold: str | list[str]) -> float:
    """
    Compute exact match score.

    Args:
        prediction: Predicted answer
        gold: Gold answer(s)

    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    pred_normalized = normalize_answer(prediction)

    if isinstance(gold, str):
        gold_answers = [gold]
    else:
        gold_answers = gold

    for gold_answer in gold_answers:
        if pred_normalized == normalize_answer(gold_answer):
            return 1.0

    return 0.0


def token_f1(prediction: str, gold: str | list[str]) -> tuple[float, float, float]:
    """
    Compute token-level F1 score.

    Args:
        prediction: Predicted answer
        gold: Gold answer(s)

    Returns:
        Tuple of (f1, precision, recall)
    """
    pred_tokens = tokenize(prediction)

    if isinstance(gold, str):
        gold_answers = [gold]
    else:
        gold_answers = gold

    best_f1 = 0.0
    best_precision = 0.0
    best_recall = 0.0

    for gold_answer in gold_answers:
        gold_tokens = tokenize(gold_answer)

        if not pred_tokens or not gold_tokens:
            continue

        # Count common tokens
        pred_counter = Counter(pred_tokens)
        gold_counter = Counter(gold_tokens)

        common = sum((pred_counter & gold_counter).values())

        if common == 0:
            continue

        precision = common / len(pred_tokens)
        recall = common / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)

        if f1 > best_f1:
            best_f1 = f1
            best_precision = precision
            best_recall = recall

    return best_f1, best_precision, best_recall


def rouge_n(
    prediction: str,
    gold: str | list[str],
    n: int = 1,
) -> float:
    """
    Compute ROUGE-N score.

    Args:
        prediction: Predicted text
        gold: Reference text(s)
        n: N-gram size

    Returns:
        ROUGE-N F1 score
    """
    pred_tokens = tokenize(prediction)

    if isinstance(gold, str):
        gold_answers = [gold]
    else:
        gold_answers = gold

    if len(pred_tokens) < n:
        return 0.0

    # Generate n-grams
    pred_ngrams = Counter(
        tuple(pred_tokens[i : i + n]) for i in range(len(pred_tokens) - n + 1)
    )

    best_score = 0.0

    for gold_answer in gold_answers:
        gold_tokens = tokenize(gold_answer)

        if len(gold_tokens) < n:
            continue

        gold_ngrams = Counter(
            tuple(gold_tokens[i : i + n]) for i in range(len(gold_tokens) - n + 1)
        )

        # Count overlapping n-grams
        overlap = sum((pred_ngrams & gold_ngrams).values())

        if overlap == 0:
            continue

        precision = overlap / sum(pred_ngrams.values())
        recall = overlap / sum(gold_ngrams.values())
        f1 = 2 * precision * recall / (precision + recall)

        best_score = max(best_score, f1)

    return best_score


def rouge_l(prediction: str, gold: str | list[str]) -> float:
    """
    Compute ROUGE-L score (longest common subsequence).

    Args:
        prediction: Predicted text
        gold: Reference text(s)

    Returns:
        ROUGE-L F1 score
    """
    pred_tokens = tokenize(prediction)

    if isinstance(gold, str):
        gold_answers = [gold]
    else:
        gold_answers = gold

    if not pred_tokens:
        return 0.0

    best_score = 0.0

    for gold_answer in gold_answers:
        gold_tokens = tokenize(gold_answer)

        if not gold_tokens:
            continue

        # Compute LCS length using dynamic programming
        lcs_length = _lcs_length(pred_tokens, gold_tokens)

        if lcs_length == 0:
            continue

        precision = lcs_length / len(pred_tokens)
        recall = lcs_length / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)

        best_score = max(best_score, f1)

    return best_score


def _lcs_length(seq1: list[str], seq2: list[str]) -> int:
    """Compute length of longest common subsequence."""
    m, n = len(seq1), len(seq2)

    # Use 1D DP for space efficiency
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev

    return prev[n]


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int,
) -> float:
    """
    Compute recall@k for retrieval.

    Args:
        retrieved_ids: List of retrieved document IDs (ordered by rank)
        relevant_ids: List of relevant document IDs
        k: Number of top results to consider

    Returns:
        Recall@k score
    """
    if not relevant_ids:
        return 0.0

    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)

    hits = len(retrieved_set & relevant_set)
    return hits / len(relevant_set)


def precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int,
) -> float:
    """
    Compute precision@k for retrieval.

    Args:
        retrieved_ids: List of retrieved document IDs (ordered by rank)
        relevant_ids: List of relevant document IDs
        k: Number of top results to consider

    Returns:
        Precision@k score
    """
    if k == 0:
        return 0.0

    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)

    hits = len(retrieved_set & relevant_set)
    return hits / k


def mean_reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).

    Args:
        retrieved_ids: List of retrieved document IDs (ordered by rank)
        relevant_ids: List of relevant document IDs

    Returns:
        MRR score (1/rank of first relevant result)
    """
    relevant_set = set(relevant_ids)

    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_set:
            return 1.0 / (i + 1)

    return 0.0


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int,
    relevance_scores: dict[str, float] | None = None,
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at k.

    Args:
        retrieved_ids: List of retrieved document IDs (ordered by rank)
        relevant_ids: List of relevant document IDs
        k: Number of top results to consider
        relevance_scores: Optional mapping of doc_id to relevance score

    Returns:
        NDCG@k score
    """
    if not relevant_ids:
        return 0.0

    # Default to binary relevance if no scores provided
    if relevance_scores is None:
        relevance_scores = {doc_id: 1.0 for doc_id in relevant_ids}

    # Compute DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        rel = relevance_scores.get(doc_id, 0.0)
        dcg += rel / np.log2(i + 2)  # +2 because i starts at 0

    # Compute ideal DCG
    ideal_scores = sorted(
        [relevance_scores.get(doc_id, 0.0) for doc_id in relevant_ids],
        reverse=True,
    )[:k]

    idcg = 0.0
    for i, rel in enumerate(ideal_scores):
        idcg += rel / np.log2(i + 2)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def answer_in_context(
    answer: str,
    passages: Sequence[str],
    fuzzy: bool = True,
) -> tuple[bool, int | None]:
    """
    Check if answer appears in retrieved passages.

    Args:
        answer: The gold answer
        passages: Retrieved passages
        fuzzy: Whether to use fuzzy matching

    Returns:
        Tuple of (found, position) where position is 0-indexed
    """
    answer_normalized = normalize_answer(answer)
    answer_tokens = set(answer_normalized.split())

    for i, passage in enumerate(passages):
        passage_normalized = normalize_answer(passage)

        # Exact substring match
        if answer_normalized in passage_normalized:
            return True, i

        # Fuzzy match: check if most answer tokens appear
        if fuzzy and answer_tokens:
            passage_tokens = set(passage_normalized.split())
            overlap = len(answer_tokens & passage_tokens) / len(answer_tokens)
            if overlap >= 0.8:
                return True, i

    return False, None


def semantic_similarity(
    prediction: str,
    gold: str,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> float:
    """
    Compute semantic similarity using embeddings.

    Args:
        prediction: Predicted text
        gold: Reference text
        embedding_model: Name of sentence-transformers model

    Returns:
        Cosine similarity score
    """
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(embedding_model)

        embeddings = model.encode([prediction, gold], normalize_embeddings=True)
        similarity = np.dot(embeddings[0], embeddings[1])

        return float(similarity)

    except Exception as e:
        logger.warning("semantic_similarity_failed", error=str(e))
        return 0.0


def compute_all_answer_metrics(
    prediction: str,
    gold: str | list[str],
) -> dict[str, float]:
    """
    Compute all answer quality metrics.

    Args:
        prediction: Predicted answer
        gold: Gold answer(s)

    Returns:
        Dictionary of metric name to score
    """
    f1, precision, recall = token_f1(prediction, gold)

    return {
        "exact_match": exact_match(prediction, gold),
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "rouge_1": rouge_n(prediction, gold, n=1),
        "rouge_2": rouge_n(prediction, gold, n=2),
        "rouge_l": rouge_l(prediction, gold),
    }


def compute_all_retrieval_metrics(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, float]:
    """
    Compute all retrieval metrics.

    Args:
        retrieved_ids: List of retrieved document IDs
        relevant_ids: List of relevant document IDs
        k_values: K values to compute metrics for

    Returns:
        Dictionary of metric name to score
    """
    metrics = {
        "mrr": mean_reciprocal_rank(retrieved_ids, relevant_ids),
    }

    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
        metrics[f"precision@{k}"] = precision_at_k(retrieved_ids, relevant_ids, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, relevant_ids, k)

    return metrics


def aggregate_metrics(
    results: Sequence[dict[str, float]],
) -> dict[str, float]:
    """
    Aggregate metrics across multiple questions.

    Args:
        results: List of per-question metric dictionaries

    Returns:
        Dictionary of averaged metrics
    """
    if not results:
        return {}

    # Collect all metric names
    all_keys = set()
    for r in results:
        all_keys.update(r.keys())

    # Compute averages
    aggregated = {}
    for key in all_keys:
        values = [r.get(key, 0.0) for r in results]
        aggregated[key] = sum(values) / len(values)

    return aggregated
