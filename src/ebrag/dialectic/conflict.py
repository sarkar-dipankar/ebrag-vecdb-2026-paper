"""
Conflict detection using Natural Language Inference.

Detects contradictions and inconsistencies between
retrieved passages using NLI models.
"""

from itertools import combinations
from typing import Any

import numpy as np

from ebrag.common import get_logger, get_settings
from ebrag.dialectic.models import (
    ConflictAnalysis,
    ConflictPair,
    ConflictType,
    EntailmentLabel,
)
from ebrag.retrieval.models import ScoredPassage

logger = get_logger(__name__)


class ConflictDetector:
    """
    Detects conflicts between passages using NLI.

    Uses a cross-encoder NLI model to classify pairs of
    passages as entailment, contradiction, or neutral.
    """

    def __init__(
        self,
        model_name: str | None = None,
        contradiction_threshold: float = 0.7,
        batch_size: int = 32,
    ) -> None:
        self.settings = get_settings()
        self.model_name = model_name or self.settings.dialectic.nli_model
        self.contradiction_threshold = contradiction_threshold
        self.batch_size = batch_size

        # Lazy-loaded model
        self._model: Any = None

        logger.info(
            "conflict_detector_created",
            model=self.model_name,
            threshold=contradiction_threshold,
        )

    @property
    def model(self) -> Any:
        """Get the NLI model (lazy loaded)."""
        if self._model is None:
            from transformers import pipeline

            self._model = pipeline(
                "text-classification",
                model=self.model_name,
                top_k=None,  # Return all scores
            )
            logger.info("nli_model_loaded", model=self.model_name)

        return self._model

    def detect_conflicts(
        self,
        passages: list[ScoredPassage],
        query: str | None = None,
    ) -> ConflictAnalysis:
        """
        Detect conflicts among a set of passages.

        Args:
            passages: List of retrieved passages
            query: Original query (for context)

        Returns:
            ConflictAnalysis with detected conflicts
        """
        if len(passages) < 2:
            return ConflictAnalysis(
                query=query or "",
                total_passages=len(passages),
            )

        # Generate all pairs
        pairs = list(combinations(range(len(passages)), 2))

        # Analyze pairs in batches
        conflict_pairs: list[ConflictPair] = []

        for i in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[i : i + self.batch_size]
            batch_results = self._analyze_batch(passages, batch_pairs)
            conflict_pairs.extend(batch_results)

        # Filter to significant conflicts
        significant_conflicts = [
            p for p in conflict_pairs
            if p.conflict_score >= self.contradiction_threshold
        ]

        # Calculate aggregate metrics
        conflict_scores = [p.conflict_score for p in conflict_pairs]
        max_score = max(conflict_scores) if conflict_scores else 0.0
        avg_score = np.mean(conflict_scores) if conflict_scores else 0.0

        num_contradictions = sum(
            1 for p in conflict_pairs
            if p.entailment_label == EntailmentLabel.CONTRADICTION
        )

        # Build conflict clusters
        clusters = self._build_conflict_clusters(significant_conflicts, passages)

        # Determine if synthesis is needed
        needs_synthesis = (
            num_contradictions > 0 or
            max_score > 0.8 or
            len(clusters) > 1
        )

        analysis = ConflictAnalysis(
            query=query or "",
            total_passages=len(passages),
            conflict_pairs=significant_conflicts,
            max_conflict_score=max_score,
            avg_conflict_score=float(avg_score),
            num_contradictions=num_contradictions,
            conflict_clusters=clusters,
            needs_synthesis=needs_synthesis,
            synthesis_strategy=self._recommend_strategy(significant_conflicts),
        )

        logger.info(
            "conflict_detection_complete",
            total_passages=len(passages),
            pairs_analyzed=len(pairs),
            conflicts_found=len(significant_conflicts),
            max_score=round(max_score, 3),
        )

        return analysis

    def _analyze_batch(
        self,
        passages: list[ScoredPassage],
        pairs: list[tuple[int, int]],
    ) -> list[ConflictPair]:
        """Analyze a batch of passage pairs."""
        results = []

        # Prepare inputs for NLI model
        inputs = []
        for i, j in pairs:
            # NLI format: premise, hypothesis
            inputs.append({
                "text": passages[i].text,
                "text_pair": passages[j].text,
            })

        # Run NLI model
        try:
            predictions = self.model(
                [f"{inp['text']} </s></s> {inp['text_pair']}" for inp in inputs],
                truncation=True,
                max_length=512,
            )
        except Exception as e:
            logger.error("nli_inference_failed", error=str(e))
            return results

        # Process predictions
        for (i, j), pred in zip(pairs, predictions):
            conflict_pair = self._process_prediction(
                passages[i],
                passages[j],
                pred,
            )
            results.append(conflict_pair)

        return results

    def _process_prediction(
        self,
        passage_a: ScoredPassage,
        passage_b: ScoredPassage,
        prediction: list[dict[str, Any]],
    ) -> ConflictPair:
        """Process NLI model prediction into ConflictPair."""
        # Extract scores by label
        scores = {p["label"].lower(): p["score"] for p in prediction}

        # Determine entailment label
        contradiction_score = scores.get("contradiction", 0.0)
        entailment_score = scores.get("entailment", 0.0)
        neutral_score = scores.get("neutral", 0.0)

        if contradiction_score > max(entailment_score, neutral_score):
            label = EntailmentLabel.CONTRADICTION
            conflict_score = contradiction_score
        elif entailment_score > max(contradiction_score, neutral_score):
            label = EntailmentLabel.ENTAILMENT
            conflict_score = 0.0  # Entailment means no conflict
        else:
            label = EntailmentLabel.NEUTRAL
            conflict_score = 0.0

        # Determine conflict type
        conflict_type = ConflictType.NONE
        if label == EntailmentLabel.CONTRADICTION:
            conflict_type = ConflictType.CONTRADICTION

        return ConflictPair(
            passage_a_id=passage_a.id,
            passage_b_id=passage_b.id,
            passage_a_text=passage_a.text[:500],
            passage_b_text=passage_b.text[:500],
            conflict_type=conflict_type,
            conflict_score=conflict_score,
            entailment_label=label,
            entailment_score=max(contradiction_score, entailment_score, neutral_score),
            passage_a_source=passage_a.source_id,
            passage_b_source=passage_b.source_id,
        )

    def _build_conflict_clusters(
        self,
        conflicts: list[ConflictPair],
        passages: list[ScoredPassage],
    ) -> list[list[str]]:
        """Build clusters of conflicting passages using union-find."""
        if not conflicts:
            return []

        # Build adjacency from conflicts
        passage_ids = {p.id for p in passages}
        parent: dict[str, str] = {pid: pid for pid in passage_ids}

        def find(x: str) -> str:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: str, y: str) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Union conflicting passages
        for conflict in conflicts:
            if conflict.passage_a_id in parent and conflict.passage_b_id in parent:
                union(conflict.passage_a_id, conflict.passage_b_id)

        # Build clusters
        clusters: dict[str, list[str]] = {}
        for pid in passage_ids:
            root = find(pid)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(pid)

        # Return only clusters with multiple members (actual conflicts)
        return [c for c in clusters.values() if len(c) > 1]

    def _recommend_strategy(self, conflicts: list[ConflictPair]) -> str:
        """Recommend synthesis strategy based on conflicts."""
        if not conflicts:
            return "none"

        # Check conflict types
        contradictions = sum(
            1 for c in conflicts
            if c.conflict_type == ConflictType.CONTRADICTION
        )

        if contradictions > 0:
            return "present_alternatives"

        # Check if conflicts are from different sources
        sources = set()
        for c in conflicts:
            sources.add(c.passage_a_source)
            sources.add(c.passage_b_source)

        if len(sources) > 2:
            return "multi_source_synthesis"

        return "clarify_differences"

    def check_pair(
        self,
        text_a: str,
        text_b: str,
    ) -> tuple[EntailmentLabel, float]:
        """
        Check entailment between two texts.

        Args:
            text_a: Premise text
            text_b: Hypothesis text

        Returns:
            Tuple of (label, score)
        """
        try:
            result = self.model(
                f"{text_a} </s></s> {text_b}",
                truncation=True,
                max_length=512,
            )

            # top_k=None returns List[Dict] for a single input on some transformers
            # versions and List[List[Dict]] on others; normalize to a flat list.
            if result and isinstance(result[0], list):
                result = result[0]

            scores = {p["label"].lower(): p["score"] for p in result}

            contradiction = scores.get("contradiction", 0.0)
            entailment = scores.get("entailment", 0.0)
            neutral = scores.get("neutral", 0.0)

            if contradiction > max(entailment, neutral):
                return EntailmentLabel.CONTRADICTION, contradiction
            elif entailment > max(contradiction, neutral):
                return EntailmentLabel.ENTAILMENT, entailment
            else:
                return EntailmentLabel.NEUTRAL, neutral

        except Exception as e:
            logger.error("nli_check_failed", error=str(e))
            return EntailmentLabel.NEUTRAL, 0.0


# Global detector instance
_detector: ConflictDetector | None = None


def get_conflict_detector() -> ConflictDetector:
    """Get the global conflict detector."""
    global _detector
    if _detector is None:
        _detector = ConflictDetector()
    return _detector
