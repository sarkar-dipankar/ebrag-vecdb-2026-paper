"""
Provenance tracking for source attribution.

Tracks which parts of generated responses are grounded
in retrieved sources, enabling citation and verification.
"""

import re
import uuid
from typing import Any

import numpy as np

from ebrag.common import get_logger, get_settings
from ebrag.dialectic.models import Citation, ProvenanceRecord
from ebrag.retrieval.models import ScoredPassage

logger = get_logger(__name__)


class ProvenanceTracker:
    """
    Tracks provenance of generated text to source passages.

    Uses semantic similarity to match generated claims
    to source passages for citation generation.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        verbatim_threshold: float = 0.95,
    ) -> None:
        self.settings = get_settings()
        self.similarity_threshold = similarity_threshold
        self.verbatim_threshold = verbatim_threshold

        # Lazy-loaded embedding model
        self._embedding_model: Any = None

        logger.info(
            "provenance_tracker_created",
            similarity_threshold=similarity_threshold,
        )

    @property
    def embedding_model(self) -> Any:
        """Get the embedding model (lazy loaded)."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(
                self.settings.embedding.model
            )
            logger.info("embedding_model_loaded", model=self.settings.embedding.model)

        return self._embedding_model

    def track(
        self,
        query: str,
        generated_text: str,
        passages: list[ScoredPassage],
        response_id: str | None = None,
    ) -> ProvenanceRecord:
        """
        Track provenance of generated text.

        Args:
            query: Original query
            generated_text: Generated response text
            passages: Source passages used for generation
            response_id: Optional response identifier

        Returns:
            ProvenanceRecord with citations
        """
        response_id = response_id or str(uuid.uuid4())[:8]

        # Split generated text into sentences/claims
        claims = self._extract_claims(generated_text)

        # Get passage embeddings
        passage_texts = [p.text for p in passages]
        passage_embeddings = self.embedding_model.encode(
            passage_texts,
            normalize_embeddings=True,
        )

        # Match claims to passages
        citations: list[Citation] = []
        unsupported_claims: list[str] = []

        for claim_text, claim_span in claims:
            citation = self._match_claim_to_passage(
                claim_text=claim_text,
                claim_span=claim_span,
                passages=passages,
                passage_embeddings=passage_embeddings,
            )

            if citation:
                citations.append(citation)
            else:
                unsupported_claims.append(claim_text)

        # Calculate coverage
        covered_chars = sum(
            c.generated_span[1] - c.generated_span[0]
            for c in citations
        )
        total_chars = len(generated_text)
        coverage_score = covered_chars / total_chars if total_chars > 0 else 0.0

        # Get unique sources
        source_ids = list(set(c.source_id for c in citations))

        record = ProvenanceRecord(
            response_id=response_id,
            query=query,
            generated_text=generated_text,
            citations=citations,
            coverage_score=coverage_score,
            unsupported_claims=unsupported_claims,
            num_sources=len(source_ids),
            source_ids=source_ids,
        )

        logger.info(
            "provenance_tracked",
            response_id=response_id,
            claims=len(claims),
            citations=len(citations),
            coverage=round(coverage_score, 3),
        )

        return record

    def _extract_claims(
        self,
        text: str,
    ) -> list[tuple[str, tuple[int, int]]]:
        """
        Extract individual claims from generated text.

        Returns list of (claim_text, (start, end)) tuples.
        """
        claims = []

        # Split on sentence boundaries
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)

        current_pos = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Find position in original text
            start = text.find(sentence, current_pos)
            if start == -1:
                start = current_pos

            end = start + len(sentence)
            claims.append((sentence, (start, end)))
            current_pos = end

        return claims

    def _match_claim_to_passage(
        self,
        claim_text: str,
        claim_span: tuple[int, int],
        passages: list[ScoredPassage],
        passage_embeddings: np.ndarray,
    ) -> Citation | None:
        """Match a claim to the best supporting passage."""
        if not passages:
            return None

        # Get claim embedding
        claim_embedding = self.embedding_model.encode(
            claim_text,
            normalize_embeddings=True,
        )

        # Calculate similarities
        similarities = np.dot(passage_embeddings, claim_embedding)
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        # Check if above threshold
        if best_score < self.similarity_threshold:
            return None

        passage = passages[best_idx]

        # Check for verbatim match
        is_verbatim = self._check_verbatim(claim_text, passage.text)

        # Check for paraphrase
        is_paraphrase = best_score >= 0.8 and not is_verbatim

        return Citation(
            citation_id=str(uuid.uuid4())[:8],
            passage_id=passage.id,
            source_id=passage.source_id,
            document_uri=passage.document_uri,
            generated_span=claim_span,
            attribution_score=best_score,
            is_verbatim=is_verbatim,
            is_paraphrase=is_paraphrase,
            verified=True,
            verification_method="semantic_similarity",
        )

    def _check_verbatim(self, claim: str, passage: str) -> bool:
        """Check if claim is verbatim from passage."""
        # Normalize for comparison
        claim_normalized = claim.lower().strip()
        passage_normalized = passage.lower()

        # Direct substring check
        if claim_normalized in passage_normalized:
            return True

        # Check word overlap
        claim_words = set(claim_normalized.split())
        passage_words = set(passage_normalized.split())

        if not claim_words:
            return False

        overlap = len(claim_words & passage_words) / len(claim_words)
        return overlap >= self.verbatim_threshold

    def verify_citation(
        self,
        claim: str,
        passage: ScoredPassage,
    ) -> tuple[bool, float]:
        """
        Verify a specific citation.

        Args:
            claim: The claimed text
            passage: The source passage

        Returns:
            Tuple of (is_valid, confidence_score)
        """
        claim_embedding = self.embedding_model.encode(
            claim,
            normalize_embeddings=True,
        )
        passage_embedding = self.embedding_model.encode(
            passage.text,
            normalize_embeddings=True,
        )

        similarity = float(np.dot(claim_embedding, passage_embedding))
        is_valid = similarity >= self.similarity_threshold

        return is_valid, similarity

    def get_citation_text(
        self,
        citation: Citation,
        format_style: str = "inline",
    ) -> str:
        """
        Format a citation for display.

        Args:
            citation: The citation to format
            format_style: "inline", "footnote", or "endnote"

        Returns:
            Formatted citation string
        """
        if format_style == "inline":
            return f"[{citation.source_id}]"
        elif format_style == "footnote":
            return f"[^{citation.citation_id}]"
        else:
            return f"({citation.source_id}, {citation.passage_id})"


# Global tracker instance
_tracker: ProvenanceTracker | None = None


def get_provenance_tracker() -> ProvenanceTracker:
    """Get the global provenance tracker."""
    global _tracker
    if _tracker is None:
        _tracker = ProvenanceTracker()
    return _tracker
