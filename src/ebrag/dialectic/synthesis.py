"""
Synthesis engine for balanced context preparation.

Combines thesis and antithesis passages into coherent
context that acknowledges multiple perspectives.
"""

from typing import Any

from ebrag.common import get_logger, get_settings
from ebrag.dialectic.models import (
    ConflictAnalysis,
    DialecticResult,
    SynthesizedContext,
)
from ebrag.dialectic.conflict import ConflictDetector, get_conflict_detector
from ebrag.dialectic.provenance import ProvenanceTracker, get_provenance_tracker
from ebrag.dialectic.calibration import ConfidenceCalibrator, get_confidence_calibrator
from ebrag.retrieval.models import RetrievalResult, ScoredPassage

logger = get_logger(__name__)


class SynthesisEngine:
    """
    Synthesizes balanced context from multiple perspectives.

    Combines thesis and antithesis passages, addresses
    conflicts, and prepares context for LLM generation.
    """

    def __init__(
        self,
        conflict_detector: ConflictDetector | None = None,
        provenance_tracker: ProvenanceTracker | None = None,
        calibrator: ConfidenceCalibrator | None = None,
    ) -> None:
        self.settings = get_settings()
        self.conflict_detector = conflict_detector or get_conflict_detector()
        self.provenance_tracker = provenance_tracker or get_provenance_tracker()
        self.calibrator = calibrator or get_confidence_calibrator()

        # LLM client for synthesis (lazy loaded)
        self._llm_client: Any = None

        logger.info("synthesis_engine_created")

    def synthesize(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        use_llm: bool = True,
    ) -> SynthesizedContext:
        """
        Synthesize balanced context from retrieval results.

        Args:
            query: Original query
            retrieval_result: Results from hybrid retrieval
            use_llm: Whether to use LLM for synthesis

        Returns:
            SynthesizedContext with balanced presentation
        """
        thesis_passages = retrieval_result.thesis_passages
        antithesis_passages = retrieval_result.antithesis_passages
        all_passages = retrieval_result.all_passages

        # Detect conflicts
        conflict_analysis = self.conflict_detector.detect_conflicts(
            all_passages,
            query=query,
        )

        # Generate summaries
        if use_llm and (thesis_passages or antithesis_passages):
            thesis_summary = self._summarize_perspective(
                query,
                thesis_passages,
                perspective="supporting",
            )
            antithesis_summary = self._summarize_perspective(
                query,
                antithesis_passages,
                perspective="alternative",
            )
        else:
            thesis_summary = self._simple_summary(thesis_passages)
            antithesis_summary = self._simple_summary(antithesis_passages)

        # Build balanced context
        balanced_context = self._build_balanced_context(
            query=query,
            thesis_summary=thesis_summary,
            antithesis_summary=antithesis_summary,
            conflict_analysis=conflict_analysis,
            all_passages=all_passages,
        )

        # List addressed conflicts
        conflicts_addressed = [
            f"{cp.passage_a_source} vs {cp.passage_b_source}: {cp.conflict_type.value}"
            for cp in conflict_analysis.conflict_pairs[:5]
        ]

        synthesis = SynthesizedContext(
            query=query,
            original_passages=[p.text for p in all_passages],
            thesis_summary=thesis_summary,
            antithesis_summary=antithesis_summary,
            balanced_context=balanced_context,
            conflicts_addressed=conflicts_addressed,
            resolution_strategy=conflict_analysis.synthesis_strategy,
            synthesis_model=self.settings.dialectic.synthesis_model if use_llm else "rule-based",
        )

        logger.info(
            "synthesis_complete",
            query_length=len(query),
            thesis_passages=len(thesis_passages),
            antithesis_passages=len(antithesis_passages),
            conflicts=len(conflicts_addressed),
        )

        return synthesis

    def analyze(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        generated_response: str | None = None,
    ) -> DialecticResult:
        """
        Perform full dialectic analysis.

        Args:
            query: Original query
            retrieval_result: Results from hybrid retrieval
            generated_response: Optional generated response for provenance

        Returns:
            DialecticResult with full analysis
        """
        all_passages = retrieval_result.all_passages

        # Conflict analysis
        conflict_analysis = self.conflict_detector.detect_conflicts(
            all_passages,
            query=query,
        )

        # Provenance tracking (if response provided)
        provenance = None
        if generated_response:
            provenance = self.provenance_tracker.track(
                query=query,
                generated_text=generated_response,
                passages=all_passages,
            )

        # Confidence calibration
        raw_confidence = retrieval_result.all_passages[0].combined_score if all_passages else 0.0
        confidence = self.calibrator.calibrate(
            raw_confidence=raw_confidence,
            retrieval_score=retrieval_result.diversity_score,
            attribution_score=provenance.coverage_score if provenance else 0.0,
        )

        # Synthesis
        synthesis = self.synthesize(
            query=query,
            retrieval_result=retrieval_result,
            use_llm=False,  # Use simple synthesis for analysis
        )

        # Calculate overall quality
        quality_factors = [
            retrieval_result.diversity_score,
            1.0 - conflict_analysis.max_conflict_score,
            confidence.calibrated_score,
        ]
        if provenance:
            quality_factors.append(provenance.coverage_score)

        information_quality = sum(quality_factors) / len(quality_factors)

        # Generate recommendation
        recommendation = self._generate_recommendation(
            conflict_analysis=conflict_analysis,
            confidence=confidence,
            provenance=provenance,
        )

        return DialecticResult(
            query=query,
            conflict_analysis=conflict_analysis,
            provenance=provenance,
            confidence=confidence,
            synthesis=synthesis,
            information_quality=information_quality,
            recommendation=recommendation,
        )

    def _summarize_perspective(
        self,
        query: str,
        passages: list[ScoredPassage],
        perspective: str,
    ) -> str:
        """Summarize passages from a particular perspective."""
        if not passages:
            return ""

        passage_texts = "\n\n".join(
            f"[{i+1}] {p.text}" for i, p in enumerate(passages[:5])
        )

        prompt = f"""Summarize the following passages that represent the {perspective} perspective on: "{query}"

Passages:
{passage_texts}

Provide a concise summary (2-3 sentences) of the key points from this perspective:"""

        try:
            client = self._get_llm_client()
            response = client.chat.completions.create(
                model=self.settings.dialectic.synthesis_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning("llm_summarization_failed", error=str(e))
            return self._simple_summary(passages)

    def _simple_summary(self, passages: list[ScoredPassage]) -> str:
        """Create a simple concatenated summary without LLM."""
        if not passages:
            return ""

        # Take first sentence from top passages
        summaries = []
        for p in passages[:3]:
            text = p.text.strip()
            # Get first sentence
            end_markers = [". ", "! ", "? "]
            min_end = len(text)
            for marker in end_markers:
                pos = text.find(marker)
                if pos > 0 and pos < min_end:
                    min_end = pos + 1

            first_sentence = text[:min_end].strip()
            if first_sentence:
                summaries.append(first_sentence)

        return " ".join(summaries)

    def _build_balanced_context(
        self,
        query: str,
        thesis_summary: str,
        antithesis_summary: str,
        conflict_analysis: ConflictAnalysis,
        all_passages: list[ScoredPassage],
    ) -> str:
        """Build balanced context for LLM generation."""
        parts = []

        # Add passage content
        parts.append("=== Retrieved Information ===")
        for i, p in enumerate(all_passages[:10]):
            source_note = f"[Source: {p.source_id}]"
            parts.append(f"\n[{i+1}] {source_note}\n{p.text}")

        # Add perspective summaries if available
        if thesis_summary or antithesis_summary:
            parts.append("\n\n=== Perspective Analysis ===")

            if thesis_summary:
                parts.append(f"\nSupporting evidence: {thesis_summary}")

            if antithesis_summary:
                parts.append(f"\nAlternative perspectives: {antithesis_summary}")

        # Add conflict notes if significant
        if conflict_analysis.num_contradictions > 0:
            parts.append("\n\n=== Important Notes ===")
            parts.append(
                f"Note: {conflict_analysis.num_contradictions} potential "
                f"contradiction(s) detected in sources. Please consider "
                f"multiple perspectives in your response."
            )

        return "\n".join(parts)

    def _generate_recommendation(
        self,
        conflict_analysis: ConflictAnalysis,
        confidence: Any,
        provenance: Any | None,
    ) -> str:
        """Generate a recommendation based on analysis."""
        issues = []

        # Check conflicts
        if conflict_analysis.num_contradictions > 0:
            issues.append(
                f"Found {conflict_analysis.num_contradictions} contradictions - "
                f"present multiple viewpoints"
            )

        # Check confidence
        if confidence.is_uncertain:
            issues.append(f"Low confidence: {confidence.uncertainty_reason}")

        # Check provenance
        if provenance and provenance.coverage_score < 0.5:
            issues.append(
                f"Only {provenance.coverage_score:.0%} of response grounded in sources"
            )

        if not issues:
            return "High quality response expected. Sources are consistent and well-grounded."

        return "Caution advised: " + "; ".join(issues)

    def _get_llm_client(self) -> Any:
        """Get the OpenAI-compatible client (honors base_url for Ollama Cloud etc.)."""
        if self._llm_client is None:
            from openai import OpenAI
            self._llm_client = OpenAI(**self.settings.llm.openai_client_kwargs())
        return self._llm_client

    def prepare_prompt_context(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        include_conflicts: bool = True,
        include_citations: bool = True,
        max_passages: int = 10,
    ) -> str:
        """
        Prepare context string for LLM prompting.

        Args:
            query: Original query
            retrieval_result: Retrieval results
            include_conflicts: Include conflict warnings
            include_citations: Include source citations
            max_passages: Maximum passages to include

        Returns:
            Formatted context string
        """
        synthesis = self.synthesize(
            query=query,
            retrieval_result=retrieval_result,
            use_llm=False,
        )

        context_parts = []

        # Add balanced context
        context_parts.append(synthesis.balanced_context)

        # Add citation guide if requested
        if include_citations:
            context_parts.append("\n\n=== Citation Guide ===")
            context_parts.append(
                "When using information from sources, cite them as [1], [2], etc. "
                "based on the passage numbers above."
            )

        return "\n".join(context_parts)


# Global engine instance
_engine: SynthesisEngine | None = None


def get_synthesis_engine() -> SynthesisEngine:
    """Get the global synthesis engine."""
    global _engine
    if _engine is None:
        _engine = SynthesisEngine()
    return _engine
