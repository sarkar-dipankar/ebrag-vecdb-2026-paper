"""
Dialectic engine models.

Data structures for conflict detection, provenance tracking,
and confidence calibration.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ConflictType(str, Enum):
    """Types of conflicts between passages."""

    CONTRADICTION = "contradiction"  # Direct logical contradiction
    INCONSISTENCY = "inconsistency"  # Factual inconsistency
    TEMPORAL = "temporal"  # Different time periods
    PERSPECTIVE = "perspective"  # Different viewpoints on same topic
    SCOPE = "scope"  # Different scope/granularity
    NONE = "none"  # No conflict detected


class StanceLabel(str, Enum):
    """Stance of a passage relative to a claim."""

    SUPPORTS = "supports"
    REFUTES = "refutes"
    NEUTRAL = "neutral"


class EntailmentLabel(str, Enum):
    """NLI entailment labels."""

    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"


class ConflictPair(BaseModel):
    """A pair of passages with potential conflict."""

    passage_a_id: str
    passage_b_id: str
    passage_a_text: str
    passage_b_text: str

    # Conflict analysis
    conflict_type: ConflictType = ConflictType.NONE
    conflict_score: float = 0.0  # 0-1, higher = more conflict
    entailment_label: EntailmentLabel = EntailmentLabel.NEUTRAL
    entailment_score: float = 0.0

    # Explanation
    conflict_explanation: str = ""
    conflicting_claims: list[str] = Field(default_factory=list)

    # Source metadata
    passage_a_source: str = ""
    passage_b_source: str = ""


class ConflictAnalysis(BaseModel):
    """Analysis of conflicts across retrieved passages."""

    query: str
    total_passages: int
    conflict_pairs: list[ConflictPair] = Field(default_factory=list)

    # Aggregate metrics
    max_conflict_score: float = 0.0
    avg_conflict_score: float = 0.0
    num_contradictions: int = 0
    num_inconsistencies: int = 0

    # Clusters of conflicting information
    conflict_clusters: list[list[str]] = Field(default_factory=list)

    # Recommended action
    needs_synthesis: bool = False
    synthesis_strategy: str = ""


class Citation(BaseModel):
    """A citation linking generated text to source."""

    citation_id: str
    passage_id: str
    source_id: str
    document_uri: str

    # Text spans
    generated_span: tuple[int, int]  # Start, end in generated text
    source_span: tuple[int, int] | None = None  # Start, end in source

    # Attribution strength
    attribution_score: float = 0.0  # 0-1, how well source supports claim
    is_verbatim: bool = False
    is_paraphrase: bool = False

    # Verification
    verified: bool = False
    verification_method: str = ""


class ProvenanceRecord(BaseModel):
    """Provenance tracking for a generated response."""

    response_id: str
    query: str
    generated_text: str

    # Citations
    citations: list[Citation] = Field(default_factory=list)

    # Coverage metrics
    coverage_score: float = 0.0  # % of response grounded in sources
    unsupported_claims: list[str] = Field(default_factory=list)

    # Source diversity
    num_sources: int = 0
    source_ids: list[str] = Field(default_factory=list)


class ConfidenceScore(BaseModel):
    """Calibrated confidence score for a prediction."""

    raw_score: float  # Original model confidence
    calibrated_score: float  # After calibration
    bin_index: int = 0  # Calibration bin

    # Components
    retrieval_confidence: float = 0.0
    generation_confidence: float = 0.0
    attribution_confidence: float = 0.0

    # Uncertainty indicators
    is_uncertain: bool = False
    uncertainty_reason: str = ""


class CalibrationBin(BaseModel):
    """A bin for Expected Calibration Error calculation."""

    bin_lower: float
    bin_upper: float
    bin_count: int = 0
    bin_accuracy: float = 0.0
    bin_confidence: float = 0.0


class CalibrationMetrics(BaseModel):
    """Calibration metrics for confidence estimation."""

    # Expected Calibration Error
    ece: float = 0.0
    max_calibration_error: float = 0.0

    # Bin statistics
    bins: list[CalibrationBin] = Field(default_factory=list)

    # Reliability
    brier_score: float = 0.0
    reliability_diagram_data: list[tuple[float, float]] = Field(default_factory=list)


class SynthesizedContext(BaseModel):
    """Synthesized context balancing multiple perspectives."""

    query: str
    original_passages: list[str] = Field(default_factory=list)

    # Synthesized output
    thesis_summary: str = ""
    antithesis_summary: str = ""
    balanced_context: str = ""

    # Conflict resolution
    conflicts_addressed: list[str] = Field(default_factory=list)
    resolution_strategy: str = ""

    # Metadata
    synthesis_model: str = ""
    synthesis_prompt: str = ""


class DialecticResult(BaseModel):
    """Complete dialectic analysis result."""

    query: str

    # Conflict analysis
    conflict_analysis: ConflictAnalysis | None = None

    # Provenance
    provenance: ProvenanceRecord | None = None

    # Confidence
    confidence: ConfidenceScore | None = None

    # Synthesized context
    synthesis: SynthesizedContext | None = None

    # Overall assessment
    information_quality: float = 0.0  # 0-1
    recommendation: str = ""
