"""
Shared types and data models for EB-RAG.

These models define the core data structures used across all modules.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================


class DraftRole(str, Enum):
    """Role for drafting agents."""

    THESIS = "thesis"
    ANTITHESIS = "antithesis"


class ConflictSeverity(str, Enum):
    """Severity level of detected conflicts."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ValidationStatus(str, Enum):
    """Citation validation status."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


# =============================================================================
# Document & Retrieval Types
# =============================================================================


class Passage(BaseModel):
    """A retrieved passage with metadata."""

    id: str
    source_id: str
    document_uri: str
    text: str
    chunk_index: int

    # Scores
    dense_score: float = 0.0
    sparse_score: float = 0.0
    combined_score: float = 0.0
    diversity_bucket: int = 0
    contradiction_seed: float = 0.0

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)
    access_level: str = "public"
    created_at: datetime | None = None


class RetrievalBundle(BaseModel):
    """Bundle of retrieved passages for thesis/antithesis."""

    thesis_passages: list[Passage] = Field(default_factory=list)
    antithesis_passages: list[Passage] = Field(default_factory=list)
    diversity_score: float = 0.0
    retrieval_time_ms: float = 0.0


# =============================================================================
# Citation Types
# =============================================================================


class Citation(BaseModel):
    """A citation linking a claim to source passage."""

    source_id: str
    chunk_index: int
    quoted_text: str
    claim_text: str
    confidence: float = Field(ge=0.0, le=1.0)
    validation_status: ValidationStatus = ValidationStatus.SKIPPED


class CitationValidationResult(BaseModel):
    """Result of citation validation."""

    citation: Citation
    is_supported: bool
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str = ""


# =============================================================================
# Dialectic Types
# =============================================================================


class Draft(BaseModel):
    """A thesis or antithesis draft answer."""

    role: DraftRole
    text: str
    citations: list[Citation] = Field(default_factory=list)
    confidence_seed: float = Field(ge=0.0, le=1.0, default=0.5)
    reasoning_steps: list[str] = Field(default_factory=list)
    token_usage: int = 0


class Conflict(BaseModel):
    """A detected conflict between drafts."""

    description: str
    severity: ConflictSeverity
    thesis_excerpt: str
    antithesis_excerpt: str
    affected_citations: list[str] = Field(default_factory=list)


class ConflictReport(BaseModel):
    """Report from the critic module."""

    conflicts: list[Conflict] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    hallucination_probability: float = Field(ge=0.0, le=1.0, default=0.0)
    overlap_score: float = Field(ge=0.0, le=1.0, default=0.0)
    overall_severity: ConflictSeverity = ConflictSeverity.LOW


class DiscrepancyFlag(BaseModel):
    """Flag indicating unresolved discrepancy in final answer."""

    description: str
    severity: ConflictSeverity
    affected_claims: list[str] = Field(default_factory=list)
    recommendation: str = ""


class SynthesisResult(BaseModel):
    """Result from the synthesizer module."""

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    citations: list[Citation] = Field(default_factory=list)
    discrepancy_flags: list[DiscrepancyFlag] = Field(default_factory=list)
    decision_memo: dict[str, Any] = Field(default_factory=dict)
    reasoning_trace: list[str] = Field(default_factory=list)


# =============================================================================
# Tool Calling Types
# =============================================================================


class ToolParameter(BaseModel):
    """Parameter for a tool."""

    name: str
    type: str
    description: str
    required: bool = False
    default: Any = None


class ToolDescriptor(BaseModel):
    """Descriptor for a registered tool."""

    tool_id: str
    name: str
    description: str
    capabilities: list[str] = Field(default_factory=list)
    parameters: list[ToolParameter] = Field(default_factory=list)
    data_classification: str = "public"
    approval_policy: str = "auto"


class ToolPlan(BaseModel):
    """Proposed tool execution plan."""

    tool_id: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    justification: str
    confidence: float = Field(ge=0.0, le=1.0)
    requires_approval: bool = False


class ToolResult(BaseModel):
    """Result from tool execution."""

    tool_id: str
    success: bool
    output: Any = None
    error: str | None = None
    execution_time_ms: float = 0.0
    audit_id: str = ""


# =============================================================================
# API Request/Response Types
# =============================================================================


class ConversationContext(BaseModel):
    """Structured conversation context for chat mode."""

    facts: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    unresolved_conflicts: list[DiscrepancyFlag] = Field(default_factory=list)
    summary: str = ""


class AnswerRequest(BaseModel):
    """Request to the /v1/answers endpoint."""

    query: str
    context_filters: dict[str, Any] = Field(default_factory=dict)
    mode: str = "eb-rag"
    compliance_profile: str | None = None
    response_format: str = "text"
    session_id: str | None = None
    conversation_context: ConversationContext | None = None
    tool_preferences: dict[str, Any] | None = None


class AnswerResponse(BaseModel):
    """Response from the /v1/answers endpoint."""

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    citations: list[Citation] = Field(default_factory=list)
    discrepancy_flags: list[DiscrepancyFlag] = Field(default_factory=list)
    audit_id: str
    trace_pointer: str
    tool_plan: ToolPlan | None = None
    tool_results: list[ToolResult] | None = None
    session_state_delta: dict[str, Any] | None = None


# =============================================================================
# Telemetry Types
# =============================================================================


class TelemetryRecord(BaseModel):
    """Telemetry data for a single request."""

    request_id: str
    tenant_id: str
    mode: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Dataset tags (for benchmark runs)
    datasets: list[str] = Field(default_factory=list)

    # Retrieval stats
    retrieval_stats: dict[str, Any] = Field(default_factory=dict)

    # Critic scores
    critic_scores: dict[str, float] = Field(default_factory=dict)

    # Final metrics
    confidence: float = 0.0
    validator_results: dict[str, Any] = Field(default_factory=dict)

    # Performance
    latency_ms: float = 0.0
    cost_estimate: float = 0.0
    token_usage: dict[str, int] = Field(default_factory=dict)
