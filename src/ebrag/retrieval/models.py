"""
Data models for the retrieval layer.

Defines query types, retrieval results, and index metadata.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class QueryIntent(str, Enum):
    """Intent classification for queries."""

    FACTUAL = "factual"
    COMPARATIVE = "comparative"
    CAUSAL = "causal"
    PROCEDURAL = "procedural"
    OPINION = "opinion"
    UNKNOWN = "unknown"


class RetrievalMode(str, Enum):
    """Retrieval mode controlling pipeline behavior."""

    VANILLA = "vanilla"  # Single query, no thesis/antithesis
    EBRAG = "eb-rag"     # Dual query with thesis/antithesis
    BENCHMARK = "benchmark"  # Same as eb-rag with benchmark tagging


class ScoredPassage(BaseModel):
    """A passage with retrieval scores."""

    id: str
    text: str
    source_id: str
    document_uri: str
    chunk_index: int

    # Scores
    dense_score: float = 0.0
    sparse_score: float = 0.0
    combined_score: float = 0.0
    rerank_score: float | None = None

    # Diversity and stance
    diversity_bucket: int = 0
    stance_score: float = 0.0  # -1 (antithesis) to +1 (thesis)

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    def final_score(self) -> float:
        """Get the final ranking score."""
        if self.rerank_score is not None:
            return self.rerank_score
        return self.combined_score


class QueryPlan(BaseModel):
    """Plan for executing a retrieval query."""

    original_query: str
    thesis_query: str
    antithesis_query: str | None = None

    # Query analysis
    intent: QueryIntent = QueryIntent.UNKNOWN
    entities: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)

    # Retrieval parameters
    thesis_k: int = 5
    antithesis_k: int = 5
    use_reranking: bool = True


class RetrievalResult(BaseModel):
    """Result from a retrieval operation."""

    query: str
    mode: RetrievalMode

    # Retrieved passages split by role
    thesis_passages: list[ScoredPassage] = Field(default_factory=list)
    antithesis_passages: list[ScoredPassage] = Field(default_factory=list)

    # All passages combined and ranked
    all_passages: list[ScoredPassage] = Field(default_factory=list)

    # Metrics
    diversity_score: float = 0.0
    conflict_potential: float = 0.0  # How likely passages contain conflicting info

    # Timing
    retrieval_time_ms: float = 0.0
    rerank_time_ms: float = 0.0

    # Debug info
    query_plan: QueryPlan | None = None


class IndexMetadata(BaseModel):
    """Metadata for a vector/sparse index."""

    index_id: str
    namespace: str
    index_type: str  # "dense" or "sparse"
    backend: str  # "usearch", "bm25", etc.

    # Stats
    num_vectors: int = 0
    dimension: int | None = None  # For dense indices

    # Timestamps
    created_at: datetime
    updated_at: datetime

    # Config
    config: dict[str, Any] = Field(default_factory=dict)


class IndexStats(BaseModel):
    """Statistics for retrieval indices."""

    namespace: str
    dense_index: IndexMetadata | None = None
    sparse_index: IndexMetadata | None = None

    total_passages: int = 0
    total_documents: int = 0

    # Health metrics
    last_query_latency_ms: float | None = None
    avg_query_latency_ms: float | None = None
