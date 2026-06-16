"""Retrieval layer for EB-RAG."""

from ebrag.retrieval.models import (
    QueryIntent,
    QueryPlan,
    RetrievalMode,
    RetrievalResult,
    ScoredPassage,
)
from ebrag.retrieval.dense import (
    DenseIndex,
    USearchIndex,
    get_dense_index_manager,
)
from ebrag.retrieval.sparse import (
    SparseIndex,
    BM25Index,
    get_sparse_index_manager,
)
from ebrag.retrieval.planner import (
    QueryPlanner,
    get_query_planner,
)
from ebrag.retrieval.hybrid import (
    HybridRetriever,
    get_retriever,
)

__all__ = [
    # Models
    "QueryIntent",
    "QueryPlan",
    "RetrievalMode",
    "RetrievalResult",
    "ScoredPassage",
    # Dense
    "DenseIndex",
    "USearchIndex",
    "get_dense_index_manager",
    # Sparse
    "SparseIndex",
    "BM25Index",
    "get_sparse_index_manager",
    # Planner
    "QueryPlanner",
    "get_query_planner",
    # Hybrid
    "HybridRetriever",
    "get_retriever",
]
