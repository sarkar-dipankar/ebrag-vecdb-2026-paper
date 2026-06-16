"""
Hybrid retrieval combining dense and sparse search.

Implements the main retrieval pipeline with:
- Dual dense + sparse search
- Score fusion and ranking
- Cross-encoder reranking
- Thesis/antithesis evidence splitting
"""

import time
from typing import Any

import numpy as np

from ebrag.common import get_logger, get_settings
from ebrag.ingestion import get_chunk_store
from ebrag.retrieval.dense import DenseIndex, get_dense_index_manager
from ebrag.retrieval.models import (
    QueryPlan,
    RetrievalMode,
    RetrievalResult,
    ScoredPassage,
)
from ebrag.retrieval.planner import QueryPlanner, get_query_planner
from ebrag.retrieval.sparse import SparseIndex, get_sparse_index_manager

logger = get_logger(__name__)


class HybridRetriever:
    """
    Hybrid retriever combining dense and sparse search.

    Implements evidence-balanced retrieval with thesis/antithesis
    query planning and cross-encoder reranking.
    """

    def __init__(
        self,
        namespace: str = "default",
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> None:
        self.namespace = namespace
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        self.settings = get_settings()
        self.planner = get_query_planner()

        # Get index managers
        self._dense_manager = get_dense_index_manager()
        self._sparse_manager = get_sparse_index_manager()

        # Lazy-loaded indices
        self._dense_index: DenseIndex | None = None
        self._sparse_index: SparseIndex | None = None

        # Embedding model (lazy loaded)
        self._embedding_model: Any = None

        # Reranker model (lazy loaded)
        self._reranker: Any = None

        logger.info(
            "hybrid_retriever_created",
            namespace=namespace,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )

    @property
    def dense_index(self) -> DenseIndex:
        """Get or create the dense index."""
        if self._dense_index is None:
            self._dense_index = self._dense_manager.get_or_create(
                self.namespace,
                dimension=self.settings.embedding.dimension,
            )
        return self._dense_index

    @property
    def sparse_index(self) -> SparseIndex:
        """Get or create the sparse index."""
        if self._sparse_index is None:
            self._sparse_index = self._sparse_manager.get_or_create(self.namespace)
        return self._sparse_index

    def _get_embedding_model(self) -> Any:
        """Get the embedding model."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(
                self.settings.embedding.model
            )
        return self._embedding_model

    def _get_reranker(self) -> Any:
        """Get the cross-encoder reranker."""
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(
                self.settings.retrieval.cross_encoder_model
            )
        return self._reranker

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a query string."""
        model = self._get_embedding_model()
        embedding = model.encode(
            query,
            normalize_embeddings=self.settings.embedding.normalize,
        )
        return np.array(embedding, dtype=np.float32)

    def retrieve(
        self,
        query: str,
        mode: RetrievalMode = RetrievalMode.EBRAG,
        k: int | None = None,
        rerank: bool | None = None,
    ) -> RetrievalResult:
        """
        Retrieve passages for a query.

        Args:
            query: The search query
            mode: Retrieval mode (vanilla or eb-rag)
            k: Total number of passages to retrieve
            rerank: Whether to apply cross-encoder reranking

        Returns:
            RetrievalResult with thesis and antithesis passages
        """
        start_time = time.perf_counter()

        k = k or self.settings.retrieval.top_k
        rerank = rerank if rerank is not None else self.settings.retrieval.use_cross_encoder

        # Create query plan
        plan = self.planner.plan(query, mode)

        # Retrieve passages for thesis query
        thesis_passages = self._retrieve_for_query(
            plan.thesis_query,
            k=plan.thesis_k if mode != RetrievalMode.VANILLA else k,
        )

        # Retrieve passages for antithesis query (if in EB-RAG mode)
        antithesis_passages: list[ScoredPassage] = []
        if plan.antithesis_query and mode != RetrievalMode.VANILLA:
            antithesis_passages = self._retrieve_for_query(
                plan.antithesis_query,
                k=plan.antithesis_k,
            )

            # Mark passages with stance scores
            for p in thesis_passages:
                p.stance_score = 1.0  # Thesis-supporting
            for p in antithesis_passages:
                p.stance_score = -1.0  # Antithesis-supporting

        retrieval_time = (time.perf_counter() - start_time) * 1000

        # Merge and deduplicate
        all_passages = self._merge_passages(thesis_passages, antithesis_passages)

        # Apply reranking if enabled
        rerank_time = 0.0
        if rerank and all_passages:
            rerank_start = time.perf_counter()
            all_passages = self._rerank_passages(query, all_passages)
            rerank_time = (time.perf_counter() - rerank_start) * 1000

        # Calculate diversity and conflict metrics
        diversity_score = self._calculate_diversity(all_passages)
        conflict_potential = self._calculate_conflict_potential(
            thesis_passages, antithesis_passages
        )

        # Assign diversity buckets
        all_passages = self._assign_diversity_buckets(all_passages)

        result = RetrievalResult(
            query=query,
            mode=mode,
            thesis_passages=thesis_passages[:plan.thesis_k],
            antithesis_passages=antithesis_passages[:plan.antithesis_k],
            all_passages=all_passages[:k],
            diversity_score=diversity_score,
            conflict_potential=conflict_potential,
            retrieval_time_ms=retrieval_time,
            rerank_time_ms=rerank_time,
            query_plan=plan,
        )

        logger.info(
            "retrieval_completed",
            query_length=len(query),
            mode=mode.value,
            thesis_count=len(result.thesis_passages),
            antithesis_count=len(result.antithesis_passages),
            total_count=len(result.all_passages),
            diversity=round(diversity_score, 3),
            retrieval_ms=round(retrieval_time, 2),
            rerank_ms=round(rerank_time, 2),
        )

        return result

    def _retrieve_for_query(
        self,
        query: str,
        k: int,
    ) -> list[ScoredPassage]:
        """Retrieve passages for a single query."""
        # Get more candidates for fusion
        candidate_k = k * 3

        # Dense retrieval
        query_embedding = self._embed_query(query)
        dense_results = self.dense_index.search(query_embedding, k=candidate_k)

        # Sparse retrieval
        sparse_results = self.sparse_index.search(query, k=candidate_k)

        # Fuse results
        passages = self._fuse_results(dense_results, sparse_results, k)

        return passages

    def _fuse_results(
        self,
        dense_results: list[tuple[str, float]],
        sparse_results: list[tuple[str, float]],
        k: int,
    ) -> list[ScoredPassage]:
        """Fuse dense and sparse results using weighted combination."""
        # Normalize scores
        dense_scores = self._normalize_scores(dense_results)
        sparse_scores = self._normalize_scores(sparse_results)

        # Combine scores
        combined: dict[str, dict[str, float]] = {}

        for doc_id, score in dense_scores:
            if doc_id not in combined:
                combined[doc_id] = {"dense": 0.0, "sparse": 0.0}
            combined[doc_id]["dense"] = score

        for doc_id, score in sparse_scores:
            if doc_id not in combined:
                combined[doc_id] = {"dense": 0.0, "sparse": 0.0}
            combined[doc_id]["sparse"] = score

        # Calculate combined scores
        results: list[tuple[str, float, float, float]] = []
        for doc_id, scores in combined.items():
            combined_score = (
                self.dense_weight * scores["dense"] +
                self.sparse_weight * scores["sparse"]
            )
            results.append((doc_id, combined_score, scores["dense"], scores["sparse"]))

        # Sort by combined score
        results.sort(key=lambda x: x[1], reverse=True)

        # Convert to ScoredPassages
        passages = []
        chunk_store = get_chunk_store()

        for doc_id, combined_score, dense_score, sparse_score in results[:k]:
            # Get chunk data from store
            chunk = chunk_store.get_chunk(doc_id)
            if chunk is None:
                continue

            passage = ScoredPassage(
                id=doc_id,
                text=chunk.text,
                source_id=chunk.metadata.source_id,
                document_uri=chunk.metadata.document_uri,
                chunk_index=chunk.metadata.chunk_index,
                dense_score=dense_score,
                sparse_score=sparse_score,
                combined_score=combined_score,
                metadata={
                    "namespace": chunk.metadata.namespace,
                    "dataset_name": chunk.metadata.dataset_name,
                },
            )
            passages.append(passage)

        return passages

    def _normalize_scores(
        self,
        results: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """Normalize scores to 0-1 range."""
        if not results:
            return []

        scores = [s for _, s in results]
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score

        if score_range == 0:
            return [(doc_id, 1.0) for doc_id, _ in results]

        return [
            (doc_id, (score - min_score) / score_range)
            for doc_id, score in results
        ]

    def _rerank_passages(
        self,
        query: str,
        passages: list[ScoredPassage],
    ) -> list[ScoredPassage]:
        """Rerank passages using cross-encoder."""
        if not passages:
            return passages

        try:
            reranker = self._get_reranker()

            # Create query-passage pairs
            pairs = [(query, p.text) for p in passages]

            # Get reranking scores
            scores = reranker.predict(pairs)

            # Update passages with rerank scores
            for passage, score in zip(passages, scores):
                passage.rerank_score = float(score)

            # Sort by rerank score
            passages.sort(key=lambda p: p.rerank_score or 0, reverse=True)

            return passages

        except Exception as e:
            logger.warning("reranking_failed", error=str(e))
            return passages

    def _merge_passages(
        self,
        thesis: list[ScoredPassage],
        antithesis: list[ScoredPassage],
    ) -> list[ScoredPassage]:
        """Merge thesis and antithesis passages, removing duplicates."""
        seen_ids: set[str] = set()
        merged: list[ScoredPassage] = []

        # Interleave thesis and antithesis for diversity
        max_len = max(len(thesis), len(antithesis))

        for i in range(max_len):
            if i < len(thesis) and thesis[i].id not in seen_ids:
                merged.append(thesis[i])
                seen_ids.add(thesis[i].id)

            if i < len(antithesis) and antithesis[i].id not in seen_ids:
                merged.append(antithesis[i])
                seen_ids.add(antithesis[i].id)

        return merged

    def _calculate_diversity(self, passages: list[ScoredPassage]) -> float:
        """Calculate diversity score based on source distribution."""
        if not passages:
            return 0.0

        # Count unique sources
        sources = set(p.source_id for p in passages)
        diversity = len(sources) / len(passages)

        return diversity

    def _calculate_conflict_potential(
        self,
        thesis: list[ScoredPassage],
        antithesis: list[ScoredPassage],
    ) -> float:
        """Estimate potential for conflicting information."""
        if not thesis or not antithesis:
            return 0.0

        # Higher overlap in top results suggests more conflict potential
        thesis_ids = set(p.id for p in thesis[:5])
        antithesis_ids = set(p.id for p in antithesis[:5])

        overlap = len(thesis_ids & antithesis_ids)

        # If same passages appear in both, there's less conflict
        # If different passages, there's more potential for conflict
        non_overlap = len(thesis_ids | antithesis_ids) - overlap

        if non_overlap == 0:
            return 0.0

        return (non_overlap - overlap) / non_overlap

    def _assign_diversity_buckets(
        self,
        passages: list[ScoredPassage],
        num_buckets: int = 5,
    ) -> list[ScoredPassage]:
        """Assign diversity buckets based on source clustering."""
        if not passages:
            return passages

        # Simple bucketing by source
        source_to_bucket: dict[str, int] = {}
        bucket_counter = 0

        for passage in passages:
            if passage.source_id not in source_to_bucket:
                source_to_bucket[passage.source_id] = bucket_counter % num_buckets
                bucket_counter += 1

            passage.diversity_bucket = source_to_bucket[passage.source_id]

        return passages

    def index_chunks(self, namespace: str | None = None) -> int:
        """
        Index all chunks from the chunk store.

        Args:
            namespace: Namespace to index (uses retriever's namespace if None)

        Returns:
            Number of chunks indexed
        """
        namespace = namespace or self.namespace
        chunk_store = get_chunk_store()

        # Collect all chunks
        chunks = list(chunk_store.get_chunks_by_namespace(namespace))

        if not chunks:
            logger.warning("no_chunks_to_index", namespace=namespace)
            return 0

        # Prepare data for indexing
        ids = [c.id for c in chunks]
        texts = [c.text for c in chunks]
        metadata = [
            {
                "source_id": c.metadata.source_id,
                "document_uri": c.metadata.document_uri,
                "chunk_index": c.metadata.chunk_index,
                "namespace": c.metadata.namespace,
            }
            for c in chunks
        ]

        # Compute embeddings
        model = self._get_embedding_model()
        embeddings = model.encode(
            texts,
            batch_size=self.settings.embedding.batch_size,
            normalize_embeddings=self.settings.embedding.normalize,
            show_progress_bar=True,
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        # Index in dense store
        self.dense_index.add(ids, embeddings, metadata)

        # Index in sparse store
        self.sparse_index.add(ids, texts, metadata)

        logger.info(
            "chunks_indexed",
            namespace=namespace,
            count=len(chunks),
        )

        return len(chunks)

    def save_indices(self) -> None:
        """Save all indices to disk."""
        self._dense_manager.save(self.namespace)
        self._sparse_manager.save(self.namespace)

    def get_stats(self) -> dict[str, Any]:
        """Get retriever statistics."""
        return {
            "namespace": self.namespace,
            "dense_index_size": self.dense_index.size,
            "sparse_index_size": self.sparse_index.size,
            "dense_weight": self.dense_weight,
            "sparse_weight": self.sparse_weight,
        }


def get_retriever(namespace: str = "default") -> HybridRetriever:
    """Get a hybrid retriever for a namespace."""
    return HybridRetriever(namespace=namespace)
