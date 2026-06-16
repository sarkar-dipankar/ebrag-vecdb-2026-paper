"""
Sparse retrieval index implementations.

Provides BM25-based lexical search for capturing keyword matches
and long-tail entities that dense retrieval might miss.
"""

import json
import pickle
import re
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from ebrag.common import get_logger, get_settings
from ebrag.retrieval.models import IndexMetadata

logger = get_logger(__name__)


class SparseIndex(ABC):
    """Abstract base class for sparse retrieval indices."""

    @abstractmethod
    def add(
        self,
        ids: list[str],
        texts: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> int:
        """Add documents to the index. Returns count added."""
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        k: int = 10,
        filter_fn: Any | None = None,
    ) -> list[tuple[str, float]]:
        """Search for k best matches. Returns list of (id, score)."""
        pass

    @abstractmethod
    def delete(self, ids: list[str]) -> int:
        """Delete documents by ID. Returns count deleted."""
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save index to disk."""
        pass

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load index from disk."""
        pass

    @abstractmethod
    def get_metadata(self) -> IndexMetadata:
        """Get index metadata."""
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        """Get number of documents in index."""
        pass


class BM25Index(SparseIndex):
    """
    BM25-based sparse index implementation.

    Uses BM25Okapi for lexical matching with TF-IDF weighting.
    Particularly effective for capturing exact keyword matches
    and domain-specific terminology.
    """

    def __init__(
        self,
        namespace: str,
        k1: float = 1.5,  # Term frequency saturation
        b: float = 0.75,  # Length normalization
        lowercase: bool = True,
        remove_stopwords: bool = True,
    ) -> None:
        self.namespace = namespace
        self.k1 = k1
        self.b = b
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords

        # Document storage
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._tokenized: list[list[str]] = []
        self._metadata: dict[str, dict[str, Any]] = {}

        # BM25 index (rebuilt on changes)
        self._bm25: BM25Okapi | None = None
        self._dirty = False

        # Basic English stopwords
        self._stopwords = {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "shall", "can", "need",
            "it", "its", "this", "that", "these", "those", "i", "you", "he", "she",
            "we", "they", "what", "which", "who", "whom", "whose", "where", "when",
            "why", "how", "all", "each", "every", "both", "few", "more", "most",
            "other", "some", "such", "no", "not", "only", "own", "same", "so",
            "than", "too", "very", "just", "also", "now", "here", "there", "then",
        }

        self.created_at = datetime.utcnow()
        self.updated_at = self.created_at

        logger.info(
            "bm25_index_created",
            namespace=namespace,
            k1=k1,
            b=b,
        )

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text for BM25."""
        if self.lowercase:
            text = text.lower()

        # Simple word tokenization
        tokens = re.findall(r'\b\w+\b', text)

        # Remove stopwords
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self._stopwords]

        return tokens

    def _rebuild_index(self) -> None:
        """Rebuild the BM25 index from tokenized documents."""
        if not self._tokenized:
            self._bm25 = None
            return

        self._bm25 = BM25Okapi(
            self._tokenized,
            k1=self.k1,
            b=self.b,
        )
        self._dirty = False

    def add(
        self,
        ids: list[str],
        texts: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> int:
        """Add documents to the index."""
        if len(ids) != len(texts):
            raise ValueError(f"ID count {len(ids)} != text count {len(texts)}")

        added = 0
        for i, (doc_id, text) in enumerate(zip(ids, texts)):
            # Check if already exists
            if doc_id in self._metadata:
                # Update existing
                idx = self._ids.index(doc_id)
                self._texts[idx] = text
                self._tokenized[idx] = self._tokenize(text)
            else:
                # Add new
                self._ids.append(doc_id)
                self._texts.append(text)
                self._tokenized.append(self._tokenize(text))

            # Store metadata
            if metadata and i < len(metadata):
                self._metadata[doc_id] = metadata[i]
            else:
                self._metadata[doc_id] = {}

            added += 1

        self._dirty = True
        self.updated_at = datetime.utcnow()

        logger.debug(
            "documents_added",
            namespace=self.namespace,
            count=added,
            total=self.size,
        )

        return added

    def search(
        self,
        query: str,
        k: int = 10,
        filter_fn: Any | None = None,
    ) -> list[tuple[str, float]]:
        """Search for k best matches."""
        if not self._ids:
            return []

        # Rebuild index if dirty
        if self._dirty or self._bm25 is None:
            self._rebuild_index()

        if self._bm25 is None:
            return []

        # Tokenize query
        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        # Get BM25 scores
        scores = self._bm25.get_scores(query_tokens)

        # Create (id, score) pairs
        results: list[tuple[str, float]] = []
        for idx, score in enumerate(scores):
            if score > 0:
                doc_id = self._ids[idx]

                # Apply filter if provided
                if filter_fn is not None:
                    meta = self._metadata.get(doc_id, {})
                    if not filter_fn(meta):
                        continue

                results.append((doc_id, float(score)))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:k]

    def delete(self, ids: list[str]) -> int:
        """Delete documents by ID."""
        deleted = 0

        for doc_id in ids:
            if doc_id in self._metadata:
                idx = self._ids.index(doc_id)
                del self._ids[idx]
                del self._texts[idx]
                del self._tokenized[idx]
                del self._metadata[doc_id]
                deleted += 1

        if deleted > 0:
            self._dirty = True
            self.updated_at = datetime.utcnow()

        return deleted

    def save(self, path: Path) -> None:
        """Save index to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save documents and tokens
        data = {
            "ids": self._ids,
            "texts": self._texts,
            "tokenized": self._tokenized,
            "metadata": self._metadata,
            "config": {
                "namespace": self.namespace,
                "k1": self.k1,
                "b": self.b,
                "lowercase": self.lowercase,
                "remove_stopwords": self.remove_stopwords,
                "created_at": self.created_at.isoformat(),
                "updated_at": self.updated_at.isoformat(),
            },
        }

        # Use pickle for efficiency with large data
        data_path = path / "bm25_data.pkl"
        with open(data_path, "wb") as f:
            pickle.dump(data, f)

        logger.info("index_saved", path=str(path), documents=self.size)

    def load(self, path: Path) -> None:
        """Load index from disk."""
        path = Path(path)

        data_path = path / "bm25_data.pkl"
        if not data_path.exists():
            raise FileNotFoundError(f"No BM25 index found at {path}")

        with open(data_path, "rb") as f:
            data = pickle.load(f)

        self._ids = data.get("ids", [])
        self._texts = data.get("texts", [])
        self._tokenized = data.get("tokenized", [])
        self._metadata = data.get("metadata", {})

        config = data.get("config", {})
        self.k1 = config.get("k1", self.k1)
        self.b = config.get("b", self.b)
        self.lowercase = config.get("lowercase", self.lowercase)
        self.remove_stopwords = config.get("remove_stopwords", self.remove_stopwords)

        if "created_at" in config:
            self.created_at = datetime.fromisoformat(config["created_at"])
        if "updated_at" in config:
            self.updated_at = datetime.fromisoformat(config["updated_at"])

        self._dirty = True  # Need to rebuild BM25 on first search

        logger.info("index_loaded", path=str(path), documents=self.size)

    def get_metadata(self) -> IndexMetadata:
        """Get index metadata."""
        return IndexMetadata(
            index_id=f"bm25-{self.namespace}",
            namespace=self.namespace,
            index_type="sparse",
            backend="bm25",
            num_vectors=self.size,
            dimension=None,
            created_at=self.created_at,
            updated_at=self.updated_at,
            config={
                "k1": self.k1,
                "b": self.b,
                "lowercase": self.lowercase,
                "remove_stopwords": self.remove_stopwords,
            },
        )

    def get_document(self, id: str) -> tuple[str, dict[str, Any]] | None:
        """Get document text and metadata by ID."""
        if id not in self._metadata:
            return None

        idx = self._ids.index(id)
        return self._texts[idx], self._metadata[id]

    @property
    def size(self) -> int:
        """Get number of documents in index."""
        return len(self._ids)


class SparseIndexManager:
    """
    Manager for sparse indices across namespaces.

    Handles index creation, persistence, and lifecycle.
    """

    def __init__(self, base_path: Path | None = None) -> None:
        settings = get_settings()
        self.base_path = base_path or settings.retrieval.index_path / "sparse"
        self.base_path.mkdir(parents=True, exist_ok=True)

        self._indices: dict[str, SparseIndex] = {}
        self.settings = settings

    def get_or_create(self, namespace: str) -> SparseIndex:
        """Get existing index or create new one."""
        if namespace in self._indices:
            return self._indices[namespace]

        # Check for saved index
        index_path = self.base_path / namespace
        if index_path.exists():
            return self.load(namespace)

        # Create new index
        index = BM25Index(namespace=namespace)
        self._indices[namespace] = index
        return index

    def load(self, namespace: str) -> SparseIndex:
        """Load an index from disk."""
        index_path = self.base_path / namespace

        if not index_path.exists():
            raise FileNotFoundError(f"No index found for namespace: {namespace}")

        index = BM25Index(namespace=namespace)
        index.load(index_path)

        self._indices[namespace] = index
        return index

    def save(self, namespace: str) -> None:
        """Save an index to disk."""
        if namespace not in self._indices:
            raise KeyError(f"No index loaded for namespace: {namespace}")

        index = self._indices[namespace]
        index_path = self.base_path / namespace
        index.save(index_path)

    def save_all(self) -> None:
        """Save all loaded indices."""
        for namespace in self._indices:
            self.save(namespace)

    def delete(self, namespace: str) -> None:
        """Delete an index."""
        if namespace in self._indices:
            del self._indices[namespace]

        index_path = self.base_path / namespace
        if index_path.exists():
            import shutil
            shutil.rmtree(index_path)

    def list_namespaces(self) -> list[str]:
        """List all available namespaces."""
        namespaces = set(self._indices.keys())

        if self.base_path.exists():
            for path in self.base_path.iterdir():
                if path.is_dir():
                    namespaces.add(path.name)

        return sorted(namespaces)


# Global manager instance
_manager: SparseIndexManager | None = None


def get_sparse_index_manager() -> SparseIndexManager:
    """Get the global sparse index manager."""
    global _manager
    if _manager is None:
        _manager = SparseIndexManager()
    return _manager
