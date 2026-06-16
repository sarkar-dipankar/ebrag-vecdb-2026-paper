"""
Dense vector index implementations.

Provides pluggable backends for approximate nearest neighbor search.
Currently supports USearch as the primary backend.
"""

import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ebrag.common import get_logger, get_settings
from ebrag.retrieval.models import IndexMetadata, ScoredPassage

logger = get_logger(__name__)


class DenseIndex(ABC):
    """Abstract base class for dense vector indices."""

    @abstractmethod
    def add(
        self,
        ids: list[str],
        vectors: np.ndarray,
        metadata: list[dict[str, Any]] | None = None,
    ) -> int:
        """Add vectors to the index. Returns count added."""
        pass

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filter_fn: Any | None = None,
    ) -> list[tuple[str, float]]:
        """Search for k nearest neighbors. Returns list of (id, score)."""
        pass

    @abstractmethod
    def delete(self, ids: list[str]) -> int:
        """Delete vectors by ID. Returns count deleted."""
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
        """Get number of vectors in index."""
        pass


class USearchIndex(DenseIndex):
    """
    Dense index implementation using USearch.

    USearch is a fast, portable vector search library supporting
    various distance metrics and quantization options.
    """

    def __init__(
        self,
        namespace: str,
        dimension: int,
        metric: str = "cos",  # cos, l2, ip (inner product)
        dtype: str = "f32",   # f32, f16, i8
        connectivity: int = 16,  # M parameter for HNSW
        expansion_add: int = 128,  # ef_construction
        expansion_search: int = 64,  # ef_search
    ) -> None:
        from usearch.index import Index, MetricKind, ScalarKind

        self.namespace = namespace
        self.dimension = dimension
        self.metric = metric
        self.dtype = dtype
        self.connectivity = connectivity
        self.expansion_add = expansion_add
        self.expansion_search = expansion_search

        # Map metric names to USearch enums
        metric_map = {
            "cos": MetricKind.Cos,
            "l2": MetricKind.L2sq,
            "ip": MetricKind.IP,
        }

        # Map dtype names to USearch enums
        dtype_map = {
            "f32": ScalarKind.F32,
            "f16": ScalarKind.F16,
            "i8": ScalarKind.I8,
        }

        self.index = Index(
            ndim=dimension,
            metric=metric_map.get(metric, MetricKind.Cos),
            dtype=dtype_map.get(dtype, ScalarKind.F32),
            connectivity=connectivity,
            expansion_add=expansion_add,
            expansion_search=expansion_search,
        )

        # Store mapping from string IDs to integer keys
        self._id_to_key: dict[str, int] = {}
        self._key_to_id: dict[int, str] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._next_key: int = 0

        self.created_at = datetime.utcnow()
        self.updated_at = self.created_at

        logger.info(
            "usearch_index_created",
            namespace=namespace,
            dimension=dimension,
            metric=metric,
        )

    def add(
        self,
        ids: list[str],
        vectors: np.ndarray,
        metadata: list[dict[str, Any]] | None = None,
    ) -> int:
        """Add vectors to the index."""
        if len(ids) != vectors.shape[0]:
            raise ValueError(f"ID count {len(ids)} != vector count {vectors.shape[0]}")

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} != index dimension {self.dimension}"
            )

        # Convert to float32 for indexing
        vectors = vectors.astype(np.float32)

        # Assign integer keys to string IDs
        keys = []
        for i, str_id in enumerate(ids):
            if str_id in self._id_to_key:
                # Update existing
                key = self._id_to_key[str_id]
            else:
                key = self._next_key
                self._next_key += 1
                self._id_to_key[str_id] = key
                self._key_to_id[key] = str_id

            keys.append(key)

            # Store metadata if provided
            if metadata and i < len(metadata):
                self._metadata[str_id] = metadata[i]

        # Add to USearch index
        keys_array = np.array(keys, dtype=np.uint64)
        self.index.add(keys_array, vectors)

        self.updated_at = datetime.utcnow()

        logger.debug(
            "vectors_added",
            namespace=self.namespace,
            count=len(ids),
            total=self.size,
        )

        return len(ids)

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filter_fn: Any | None = None,
    ) -> list[tuple[str, float]]:
        """Search for k nearest neighbors."""
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        query_vector = query_vector.astype(np.float32)

        # Search the index
        matches = self.index.search(query_vector, k)

        results: list[tuple[str, float]] = []

        # USearch returns matches object with keys and distances
        for key, distance in zip(matches.keys[0], matches.distances[0]):
            if key in self._key_to_id:
                str_id = self._key_to_id[key]

                # Apply filter if provided
                if filter_fn is not None:
                    meta = self._metadata.get(str_id, {})
                    if not filter_fn(meta):
                        continue

                # Convert distance to similarity score
                # For cosine, distance is 1 - similarity
                if self.metric == "cos":
                    score = 1.0 - float(distance)
                else:
                    # For L2, use inverse
                    score = 1.0 / (1.0 + float(distance))

                results.append((str_id, score))

        return results

    def delete(self, ids: list[str]) -> int:
        """Delete vectors by ID."""
        # USearch doesn't support deletion directly in all versions
        # We mark as deleted and rebuild periodically
        deleted = 0
        for str_id in ids:
            if str_id in self._id_to_key:
                key = self._id_to_key[str_id]
                del self._id_to_key[str_id]
                del self._key_to_id[key]
                if str_id in self._metadata:
                    del self._metadata[str_id]
                deleted += 1

        self.updated_at = datetime.utcnow()
        return deleted

    def save(self, path: Path) -> None:
        """Save index to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save USearch index
        index_path = path / "index.usearch"
        self.index.save(str(index_path))

        # Save ID mappings and metadata
        mappings = {
            "id_to_key": self._id_to_key,
            "key_to_id": {str(k): v for k, v in self._key_to_id.items()},
            "metadata": self._metadata,
            "next_key": self._next_key,
            "config": {
                "namespace": self.namespace,
                "dimension": self.dimension,
                "metric": self.metric,
                "dtype": self.dtype,
                "connectivity": self.connectivity,
                "expansion_add": self.expansion_add,
                "expansion_search": self.expansion_search,
                "created_at": self.created_at.isoformat(),
                "updated_at": self.updated_at.isoformat(),
            },
        }

        mappings_path = path / "mappings.json"
        with open(mappings_path, "w") as f:
            json.dump(mappings, f)

        logger.info("index_saved", path=str(path), vectors=self.size)

    def load(self, path: Path) -> None:
        """Load index from disk."""
        path = Path(path)

        # Load USearch index
        index_path = path / "index.usearch"
        if index_path.exists():
            self.index.load(str(index_path))

        # Load ID mappings and metadata
        mappings_path = path / "mappings.json"
        if mappings_path.exists():
            with open(mappings_path) as f:
                mappings = json.load(f)

            self._id_to_key = mappings.get("id_to_key", {})
            self._key_to_id = {int(k): v for k, v in mappings.get("key_to_id", {}).items()}
            self._metadata = mappings.get("metadata", {})
            self._next_key = mappings.get("next_key", 0)

            config = mappings.get("config", {})
            if "created_at" in config:
                self.created_at = datetime.fromisoformat(config["created_at"])
            if "updated_at" in config:
                self.updated_at = datetime.fromisoformat(config["updated_at"])

        logger.info("index_loaded", path=str(path), vectors=self.size)

    def get_metadata(self) -> IndexMetadata:
        """Get index metadata."""
        return IndexMetadata(
            index_id=f"usearch-{self.namespace}",
            namespace=self.namespace,
            index_type="dense",
            backend="usearch",
            num_vectors=self.size,
            dimension=self.dimension,
            created_at=self.created_at,
            updated_at=self.updated_at,
            config={
                "metric": self.metric,
                "dtype": self.dtype,
                "connectivity": self.connectivity,
            },
        )

    def get_vector_metadata(self, id: str) -> dict[str, Any] | None:
        """Get metadata for a specific vector."""
        return self._metadata.get(id)

    @property
    def size(self) -> int:
        """Get number of vectors in index."""
        return len(self._id_to_key)


class DenseIndexManager:
    """
    Manager for dense indices across namespaces.

    Handles index creation, persistence, and lifecycle.
    """

    def __init__(self, base_path: Path | None = None) -> None:
        settings = get_settings()
        self.base_path = base_path or settings.retrieval.index_path / "dense"
        self.base_path.mkdir(parents=True, exist_ok=True)

        self._indices: dict[str, DenseIndex] = {}
        self.settings = settings

    def get_or_create(
        self,
        namespace: str,
        dimension: int | None = None,
    ) -> DenseIndex:
        """Get existing index or create new one."""
        if namespace in self._indices:
            return self._indices[namespace]

        # Check for saved index
        index_path = self.base_path / namespace
        if index_path.exists():
            return self.load(namespace)

        # Create new index
        dim = dimension or self.settings.embedding.dimension

        index = USearchIndex(
            namespace=namespace,
            dimension=dim,
            metric="cos",
        )

        self._indices[namespace] = index
        return index

    def load(self, namespace: str) -> DenseIndex:
        """Load an index from disk."""
        index_path = self.base_path / namespace

        if not index_path.exists():
            raise FileNotFoundError(f"No index found for namespace: {namespace}")

        # Read config to determine dimension
        mappings_path = index_path / "mappings.json"
        if mappings_path.exists():
            with open(mappings_path) as f:
                mappings = json.load(f)
            config = mappings.get("config", {})
            dimension = config.get("dimension", self.settings.embedding.dimension)
        else:
            dimension = self.settings.embedding.dimension

        index = USearchIndex(namespace=namespace, dimension=dimension)
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
_manager: DenseIndexManager | None = None


def get_dense_index_manager() -> DenseIndexManager:
    """Get the global dense index manager."""
    global _manager
    if _manager is None:
        _manager = DenseIndexManager()
    return _manager
