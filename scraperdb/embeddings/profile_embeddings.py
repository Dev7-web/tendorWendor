# embeddings/profile_embeddings.py
"""
Multi-vector profile representation.
Instead of using mean pooling (which loses information), this module creates
multiple representative embeddings that capture diverse aspects of a company profile.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from embeddings.tender_embedder import TenderEmbedder
    from embeddings.chunker import chunk_text
except ModuleNotFoundError:
    from tender_embedder import TenderEmbedder
    from chunker import chunk_text

logger = logging.getLogger(__name__)


def _normalize_vector(vec: List[float]) -> List[float]:
    """Normalize a vector to unit length."""
    norm = math.sqrt(sum(float(v) * float(v) for v in vec))
    if norm <= 0:
        return vec
    return [float(v) / norm for v in vec]


def _mean_vectors(vectors: List[List[float]]) -> Optional[List[float]]:
    """Compute mean of vectors and normalize."""
    if not vectors:
        return None
    arr = np.array(vectors)
    mean = np.mean(arr, axis=0)
    return _normalize_vector(mean.tolist())


def _max_pool_vectors(vectors: List[List[float]]) -> Optional[List[float]]:
    """Compute element-wise max of vectors (captures strongest signals)."""
    if not vectors:
        return None
    arr = np.array(vectors)
    max_vec = np.max(arr, axis=0)
    return _normalize_vector(max_vec.tolist())


def _cosine_distance(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine distance between two vectors."""
    arr1 = np.array(vec1)
    arr2 = np.array(vec2)
    dot = np.dot(arr1, arr2)
    norm1 = np.linalg.norm(arr1)
    norm2 = np.linalg.norm(arr2)
    if norm1 == 0 or norm2 == 0:
        return 1.0
    similarity = dot / (norm1 * norm2)
    return 1.0 - similarity


def _simple_kmeans(
    vectors: np.ndarray,
    n_clusters: int,
    max_iterations: int = 100,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple K-means implementation to avoid sklearn dependency.
    Returns (cluster_centers, labels).
    """
    np.random.seed(seed)
    n_samples = vectors.shape[0]

    if n_clusters >= n_samples:
        # Return each vector as its own cluster
        return vectors.copy(), np.arange(n_samples)

    # Initialize centroids randomly
    indices = np.random.choice(n_samples, n_clusters, replace=False)
    centroids = vectors[indices].copy()

    labels = np.zeros(n_samples, dtype=int)

    for _ in range(max_iterations):
        # Assign points to nearest centroid
        for i in range(n_samples):
            distances = np.linalg.norm(vectors[i] - centroids, axis=1)
            labels[i] = np.argmin(distances)

        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for k in range(n_clusters):
            cluster_points = vectors[labels == k]
            if len(cluster_points) > 0:
                new_centroids[k] = cluster_points.mean(axis=0)
            else:
                new_centroids[k] = centroids[k]

        # Check for convergence
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return centroids, labels


def select_representative_embeddings(
    embeddings: List[List[float]],
    chunks: List[str],
    n_representatives: int = 5,
) -> Dict[str, Any]:
    """
    Select representative embeddings using clustering.

    This finds diverse representative vectors that capture different aspects
    of the company profile, rather than a single mean that dilutes everything.

    Args:
        embeddings: List of chunk embeddings
        chunks: List of text chunks (same order as embeddings)
        n_representatives: Number of representative embeddings to select

    Returns:
        Dictionary with representative embeddings, chunks, and indices
    """
    if not embeddings or not chunks:
        return {
            "representative_embeddings": [],
            "representative_chunks": [],
            "representative_indices": [],
        }

    n_clusters = min(n_representatives, len(embeddings))
    embeddings_np = np.array(embeddings)

    # Use simple k-means to find clusters
    centroids, labels = _simple_kmeans(embeddings_np, n_clusters)

    # For each cluster, find the chunk closest to the centroid
    representative_indices: List[int] = []
    representative_embeddings: List[List[float]] = []
    representative_chunks: List[str] = []

    for cluster_id in range(n_clusters):
        # Get indices of points in this cluster
        cluster_mask = labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]

        if len(cluster_indices) == 0:
            continue

        cluster_embeddings = embeddings_np[cluster_mask]

        # Find the point closest to the centroid
        distances = np.linalg.norm(cluster_embeddings - centroids[cluster_id], axis=1)
        closest_in_cluster = np.argmin(distances)
        closest_idx = cluster_indices[closest_in_cluster]

        representative_indices.append(int(closest_idx))
        representative_embeddings.append(embeddings[closest_idx])
        representative_chunks.append(chunks[closest_idx])

    return {
        "representative_embeddings": representative_embeddings,
        "representative_chunks": representative_chunks,
        "representative_indices": representative_indices,
    }


def create_profile_representation(
    chunks: List[str],
    embedder: TenderEmbedder,
    n_representatives: int = 5,
) -> Dict[str, Any]:
    """
    Create a rich multi-vector profile representation.

    This is the main function to create profile embeddings that capture
    the diverse aspects of a company rather than a single blurry mean.

    Args:
        chunks: List of text chunks from the profile
        embedder: TenderEmbedder instance
        n_representatives: Number of representative embeddings

    Returns:
        Dictionary containing:
        - representative_embeddings: List of diverse representative embeddings
        - representative_chunks: Corresponding text chunks
        - mean_embedding: Traditional mean pooling (for compatibility)
        - max_embedding: Max pooling (captures strongest signals)
        - chunk_count: Total number of chunks
    """
    if not chunks:
        return {
            "representative_embeddings": [],
            "representative_chunks": [],
            "mean_embedding": None,
            "max_embedding": None,
            "chunk_count": 0,
        }

    logger.info(f"Creating profile representation from {len(chunks)} chunks")

    # Generate embeddings for all chunks
    embeddings = embedder.embed(chunks)

    if not embeddings:
        return {
            "representative_embeddings": [],
            "representative_chunks": [],
            "mean_embedding": None,
            "max_embedding": None,
            "chunk_count": 0,
        }

    # Select representative embeddings using clustering
    representatives = select_representative_embeddings(
        embeddings=embeddings,
        chunks=chunks,
        n_representatives=n_representatives,
    )

    # Also compute traditional mean and max pooling for compatibility
    mean_embedding = _mean_vectors(embeddings)
    max_embedding = _max_pool_vectors(embeddings)

    return {
        "representative_embeddings": representatives["representative_embeddings"],
        "representative_chunks": representatives["representative_chunks"],
        "representative_indices": representatives["representative_indices"],
        "mean_embedding": mean_embedding,
        "max_embedding": max_embedding,
        "all_embeddings": embeddings,  # Keep all for storage in vector DB
        "chunk_count": len(chunks),
    }


def create_profile_from_text(
    text: str,
    embedder: Optional[TenderEmbedder] = None,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    n_representatives: int = 5,
) -> Dict[str, Any]:
    """
    Convenience function to create profile representation from raw text.

    Args:
        text: Raw profile text
        embedder: Optional TenderEmbedder (creates one if not provided)
        chunk_size: Maximum chunk size in characters
        chunk_overlap: Overlap between chunks
        n_representatives: Number of representative embeddings

    Returns:
        Profile representation dictionary
    """
    if embedder is None:
        embedder = TenderEmbedder()

    chunks = chunk_text(text, max_chars=chunk_size, overlap_chars=chunk_overlap)

    if not chunks:
        return {
            "chunks": [],
            "representative_embeddings": [],
            "representative_chunks": [],
            "mean_embedding": None,
            "max_embedding": None,
            "chunk_count": 0,
        }

    representation = create_profile_representation(
        chunks=chunks,
        embedder=embedder,
        n_representatives=n_representatives,
    )

    representation["chunks"] = chunks
    return representation


def compute_multi_vector_similarity(
    profile_repr: Dict[str, Any],
    query_embedding: List[float],
    aggregation: str = "max",
) -> float:
    """
    Compute similarity between a profile representation and a query embedding.

    Args:
        profile_repr: Profile representation from create_profile_representation
        query_embedding: Query embedding vector
        aggregation: How to aggregate scores ("max", "mean", "weighted")

    Returns:
        Similarity score (0-1, higher is better)
    """
    rep_embeddings = profile_repr.get("representative_embeddings", [])

    if not rep_embeddings:
        # Fall back to mean embedding
        mean_emb = profile_repr.get("mean_embedding")
        if mean_emb:
            return 1.0 - _cosine_distance(mean_emb, query_embedding)
        return 0.0

    # Compute similarity with each representative
    similarities = []
    for rep_emb in rep_embeddings:
        sim = 1.0 - _cosine_distance(rep_emb, query_embedding)
        similarities.append(sim)

    if aggregation == "max":
        return max(similarities)
    elif aggregation == "mean":
        return sum(similarities) / len(similarities)
    elif aggregation == "weighted":
        # Weight higher similarities more
        weights = [s ** 2 for s in similarities]
        weighted_sum = sum(s * w for s, w in zip(similarities, weights))
        return weighted_sum / max(sum(weights), 1e-10)
    else:
        return max(similarities)


class ProfileRepresentationBuilder:
    """
    Builder class for creating profile representations with caching.
    """

    def __init__(
        self,
        embedder: Optional[TenderEmbedder] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 80,
        n_representatives: int = 5,
    ):
        self.embedder = embedder or TenderEmbedder()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.n_representatives = n_representatives
        self._cache: Dict[str, Dict[str, Any]] = {}

    def build(self, text: str, cache_key: Optional[str] = None) -> Dict[str, Any]:
        """Build profile representation, optionally caching by key."""
        if cache_key and cache_key in self._cache:
            return self._cache[cache_key]

        representation = create_profile_from_text(
            text=text,
            embedder=self.embedder,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            n_representatives=self.n_representatives,
        )

        if cache_key:
            self._cache[cache_key] = representation

        return representation

    def clear_cache(self):
        """Clear the representation cache."""
        self._cache.clear()
