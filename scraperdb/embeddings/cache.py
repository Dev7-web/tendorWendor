# embeddings/cache.py
"""
Caching layer for Docling outputs and embeddings.
Provides file-based and MongoDB-based caching to avoid redundant processing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Configuration
CACHE_DIR = os.getenv("DOCLING_CACHE_DIR", "")
ENABLE_FILE_CACHE = os.getenv("ENABLE_DOCLING_CACHE", "true").lower() in ("true", "1", "yes")
CACHE_TTL_HOURS = int(os.getenv("CACHE_TTL_HOURS", "168"))  # 1 week default


def compute_file_hash(file_path: str) -> str:
    """
    Compute SHA256 hash of a file for cache keying.

    Args:
        file_path: Path to the file

    Returns:
        Hex string of SHA256 hash
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def compute_text_hash(text: str) -> str:
    """
    Compute SHA256 hash of text content.

    Args:
        text: Text content to hash

    Returns:
        Hex string of SHA256 hash
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class FileCache:
    """
    File-based cache for Docling outputs.
    Stores JSON files in a cache directory.
    """

    def __init__(self, cache_dir: Optional[str] = None, ttl_hours: int = 168):
        """
        Initialize file cache.

        Args:
            cache_dir: Directory for cache files (uses env var if not provided)
            ttl_hours: Cache TTL in hours (default 168 = 1 week)
        """
        self.cache_dir = cache_dir or CACHE_DIR
        self.ttl_hours = ttl_hours
        self.enabled = ENABLE_FILE_CACHE and bool(self.cache_dir)

        if self.enabled:
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info(f"File cache enabled at: {self.cache_dir}")

    def _cache_path(self, key: str) -> str:
        """Get cache file path for a key."""
        return os.path.join(self.cache_dir, f"{key}.json")

    def _is_expired(self, cache_file: str) -> bool:
        """Check if cache file is expired."""
        if self.ttl_hours <= 0:
            return False
        try:
            mtime = os.path.getmtime(cache_file)
            age_hours = (time.time() - mtime) / 3600
            return age_hours > self.ttl_hours
        except Exception:
            return True

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached data by key.

        Args:
            key: Cache key (usually file hash)

        Returns:
            Cached data dict or None if not found/expired
        """
        if not self.enabled:
            return None

        cache_file = self._cache_path(key)
        if not os.path.exists(cache_file):
            return None

        if self._is_expired(cache_file):
            logger.debug(f"Cache expired for {key}")
            try:
                os.remove(cache_file)
            except Exception:
                pass
            return None

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.debug(f"Cache hit for {key}")
            return data
        except Exception as e:
            logger.warning(f"Failed to load cache for {key}: {e}")
            return None

    def set(self, key: str, data: Dict[str, Any]) -> bool:
        """
        Store data in cache.

        Args:
            key: Cache key
            data: Data to cache (must be JSON serializable)

        Returns:
            True if cached successfully
        """
        if not self.enabled:
            return False

        cache_file = self._cache_path(key)
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f)
            logger.debug(f"Cached data for {key}")
            return True
        except Exception as e:
            logger.warning(f"Failed to cache data for {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """
        Delete cached data.

        Args:
            key: Cache key

        Returns:
            True if deleted successfully
        """
        if not self.enabled:
            return False

        cache_file = self._cache_path(key)
        try:
            if os.path.exists(cache_file):
                os.remove(cache_file)
            return True
        except Exception as e:
            logger.warning(f"Failed to delete cache for {key}: {e}")
            return False

    def clear(self) -> int:
        """
        Clear all cached data.

        Returns:
            Number of files deleted
        """
        if not self.enabled or not os.path.exists(self.cache_dir):
            return 0

        count = 0
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith(".json"):
                    os.remove(os.path.join(self.cache_dir, filename))
                    count += 1
        except Exception as e:
            logger.warning(f"Error clearing cache: {e}")

        logger.info(f"Cleared {count} cached files")
        return count

    def cleanup_expired(self) -> int:
        """
        Remove expired cache entries.

        Returns:
            Number of expired files removed
        """
        if not self.enabled or not os.path.exists(self.cache_dir):
            return 0

        count = 0
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith(".json"):
                    filepath = os.path.join(self.cache_dir, filename)
                    if self._is_expired(filepath):
                        os.remove(filepath)
                        count += 1
        except Exception as e:
            logger.warning(f"Error cleaning up cache: {e}")

        if count > 0:
            logger.info(f"Cleaned up {count} expired cache files")
        return count


class EmbeddingCache:
    """
    MongoDB-based cache for embeddings.
    Uses the existing MongoDB collections for caching.
    """

    def __init__(self, db=None):
        """
        Initialize embedding cache.

        Args:
            db: MongoDB database instance (uses get_db() if not provided)
        """
        self._db = db

    @property
    def db(self):
        """Lazy load database connection."""
        if self._db is None:
            from api.services.mongo import get_db
            self._db = get_db()
        return self._db

    def get_profile_embedding(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get cached profile embedding by file hash.

        Args:
            file_hash: File hash

        Returns:
            Embedding data dict or None
        """
        try:
            collection = self.db["company_profile_embeddings"]
            doc = collection.find_one({"file_hash": file_hash})
            if doc and doc.get("summary_embedding"):
                return {
                    "summary_embedding": doc["summary_embedding"],
                    "representative_embeddings": doc.get("representative_embeddings", []),
                    "representative_chunks": doc.get("representative_chunks", []),
                    "chunk_count": doc.get("chunk_count", 0),
                    "profile_data": doc.get("profile_data"),
                    "filter_metadata": doc.get("filter_metadata"),
                }
            return None
        except Exception as e:
            logger.warning(f"Failed to get cached embedding for {file_hash}: {e}")
            return None

    def get_docling_output(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get cached Docling output by file hash.

        Args:
            file_hash: File hash

        Returns:
            Docling output dict or None
        """
        try:
            collection = self.db["company_profile_outputs"]
            doc = collection.find_one({"file_hash": file_hash})
            if doc and doc.get("text"):
                return {
                    "text": doc.get("text", ""),
                    "relevant_text": doc.get("relevant_text"),
                    "tables": doc.get("tables", []),
                    "sections": doc.get("sections", []),
                    "extraction_stats": doc.get("extraction_stats"),
                }
            return None
        except Exception as e:
            logger.warning(f"Failed to get cached Docling output for {file_hash}: {e}")
            return None


# Global cache instances
_file_cache: Optional[FileCache] = None
_embedding_cache: Optional[EmbeddingCache] = None


def get_file_cache() -> FileCache:
    """Get global file cache instance."""
    global _file_cache
    if _file_cache is None:
        _file_cache = FileCache()
    return _file_cache


def get_embedding_cache() -> EmbeddingCache:
    """Get global embedding cache instance."""
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache()
    return _embedding_cache


def cache_docling_output(file_hash: str, output: Dict[str, Any]) -> bool:
    """
    Cache Docling output using both file and MongoDB cache.

    Args:
        file_hash: File hash
        output: Docling output to cache

    Returns:
        True if cached successfully
    """
    file_cache = get_file_cache()

    # Prepare cacheable data (remove non-serializable items)
    cacheable = {
        "file_hash": file_hash,
        "text": output.get("text", ""),
        "relevant_text": output.get("relevant_text"),
        "tables": output.get("tables", []),
        "sections": output.get("sections", []),
        "extraction_stats": output.get("extraction_stats"),
        "cached_at": datetime.now(timezone.utc).isoformat(),
    }

    return file_cache.set(file_hash, cacheable)


def get_cached_docling_output(file_hash: str) -> Optional[Dict[str, Any]]:
    """
    Get cached Docling output, checking file cache first then MongoDB.

    Args:
        file_hash: File hash

    Returns:
        Cached output or None
    """
    # Try file cache first (faster)
    file_cache = get_file_cache()
    cached = file_cache.get(file_hash)
    if cached:
        return cached

    # Fall back to MongoDB cache
    embedding_cache = get_embedding_cache()
    return embedding_cache.get_docling_output(file_hash)


def get_cached_embedding(file_hash: str) -> Optional[Dict[str, Any]]:
    """
    Get cached embedding from MongoDB.

    Args:
        file_hash: File hash

    Returns:
        Cached embedding or None
    """
    embedding_cache = get_embedding_cache()
    return embedding_cache.get_profile_embedding(file_hash)
