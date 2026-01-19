# embeddings/vector_store.py

from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache
from typing import Optional, Tuple

import chromadb
from chromadb.config import Settings

# ✅ Make the path explicit + stable
DEFAULT_CHROMA_PATH = os.getenv(
    "CHROMA_PATH",
    str(Path(__file__).resolve().parents[1] / "data" / "chroma"),
)
DEFAULT_COLLECTION = os.getenv("CHROMA_COLLECTION", "tender_embeddings")


@lru_cache(maxsize=1)
def get_chroma_client(persist_directory: Optional[str] = None) -> chromadb.Client:
    """
    ✅ Always use PersistentClient for on-disk storage.
    """
    path = persist_directory or DEFAULT_CHROMA_PATH
    os.makedirs(path, exist_ok=True)

    try:
        return chromadb.PersistentClient(
            path=path,
            settings=Settings(
                anonymized_telemetry=False,
            ),
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize PersistentClient at {path!r}") from exc


def get_chroma_collection(
    name: Optional[str] = None,
    persist_directory: Optional[str] = None,
    space: str = "cosine",
) -> chromadb.Collection:
    """
    Get or create a Chroma collection with safe defaults.
    """
    collection_name = name or DEFAULT_COLLECTION
    client = get_chroma_client(persist_directory=persist_directory)

    try:
        return client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": space},
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to get/create Chroma collection {collection_name!r}"
        ) from exc
