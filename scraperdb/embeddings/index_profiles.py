# embeddings/index_profiles.py
"""
Batch index company profile embeddings with multi-vector representation.

This module now implements the optimized pipeline:
1. Smart section extraction (reduces processing by 5-10x)
2. Multi-vector representation (better match quality)
3. Backwards compatible with legacy single-vector profiles
"""

from __future__ import annotations

import argparse
import logging
from typing import Any, Dict, Iterable, List, Optional
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime, timezone
import os

try:
    from embeddings.chunker import chunk_text
    from embeddings.tender_embedder import TenderEmbedder
    from embeddings.vector_store import get_chroma_collection
    from embeddings.smart_extractor import extract_relevant_sections, get_extraction_stats
    from embeddings.profile_embeddings import create_profile_representation
except ModuleNotFoundError:
    from chunker import chunk_text
    from tender_embedder import TenderEmbedder
    from vector_store import get_chroma_collection
    from smart_extractor import extract_relevant_sections, get_extraction_stats
    from profile_embeddings import create_profile_representation

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "tender_db")

BATCH_SIZE = int(os.getenv("PROFILE_BATCH_SIZE", "64"))
CHUNK_SIZE = int(os.getenv("PROFILE_CHUNK_SIZE", "800"))  # Increased for profiles
CHUNK_OVERLAP = int(os.getenv("PROFILE_CHUNK_OVERLAP", "100"))
N_REPRESENTATIVES = int(os.getenv("PROFILE_N_REPRESENTATIVES", "5"))
USE_SMART_EXTRACTION = os.getenv("USE_SMART_EXTRACTION", "true").lower() in ("true", "1", "yes")

PROFILE_COLLECTION_NAME = os.getenv("PROFILE_CHROMA_COLLECTION", "company_profiles")
PROFILE_OUTPUTS_COLLECTION = "company_profile_outputs"
PROFILE_EMBEDDINGS_COLLECTION = "company_profile_embeddings"

mongo = MongoClient(MONGO_URI)
db = mongo[DB_NAME]
profile_outputs = db[PROFILE_OUTPUTS_COLLECTION]
profile_embeddings = db[PROFILE_EMBEDDINGS_COLLECTION]
company_profiles = db["company_profiles"]

collection = get_chroma_collection(name=PROFILE_COLLECTION_NAME)
embedder = TenderEmbedder()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _combine_text_and_tables(doc: dict) -> str:
    """Combine text and table content from profile output."""
    text = doc.get("text", "") or ""
    tables = doc.get("tables", []) or []

    table_texts: List[str] = []
    for table in tables:
        if isinstance(table, dict):
            if "text" in table and isinstance(table["text"], str):
                table_texts.append(table["text"])
            else:
                for v in table.values():
                    if isinstance(v, str) and v.strip():
                        table_texts.append(v.strip())
        else:
            if isinstance(table, str) and table.strip():
                table_texts.append(table.strip())

    return "\n".join([text] + table_texts).strip()


def _batch_items(items: List[str], batch_size: int) -> List[List[str]]:
    """Split items into batches."""
    if batch_size <= 0:
        return [items]
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _normalize_vector(vec: List[float]) -> List[float]:
    """Normalize vector to unit length."""
    norm = 0.0
    for v in vec:
        norm += float(v) * float(v)
    if norm <= 0:
        return vec
    scale = norm ** 0.5
    return [float(v) / scale for v in vec]


def _mean_vectors(vectors: Iterable[List[float]]) -> Optional[List[float]]:
    """Compute mean of vectors and normalize."""
    vectors = list(vectors)
    if not vectors:
        return None
    dim = len(vectors[0])
    mean = [0.0] * dim
    for vec in vectors:
        for i in range(dim):
            mean[i] += float(vec[i])
    mean = [v / len(vectors) for v in mean]
    return _normalize_vector(mean)


def _update_profiles_for_file_hash(file_hash: str) -> None:
    """
    Update company profiles that contain this file hash.
    Now includes multi-vector data aggregation.
    """
    profiles = list(company_profiles.find({"file_hashes": file_hash}))
    for profile in profiles:
        file_hashes = profile.get("file_hashes") or []
        if not file_hashes:
            continue

        # Get all embeddings for this profile's files
        summaries = list(
            profile_embeddings.find(
                {"file_hash": {"$in": file_hashes}},
                {
                    "summary_embedding": 1,
                    "representative_embeddings": 1,
                    "representative_chunks": 1,
                },
            )
        )

        # Collect mean embeddings
        mean_vectors = [s["summary_embedding"] for s in summaries if s.get("summary_embedding")]
        profile_embedding = _mean_vectors(mean_vectors)

        if not profile_embedding:
            continue

        # Collect all representative embeddings and chunks
        all_rep_embeddings: List[List[float]] = []
        all_rep_chunks: List[str] = []

        for summary in summaries:
            rep_embs = summary.get("representative_embeddings", [])
            rep_chunks = summary.get("representative_chunks", [])
            if rep_embs:
                all_rep_embeddings.extend(rep_embs)
            if rep_chunks:
                all_rep_chunks.extend(rep_chunks)

        update_data: Dict[str, Any] = {
            "status": "READY",
            "profile_embedding": profile_embedding,
            "updated_at": datetime.now(timezone.utc),
        }

        # Add multi-vector data if available
        if all_rep_embeddings:
            update_data["representative_embeddings"] = all_rep_embeddings
            update_data["representative_chunks"] = all_rep_chunks
            update_data["embedding_version"] = "v2_multi_vector"

        company_profiles.update_one(
            {"_id": profile["_id"]},
            {"$set": update_data},
        )


def index_pending_profiles(limit: int = 10, use_multi_vector: bool = True) -> None:
    """
    Index pending profile outputs with optional multi-vector representation.

    Args:
        limit: Maximum number of profiles to process
        use_multi_vector: Whether to use multi-vector representation (default True)
    """
    query = {"doc_type": "profile", "indexed": {"$ne": True}}
    if limit and limit > 0:
        pending = profile_outputs.find(query, limit=limit)
    else:
        pending = profile_outputs.find(query)

    for doc in pending:
        file_hash = doc.get("file_hash")
        if not file_hash:
            logger.warning("Skipping profile output with missing file_hash: %s", doc.get("_id"))
            continue

        # Use smart-extracted text if available, otherwise full text
        if USE_SMART_EXTRACTION and doc.get("relevant_text"):
            combined_text = doc["relevant_text"]
            logger.info("Using smart-extracted text for %s", file_hash)
        else:
            combined_text = _combine_text_and_tables(doc)
            # Apply smart extraction if enabled and we have sections
            if USE_SMART_EXTRACTION and (doc.get("sections") or doc.get("tables")):
                combined_text = extract_relevant_sections(doc)
                stats = get_extraction_stats(doc.get("text", ""), combined_text)
                logger.info(
                    "Smart extraction for %s: %d -> %d pages",
                    file_hash,
                    stats["original_pages_approx"],
                    stats["extracted_pages_approx"],
                )

        if not combined_text:
            logger.warning("Skipping empty profile output: %s", file_hash)
            continue

        chunks = chunk_text(combined_text, max_chars=CHUNK_SIZE, overlap_chars=CHUNK_OVERLAP)
        if not chunks:
            logger.warning("No chunks created for profile output: %s", file_hash)
            continue

        logger.info("Indexing profile output %s -> %d chunks", file_hash, len(chunks))

        try:
            if use_multi_vector:
                # Use multi-vector representation
                _index_with_multi_vector(doc, file_hash, chunks)
            else:
                # Legacy single-vector indexing
                _index_legacy(doc, file_hash, chunks)

            _update_profiles_for_file_hash(file_hash)
            logger.info("Indexed profile output successfully: %s", file_hash)

        except Exception as exc:
            profile_outputs.update_one(
                {"_id": doc["_id"]},
                {
                    "$set": {
                        "indexed": False,
                        "index_error": str(exc),
                        "failed_at": datetime.now(timezone.utc),
                    }
                },
            )
            logger.exception("Failed to index profile output %s", file_hash)


def _index_with_multi_vector(doc: dict, file_hash: str, chunks: List[str]) -> None:
    """
    Index profile with multi-vector representation.
    Creates representative embeddings using clustering.
    """
    # Create multi-vector representation
    representation = create_profile_representation(
        chunks=chunks,
        embedder=embedder,
        n_representatives=N_REPRESENTATIVES,
    )

    all_embeddings = representation.get("all_embeddings", [])
    if not all_embeddings:
        raise RuntimeError("No embeddings generated")

    representative_indices = set(representation.get("representative_indices", []))

    # Store all chunks in Chroma
    for batch_index, batch_start in enumerate(range(0, len(chunks), BATCH_SIZE)):
        batch_end = min(batch_start + BATCH_SIZE, len(chunks))
        batch_chunks = chunks[batch_start:batch_end]
        batch_embeddings = all_embeddings[batch_start:batch_end]

        ids = [f"profile:{file_hash}:{i}" for i in range(batch_start, batch_end)]

        metadatas = [
            {
                "doc_type": "profile",
                "file_hash": file_hash,
                "chunk_index": i,
                "model_name": embedder.model_name,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "is_representative": i in representative_indices,
            }
            for i in range(batch_start, batch_end)
        ]

        collection.upsert(
            ids=ids,
            documents=batch_chunks,
            embeddings=batch_embeddings,
            metadatas=metadatas,
        )

    # Store embedding metadata in MongoDB
    now = datetime.now(timezone.utc)
    profile_embeddings.update_one(
        {"file_hash": file_hash},
        {
            "$set": {
                "file_hash": file_hash,
                "summary_embedding": representation["mean_embedding"],
                "max_embedding": representation.get("max_embedding"),
                "representative_embeddings": representation["representative_embeddings"],
                "representative_chunks": representation["representative_chunks"],
                "representative_indices": list(representative_indices),
                "chunk_count": len(chunks),
                "index_model": embedder.model_name,
                "indexed_at": now,
                "version": "v2_multi_vector",
            }
        },
        upsert=True,
    )

    profile_outputs.update_one(
        {"_id": doc["_id"]},
        {
            "$set": {
                "indexed": True,
                "indexed_at": now,
                "chunk_count": len(chunks),
                "index_model": embedder.model_name,
                "index_version": "v2_multi_vector",
            },
            "$unset": {"index_error": "", "failed_at": ""},
        },
    )


def _index_legacy(doc: dict, file_hash: str, chunks: List[str]) -> None:
    """
    Legacy single-vector indexing for backwards compatibility.
    """
    all_embeddings: List[List[float]] = []

    for batch_index, batch in enumerate(_batch_items(chunks, BATCH_SIZE)):
        batch_embeddings = embedder.embed(batch)
        if len(batch_embeddings) != len(batch):
            raise RuntimeError("Embedding count mismatch")

        all_embeddings.extend(batch_embeddings)

        start_index = batch_index * BATCH_SIZE
        ids = [f"profile:{file_hash}:{i}" for i in range(start_index, start_index + len(batch))]

        metadatas = [
            {
                "doc_type": "profile",
                "file_hash": file_hash,
                "chunk_index": i,
                "model_name": embedder.model_name,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
            }
            for i in range(start_index, start_index + len(batch))
        ]

        collection.upsert(ids=ids, documents=batch, embeddings=batch_embeddings, metadatas=metadatas)

    summary = _mean_vectors(all_embeddings)
    if summary:
        profile_embeddings.update_one(
            {"file_hash": file_hash},
            {
                "$set": {
                    "file_hash": file_hash,
                    "summary_embedding": summary,
                    "chunk_count": len(chunks),
                    "index_model": embedder.model_name,
                    "indexed_at": datetime.now(timezone.utc),
                }
            },
            upsert=True,
        )

    profile_outputs.update_one(
        {"_id": doc["_id"]},
        {
            "$set": {
                "indexed": True,
                "indexed_at": datetime.now(timezone.utc),
                "chunk_count": len(chunks),
                "index_model": embedder.model_name,
            },
            "$unset": {"index_error": "", "failed_at": ""},
        },
    )


def reindex_all_profiles(use_multi_vector: bool = True) -> None:
    """
    Re-index all profiles with the new multi-vector representation.
    Useful for upgrading existing profiles to the new format.
    """
    logger.info("Re-indexing all profiles with multi_vector=%s", use_multi_vector)

    # Reset indexed flag on all profile outputs
    profile_outputs.update_many(
        {"doc_type": "profile"},
        {"$set": {"indexed": False}, "$unset": {"indexed_at": "", "index_version": ""}},
    )

    # Index all
    index_pending_profiles(limit=0, use_multi_vector=use_multi_vector)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10, help="Max profiles to index")
    parser.add_argument("--legacy", action="store_true", help="Use legacy single-vector mode")
    parser.add_argument("--reindex", action="store_true", help="Re-index all profiles")
    args = parser.parse_args()

    if args.reindex:
        reindex_all_profiles(use_multi_vector=not args.legacy)
    else:
        index_pending_profiles(limit=args.limit, use_multi_vector=not args.legacy)


if __name__ == "__main__":
    main()
