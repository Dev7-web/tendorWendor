# api/services/match.py
"""
Single-shot profile matching service.
Now uses multi-vector representation for better match quality.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from bson import ObjectId

from api.services.mongo import get_db
from api.services.profile_ingest import build_profile_embedding_from_file
from api.services.search import search_tenders_with_embedding, search_tenders_with_profile_representation


def match_search(
    *,
    file_hash: str,
    file_path: str,
    query: Optional[str],
    filters: Optional[Dict[str, Any]],
    top_k: int,
) -> Optional[List[Dict[str, Any]]]:
    """
    Perform single-shot profile matching against tenders.

    Uses the optimized multi-vector representation if available.

    Args:
        file_hash: SHA256 hash of the profile file
        file_path: Path to the profile PDF
        query: Optional text query to refine search
        filters: Optional filters (location, dept, etc.)
        top_k: Number of results to return

    Returns:
        List of matched tenders or None if processing failed
    """
    # Build profile representation (now returns multi-vector data)
    profile_repr = build_profile_embedding_from_file(file_hash, file_path)
    if not profile_repr:
        return None

    # Check if we got new multi-vector format or legacy single-vector
    if isinstance(profile_repr, dict) and profile_repr.get("representative_embeddings"):
        # New multi-vector format - use optimized search
        profile_repr["file_hashes"] = [file_hash]
        return search_tenders_with_profile_representation(
            profile_representation=profile_repr,
            top_k=top_k,
            query=query,
            filters=filters,
        )
    elif isinstance(profile_repr, dict) and profile_repr.get("mean_embedding"):
        # New format but without representatives - use mean embedding
        return search_tenders_with_embedding(
            profile_embedding=profile_repr["mean_embedding"],
            top_k=top_k,
            query=query,
            filters=filters,
            profile_file_hashes=[file_hash],
            representative_embeddings=profile_repr.get("representative_embeddings"),
            profile_data=profile_repr.get("profile_data"),
        )
    elif isinstance(profile_repr, list):
        # Legacy format - single embedding vector
        return search_tenders_with_embedding(
            profile_embedding=profile_repr,
            top_k=top_k,
            query=query,
            filters=filters,
            profile_file_hashes=[file_hash],
        )
    else:
        return None


def process_match_job(job_id: str) -> None:
    """
    Process a match job asynchronously.
    Called by the job worker for large files.
    """
    db = get_db()
    jobs = db["jobs"]

    job = jobs.find_one({"_id": ObjectId(job_id)})
    if not job:
        return

    now = datetime.now(timezone.utc)
    jobs.update_one(
        {"_id": ObjectId(job_id)},
        {"$set": {"status": "running", "step": "embedding", "progress": 10, "updated_at": now}},
    )

    file_hash = job.get("file_hash")
    file_path = job.get("file_path")
    query = job.get("query")
    filters = job.get("filters") or {}
    top_k = int(job.get("top_k") or 5)

    if not file_hash or not file_path:
        jobs.update_one(
            {"_id": ObjectId(job_id)},
            {
                "$set": {
                    "status": "failed",
                    "step": "error",
                    "error": "Missing file_hash or file_path",
                    "updated_at": datetime.now(timezone.utc),
                }
            },
        )
        return

    results = match_search(
        file_hash=file_hash,
        file_path=file_path,
        query=query,
        filters=filters,
        top_k=top_k,
    )
    if results is None:
        jobs.update_one(
            {"_id": ObjectId(job_id)},
            {
                "$set": {
                    "status": "failed",
                    "step": "error",
                    "error": "Profile processing failed",
                    "updated_at": datetime.now(timezone.utc),
                }
            },
        )
        return

    jobs.update_one(
        {"_id": ObjectId(job_id)},
        {
            "$set": {
                "status": "done",
                "step": "ready",
                "progress": 100,
                "result": results,
                "updated_at": datetime.now(timezone.utc),
            }
        },
    )
