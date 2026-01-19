# api/services/hybrid_search.py
"""
Hybrid search combining:
1. Multi-vector profile representation (searches with multiple representative embeddings)
2. Metadata pre-filtering (uses structured profile data)
3. Keyword boosting (boosts matches on important terms)
4. Score fusion (combines results from multiple search strategies)

This provides significantly better match quality than single-vector mean pooling.
"""

from __future__ import annotations

import logging
import math
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from bson import ObjectId

from api.services.mongo import get_db
from embeddings.vector_store import get_chroma_collection
from embeddings.tender_embedder import TenderEmbedder

logger = logging.getLogger(__name__)

TENDER_CHROMA_COLLECTION = os.getenv("TENDER_CHROMA_COLLECTION", "tender_embeddings")
PROFILE_CHROMA_COLLECTION = os.getenv("PROFILE_CHROMA_COLLECTION", "company_profiles")

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> Set[str]:
    """Tokenize text to lowercase words."""
    return {t.lower() for t in _TOKEN_RE.findall(text or "")}


def _similarity_from_distance(distance: float) -> float:
    """Convert Chroma distance to similarity score."""
    try:
        return max(0.0, 1.0 - float(distance))
    except Exception:
        return 0.0


def _normalize_vector(vec: List[float]) -> List[float]:
    """Normalize vector to unit length."""
    norm = math.sqrt(sum(float(v) * float(v) for v in vec))
    if norm <= 0:
        return vec
    return [float(v) / norm for v in vec]


def _compute_keyword_boost(
    profile_keywords: Set[str],
    tender_text: str,
    max_boost: float = 0.2,
) -> float:
    """
    Compute keyword overlap boost between profile keywords and tender text.

    Args:
        profile_keywords: Set of important keywords from profile
        tender_text: Tender document text
        max_boost: Maximum boost value (default 0.2 = 20%)

    Returns:
        Boost value between 0 and max_boost
    """
    if not profile_keywords:
        return 0.0

    tender_words = _tokenize(tender_text)
    if not tender_words:
        return 0.0

    overlap = len(profile_keywords & tender_words)
    overlap_ratio = overlap / max(len(profile_keywords), 1)

    # Scale to max_boost
    return min(overlap_ratio * max_boost * 2, max_boost)


def _extract_profile_keywords(profile_data: Optional[Dict[str, Any]]) -> Set[str]:
    """
    Extract important keywords from structured profile data.

    These keywords will be used for keyword boosting in search.
    """
    keywords: Set[str] = set()

    if not profile_data:
        return keywords

    # Extract from various fields
    keyword_fields = [
        "capabilities",
        "key_technologies",
        "domain_expertise",
        "industries",
        "project_types",
        "certifications",
    ]

    for field in keyword_fields:
        values = profile_data.get(field, [])
        if isinstance(values, list):
            for value in values:
                if isinstance(value, str):
                    # Add whole value and individual words
                    keywords.add(value.lower())
                    keywords.update(_tokenize(value))

    # Remove common stop words
    stop_words = {"and", "or", "the", "of", "in", "for", "to", "a", "an", "is", "are"}
    keywords = keywords - stop_words

    return keywords


def search_with_multi_vector(
    representative_embeddings: List[List[float]],
    tender_collection,
    top_k: int = 10,
    score_weight: float = 1.0,
) -> Dict[str, float]:
    """
    Search tender collection with multiple representative embeddings.

    Uses score fusion to combine results from different representative vectors.

    Args:
        representative_embeddings: List of representative profile embeddings
        tender_collection: Chroma collection for tenders
        top_k: Number of results per embedding
        score_weight: Weight for this search strategy

    Returns:
        Dictionary mapping tender chunk IDs to aggregated scores
    """
    all_results: Dict[str, float] = {}

    for emb in representative_embeddings:
        results = tender_collection.query(
            query_embeddings=[emb],
            n_results=top_k * 2,
            include=["distances", "metadatas"],
        )

        ids = (results.get("ids") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        for chunk_id, dist in zip(ids, distances):
            score = _similarity_from_distance(dist) * score_weight
            # Take maximum score across all representatives
            if chunk_id in all_results:
                all_results[chunk_id] = max(all_results[chunk_id], score)
            else:
                all_results[chunk_id] = score

    return all_results


def search_with_mean_embedding(
    mean_embedding: List[float],
    tender_collection,
    top_k: int = 10,
    score_weight: float = 0.8,
) -> Dict[str, float]:
    """
    Search with traditional mean embedding (for general matches).

    Args:
        mean_embedding: Mean-pooled profile embedding
        tender_collection: Chroma collection for tenders
        top_k: Number of results
        score_weight: Weight for this strategy (lower than multi-vector)

    Returns:
        Dictionary mapping tender chunk IDs to scores
    """
    results = tender_collection.query(
        query_embeddings=[mean_embedding],
        n_results=top_k,
        include=["distances"],
    )

    ids = (results.get("ids") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    return {
        chunk_id: _similarity_from_distance(dist) * score_weight
        for chunk_id, dist in zip(ids, distances)
    }


def fuse_search_results(
    *result_dicts: Dict[str, float],
) -> List[Tuple[str, float]]:
    """
    Fuse results from multiple search strategies.

    Takes the maximum score for each tender across all strategies.

    Args:
        result_dicts: Variable number of result dictionaries

    Returns:
        List of (chunk_id, score) tuples sorted by score descending
    """
    fused: Dict[str, float] = {}

    for results in result_dicts:
        for chunk_id, score in results.items():
            if chunk_id in fused:
                fused[chunk_id] = max(fused[chunk_id], score)
            else:
                fused[chunk_id] = score

    # Sort by score descending
    sorted_results = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return sorted_results


def search_tenders_hybrid(
    profile_representation: Dict[str, Any],
    profile_data: Optional[Dict[str, Any]] = None,
    query: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Hybrid search combining multi-vector, mean embedding, and keyword boosting.

    Args:
        profile_representation: Profile representation from create_profile_representation
            Should contain: representative_embeddings, mean_embedding
        profile_data: Optional structured profile data for keyword extraction
        query: Optional text query to combine with profile
        filters: Optional filters (location, dept, deadline)
        top_k: Number of results to return

    Returns:
        List of matched tenders with scores and metadata
    """
    db = get_db()
    tender_docs = db["tender_documents"]
    raw_tenders = db["raw_tenders"]

    tender_collection = get_chroma_collection(name=TENDER_CHROMA_COLLECTION)
    profile_collection = get_chroma_collection(name=PROFILE_CHROMA_COLLECTION)

    embedder = TenderEmbedder()

    # Get embeddings from profile representation
    rep_embeddings = profile_representation.get("representative_embeddings", [])
    mean_embedding = profile_representation.get("mean_embedding")

    # Process optional query
    query_embedding: Optional[List[float]] = None
    if query and query.strip():
        embedded = embedder.embed(query.strip())
        if embedded:
            query_embedding = embedded[0]

    # Collect search results from different strategies
    all_results: Dict[str, float] = {}

    # Strategy 1: Multi-vector search (primary)
    if rep_embeddings:
        multi_vector_results = search_with_multi_vector(
            representative_embeddings=rep_embeddings,
            tender_collection=tender_collection,
            top_k=top_k * 3,
            score_weight=1.0,
        )
        for chunk_id, score in multi_vector_results.items():
            all_results[chunk_id] = max(all_results.get(chunk_id, 0), score)

    # Strategy 2: Mean embedding search (fallback/supplement)
    if mean_embedding:
        mean_results = search_with_mean_embedding(
            mean_embedding=mean_embedding,
            tender_collection=tender_collection,
            top_k=top_k * 2,
            score_weight=0.8,
        )
        for chunk_id, score in mean_results.items():
            if chunk_id in all_results:
                all_results[chunk_id] = max(all_results[chunk_id], score)
            else:
                all_results[chunk_id] = score

    # Strategy 3: Query-based search (if query provided)
    if query_embedding:
        query_results = tender_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 2,
            include=["distances"],
        )
        ids = (query_results.get("ids") or [[]])[0]
        distances = (query_results.get("distances") or [[]])[0]

        for chunk_id, dist in zip(ids, distances):
            score = _similarity_from_distance(dist) * 0.9  # Slightly lower weight
            if chunk_id in all_results:
                all_results[chunk_id] = max(all_results[chunk_id], score)
            else:
                all_results[chunk_id] = score

    # Sort results by score
    sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)

    # Extract profile keywords for boosting
    profile_keywords = _extract_profile_keywords(profile_data)
    if query:
        profile_keywords.update(_tokenize(query))

    # Process and filter results
    results: List[Dict[str, Any]] = []
    seen_tender_ids: Set[str] = set()

    for chunk_id, base_score in sorted_results[:top_k * 5]:  # Get more for filtering
        # Parse chunk ID to get tender info
        # Format: "tender:{document_id}:{chunk_index}"
        parts = chunk_id.split(":")
        if len(parts) < 3 or parts[0] != "tender":
            continue

        document_id = parts[1]

        # Skip if we already have a result for this tender
        if document_id in seen_tender_ids:
            continue

        # Get tender document
        try:
            tender_pdf = tender_docs.find_one({"_id": ObjectId(document_id)})
        except Exception:
            tender_pdf = None

        if not tender_pdf:
            continue

        # Check expiration
        expires_at = tender_pdf.get("expires_at")
        if expires_at and expires_at <= datetime.utcnow():
            continue

        # Get tender ID for raw_tender lookup
        tender_id = tender_pdf.get("tender_id")

        # Apply custom filters
        raw_tender = None
        if tender_id:
            try:
                raw_tender = raw_tenders.find_one({"_id": ObjectId(tender_id)})
            except Exception:
                pass

        if filters:
            if not _passes_filters(tender_pdf, raw_tender, filters):
                continue

        # Get tender snippet from vector store for keyword boosting
        chunk_result = tender_collection.get(
            ids=[chunk_id],
            include=["documents"],
        )
        tender_snippet = ""
        if chunk_result.get("documents"):
            tender_snippet = (chunk_result["documents"][0] or "")[:500]

        # Apply keyword boost
        keyword_boost = _compute_keyword_boost(profile_keywords, tender_snippet)
        final_score = round(base_score + keyword_boost, 4)

        # Get profile snippet for explanation
        profile_snippet = _select_profile_snippet(
            tender_snippet=tender_snippet,
            embedder=embedder,
            profile_collection=profile_collection,
            profile_file_hashes=profile_representation.get("file_hashes"),
        )

        seen_tender_ids.add(document_id)
        results.append({
            "tender_id": str(tender_id) if tender_id else None,
            "document_id": document_id,
            "score": round(base_score, 4),
            "keyword_boost": round(keyword_boost, 4),
            "final_score": final_score,
            "pdf_url": tender_pdf.get("pdf_url"),
            "local_path": tender_pdf.get("local_path"),
            "source": tender_pdf.get("source"),
            "title": raw_tender.get("title") if raw_tender else None,
            "tender_ref_no": raw_tender.get("tender_ref_no") if raw_tender else None,
            "duration": raw_tender.get("duration") if raw_tender else None,
            "because": {
                "tender_snippet": tender_snippet[:350],
                "profile_snippet": profile_snippet[:350],
                "matched_keywords": list(profile_keywords & _tokenize(tender_snippet))[:10],
            },
        })

        if len(results) >= top_k:
            break

    # Sort by final score
    results.sort(key=lambda r: r.get("final_score", 0.0), reverse=True)

    return results


def _passes_filters(
    tender_doc: Optional[dict],
    raw_tender: Optional[dict],
    filters: Optional[Dict[str, Any]],
) -> bool:
    """Check if tender passes all filters."""
    if not filters:
        return True

    for key, value in filters.items():
        if value is None or value == "":
            continue
        haystack = None
        if tender_doc and key in tender_doc:
            haystack = tender_doc.get(key)
        if haystack is None and raw_tender and key in raw_tender:
            haystack = raw_tender.get(key)
        if haystack is None:
            continue
        if str(value).lower() not in str(haystack).lower():
            return False
    return True


def _select_profile_snippet(
    *,
    tender_snippet: str,
    embedder: TenderEmbedder,
    profile_collection,
    profile_file_hashes: Optional[List[str]],
) -> str:
    """Select best matching profile snippet for explanation."""
    if not tender_snippet:
        return ""
    try:
        emb = embedder.embed(tender_snippet)[0]
        pr = profile_collection.query(
            query_embeddings=[emb],
            n_results=5,
            include=["documents", "metadatas"],
        )
        documents = (pr.get("documents") or [[]])[0]
        metadatas = (pr.get("metadatas") or [[]])[0]
        for doc_text, meta in zip(documents, metadatas):
            if not profile_file_hashes or meta.get("file_hash") in profile_file_hashes:
                return (doc_text or "")[:350]
    except Exception:
        pass
    return ""


def explain_match(
    profile_data: Dict[str, Any],
    tender_text: str,
) -> Dict[str, Any]:
    """
    Generate an explanation for why a profile matches a tender.

    Args:
        profile_data: Structured profile data
        tender_text: Tender document text

    Returns:
        Dictionary with match explanation
    """
    profile_keywords = _extract_profile_keywords(profile_data)
    tender_words = _tokenize(tender_text)

    matched_keywords = profile_keywords & tender_words

    # Categorize matched keywords
    capability_matches = []
    technology_matches = []
    domain_matches = []

    capabilities = {w.lower() for cap in profile_data.get("capabilities", []) for w in _tokenize(cap)}
    technologies = {w.lower() for tech in profile_data.get("key_technologies", []) for w in _tokenize(tech)}
    domains = {w.lower() for dom in profile_data.get("domain_expertise", []) for w in _tokenize(dom)}

    for keyword in matched_keywords:
        if keyword in capabilities:
            capability_matches.append(keyword)
        elif keyword in technologies:
            technology_matches.append(keyword)
        elif keyword in domains:
            domain_matches.append(keyword)

    return {
        "total_matched_keywords": len(matched_keywords),
        "capability_matches": capability_matches[:5],
        "technology_matches": technology_matches[:5],
        "domain_matches": domain_matches[:5],
        "match_strength": "strong" if len(matched_keywords) > 10 else "medium" if len(matched_keywords) > 5 else "weak",
    }
