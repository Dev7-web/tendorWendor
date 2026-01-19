# api/services/profile_ingest.py
"""
Optimized profile PDF processing & embedding service.

This module implements the optimized pipeline from the analysis:
1. Smart section extraction (reduces 200 pages -> 20-30 pages)
2. Parallel PDF processing (3-5x faster)
3. Multi-vector representation (better match quality)
4. Optional LLM extraction (structured metadata)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Load dotenv early to ensure env vars are available
from pathlib import Path
from dotenv import load_dotenv

# Find the .env file relative to this file's location (in scraperdb/)
_this_dir = Path(__file__).resolve().parent  # api/services/
_project_root = _this_dir.parent.parent  # scraperdb/
_env_file = _project_root / ".env"
load_dotenv(_env_file)

from bson import ObjectId
from docling.document_converter import DocumentConverter

from api.services.mongo import get_db
from embeddings.chunker import chunk_text
from embeddings.tender_embedder import TenderEmbedder
from embeddings.vector_store import get_chroma_collection
from embeddings.smart_extractor import extract_relevant_sections, get_extraction_stats
from embeddings.profile_embeddings import create_profile_representation
from embeddings.parallel_processor import process_single_pdf, process_pdfs_parallel, merge_pdf_outputs
from embeddings.llm_extractor import (
    extract_profile_with_llm,
    create_searchable_profile_text,
    extract_filter_metadata,
    is_llm_extraction_available,
)

logger = logging.getLogger(__name__)

PROFILE_OUTPUTS_COLLECTION = "company_profile_outputs"
PROFILE_EMBEDDINGS_COLLECTION = "company_profile_embeddings"
PROFILE_CHROMA_COLLECTION = os.getenv("PROFILE_CHROMA_COLLECTION", "company_profiles")

CHUNK_SIZE = int(os.getenv("PROFILE_CHUNK_SIZE", "800"))  # Increased for profiles
CHUNK_OVERLAP = int(os.getenv("PROFILE_CHUNK_OVERLAP", "100"))
BATCH_SIZE = int(os.getenv("PROFILE_BATCH_SIZE", "64"))
N_REPRESENTATIVES = int(os.getenv("PROFILE_N_REPRESENTATIVES", "5"))
USE_SMART_EXTRACTION = os.getenv("USE_SMART_EXTRACTION", "true").lower() in ("true", "1", "yes")
USE_LLM = os.getenv("USE_LLM_EXTRACTION", "false").lower() in ("true", "1", "yes")

# Debug: print configuration
print(f"[PROFILE_INGEST] USE_LLM={USE_LLM}, USE_SMART_EXTRACTION={USE_SMART_EXTRACTION}")


def _serialize_docling_value(value):
    """Serialize Docling output values for MongoDB storage."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_serialize_docling_value(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _serialize_docling_value(v) for k, v in value.items()}
    for attr in ("model_dump", "to_dict", "dict"):
        method = getattr(value, attr, None)
        if callable(method):
            try:
                return _serialize_docling_value(method())
            except Exception:
                break
    if hasattr(value, "__dict__"):
        return {str(k): _serialize_docling_value(v) for k, v in value.__dict__.items()}
    return str(value)


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


def _combine_text_and_tables(output: dict) -> str:
    """Combine text and table content from Docling output."""
    text = output.get("text", "") or ""
    tables = output.get("tables", []) or []

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


def _ensure_profile_output(
    *,
    outputs,
    converter: DocumentConverter,
    file_hash: str,
    file_path: Optional[str],
    profile_id: Optional[ObjectId],
    use_smart_extraction: bool = True,
) -> Optional[dict]:
    """
    Ensure profile output exists, processing with Docling if needed.
    Now uses smart section extraction to reduce processing.
    """
    existing = outputs.find_one({"file_hash": file_hash})
    if existing:
        if profile_id:
            outputs.update_one(
                {"file_hash": file_hash},
                {"$addToSet": {"profile_ids": profile_id}},
            )
        return existing

    if not file_path:
        return None

    try:
        # Use optimized parallel processor with smart extraction
        result = process_single_pdf(
            pdf_path=file_path,
            converter=converter,
            use_smart_extraction=use_smart_extraction,
            file_hash=file_hash,
        )

        if not result.get("success"):
            logger.error(f"Failed to process PDF: {result.get('error')}")
            return None

        now = datetime.now(timezone.utc)

        output_doc = {
            "doc_type": "profile",
            "file_hash": file_hash,
            "text": result.get("text", ""),
            "relevant_text": result.get("relevant_text"),  # Smart extracted text
            "tables": result.get("tables", []),
            "sections": result.get("sections", []),
            "extraction_stats": result.get("extraction_stats"),
            "extracted_at": now,
            "docling_version": "v2_optimized",
            "indexed": False,
            "updated_at": now,
        }

        update = {"$set": output_doc, "$setOnInsert": {"created_at": now}}
        if profile_id:
            update["$addToSet"] = {"profile_ids": profile_id}

        outputs.update_one({"file_hash": file_hash}, update, upsert=True)

        stats = result.get("extraction_stats", {})
        logger.info(
            f"Processed profile {file_hash}: "
            f"{stats.get('original_pages_approx', '?')} pages -> "
            f"{stats.get('extracted_pages_approx', '?')} pages"
        )

        return output_doc

    except Exception as e:
        logger.exception(f"Failed to process profile PDF: {e}")
        return None


def _ensure_profile_embedding(
    *,
    outputs,
    embeddings_meta,
    profile_collection,
    converter: DocumentConverter,
    embedder: TenderEmbedder,
    file_hash: str,
    file_path: Optional[str],
    profile_id: Optional[ObjectId],
    use_smart_extraction: bool = True,
    use_llm_extraction: bool = False,
) -> Optional[Tuple[Dict[str, Any], int]]:
    """
    Ensure profile embedding exists, creating multi-vector representation if needed.

    Returns:
        Tuple of (profile_representation_dict, chunk_count) or None
    """
    existing = embeddings_meta.find_one({"file_hash": file_hash})
    if existing and existing.get("representative_embeddings"):
        print(f"[PROFILE_INGEST] CACHE HIT: Returning cached embedding for {file_hash[:16]}... (has profile_data: {existing.get('profile_data') is not None})")
        if profile_id:
            embeddings_meta.update_one(
                {"file_hash": file_hash},
                {"$addToSet": {"profile_ids": profile_id}},
            )
        # Return existing representation
        return {
            "representative_embeddings": existing.get("representative_embeddings", []),
            "representative_chunks": existing.get("representative_chunks", []),
            "mean_embedding": existing.get("summary_embedding"),  # Backwards compatible
            "max_embedding": existing.get("max_embedding"),
            "profile_data": existing.get("profile_data"),
            "filter_metadata": existing.get("filter_metadata"),
        }, int(existing.get("chunk_count", 0))

    # Process the PDF
    print(f"[PROFILE_INGEST] CACHE MISS: Processing new profile {file_hash[:16]}... (use_llm_extraction={use_llm_extraction})")
    output = _ensure_profile_output(
        outputs=outputs,
        converter=converter,
        file_hash=file_hash,
        file_path=file_path,
        profile_id=profile_id,
        use_smart_extraction=use_smart_extraction,
    )
    if not output:
        return None

    # Use relevant text if available (smart extraction), otherwise full text
    text_to_process = output.get("relevant_text") or _combine_text_and_tables(output)
    if not text_to_process:
        return None

    # Optional: LLM extraction for structured data
    profile_data: Optional[Dict[str, Any]] = None
    filter_metadata: Dict[str, Any] = {}
    llm_searchable_text: Optional[str] = None

    logger.info(
        "LLM extraction check: use_llm_extraction=%s, is_available=%s",
        use_llm_extraction,
        is_llm_extraction_available(),
    )

    if use_llm_extraction and is_llm_extraction_available():
        print(f"[PROFILE_INGEST] >>> CALLING LLM EXTRACTION (text length: {len(text_to_process)})")
        logger.info(">>> Starting LLM extraction for profile (text length: %d)", len(text_to_process))
        try:
            profile_data = extract_profile_with_llm(text_to_process)
            if profile_data:
                llm_searchable_text = create_searchable_profile_text(profile_data)
                filter_metadata = extract_filter_metadata(profile_data)
                logger.info(f"LLM extracted profile: {profile_data.get('company_name', 'Unknown')}")
            else:
                logger.warning("LLM extraction returned None")
        except Exception as e:
            logger.warning(f"LLM extraction failed, continuing without: {e}")
    else:
        print(f"[PROFILE_INGEST] SKIPPING LLM: use_llm_extraction={use_llm_extraction}, is_available={is_llm_extraction_available()}")
        logger.info("Skipping LLM extraction (disabled or unavailable)")

    # Chunk the text
    # If we have LLM searchable text, prepend it for better embedding
    if llm_searchable_text:
        text_to_embed = f"{llm_searchable_text}\n\n{text_to_process}"
    else:
        text_to_embed = text_to_process

    chunks = chunk_text(text_to_embed, max_chars=CHUNK_SIZE, overlap_chars=CHUNK_OVERLAP)
    if not chunks:
        return None

    logger.info(f"Creating multi-vector representation from {len(chunks)} chunks for {file_hash}")

    # Create multi-vector representation
    representation = create_profile_representation(
        chunks=chunks,
        embedder=embedder,
        n_representatives=N_REPRESENTATIVES,
    )

    all_embeddings = representation.get("all_embeddings", [])
    if not all_embeddings:
        return None

    # Store all chunks in Chroma for search
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
                "is_representative": i in representation.get("representative_indices", []),
            }
            for i in range(batch_start, batch_end)
        ]

        profile_collection.upsert(
            ids=ids,
            documents=batch_chunks,
            embeddings=batch_embeddings,
            metadatas=metadatas,
        )

    # Store embedding metadata in MongoDB
    now = datetime.now(timezone.utc)
    embedding_doc = {
        "file_hash": file_hash,
        "summary_embedding": representation["mean_embedding"],  # Backwards compatible
        "max_embedding": representation.get("max_embedding"),
        "representative_embeddings": representation["representative_embeddings"],
        "representative_chunks": representation["representative_chunks"],
        "representative_indices": representation.get("representative_indices", []),
        "chunk_count": len(chunks),
        "index_model": embedder.model_name,
        "indexed_at": now,
        "updated_at": now,
        "version": "v2_multi_vector",
    }

    # Add LLM data if available
    if profile_data:
        embedding_doc["profile_data"] = profile_data
        embedding_doc["filter_metadata"] = filter_metadata

    update = {"$set": embedding_doc}
    if profile_id:
        update["$addToSet"] = {"profile_ids": profile_id}

    embeddings_meta.update_one({"file_hash": file_hash}, update, upsert=True)
    outputs.update_one(
        {"file_hash": file_hash},
        {"$set": {"indexed": True, "indexed_at": now}},
    )

    result_repr = {
        "representative_embeddings": representation["representative_embeddings"],
        "representative_chunks": representation["representative_chunks"],
        "mean_embedding": representation["mean_embedding"],
        "max_embedding": representation.get("max_embedding"),
        "profile_data": profile_data,
        "filter_metadata": filter_metadata,
    }

    return result_repr, len(chunks)


def build_profile_embedding_from_file(
    file_hash: str,
    file_path: str,
    use_smart_extraction: bool = True,
    use_llm_extraction: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Build profile embedding from a file, returning multi-vector representation.

    Args:
        file_hash: SHA256 hash of the file
        file_path: Path to the PDF file
        use_smart_extraction: Whether to use smart section extraction
        use_llm_extraction: Whether to use LLM for structured extraction

    Returns:
        Profile representation dictionary with:
        - representative_embeddings
        - mean_embedding (for backwards compatibility)
        - profile_data (if LLM used)
        - filter_metadata (if LLM used)
    """
    db = get_db()
    outputs = db[PROFILE_OUTPUTS_COLLECTION]
    embeddings_meta = db[PROFILE_EMBEDDINGS_COLLECTION]
    converter = DocumentConverter()
    embedder = TenderEmbedder()
    profile_collection = get_chroma_collection(name=PROFILE_CHROMA_COLLECTION)

    result = _ensure_profile_embedding(
        outputs=outputs,
        embeddings_meta=embeddings_meta,
        profile_collection=profile_collection,
        converter=converter,
        embedder=embedder,
        file_hash=file_hash,
        file_path=file_path,
        profile_id=None,
        use_smart_extraction=use_smart_extraction,
        use_llm_extraction=use_llm_extraction,
    )

    if not result:
        return None

    representation, _chunk_count = result
    return representation


def build_profile_embedding_from_files(
    file_infos: List[Dict[str, str]],
    use_parallel: bool = True,
    max_workers: int = 4,
    progress_callback=None,
) -> Optional[Dict[str, Any]]:
    """
    Build combined profile embedding from multiple PDF files.

    Args:
        file_infos: List of dicts with 'file_hash' and 'file_path' keys
        use_parallel: Whether to process files in parallel
        max_workers: Maximum parallel workers
        progress_callback: Optional callback(completed, total, current_file)

    Returns:
        Combined profile representation
    """
    if not file_infos:
        return None

    db = get_db()
    outputs = db[PROFILE_OUTPUTS_COLLECTION]
    embeddings_meta = db[PROFILE_EMBEDDINGS_COLLECTION]
    converter = DocumentConverter()
    embedder = TenderEmbedder()
    profile_collection = get_chroma_collection(name=PROFILE_CHROMA_COLLECTION)

    all_representations: List[Dict[str, Any]] = []
    file_hashes: List[str] = []

    if use_parallel and len(file_infos) > 1:
        # Process PDFs in parallel
        pdf_paths = [info["file_path"] for info in file_infos]
        results = process_pdfs_parallel(
            pdf_paths=pdf_paths,
            max_workers=max_workers,
            use_smart_extraction=USE_SMART_EXTRACTION,
            progress_callback=progress_callback,
        )

        # Now process embeddings (can't easily parallelize due to shared resources)
        for file_info, (_, pdf_result) in zip(file_infos, results):
            if not pdf_result.get("success"):
                logger.warning(f"Skipping failed PDF: {file_info['file_path']}")
                continue

            result = _ensure_profile_embedding(
                outputs=outputs,
                embeddings_meta=embeddings_meta,
                profile_collection=profile_collection,
                converter=converter,
                embedder=embedder,
                file_hash=file_info["file_hash"],
                file_path=file_info["file_path"],
                profile_id=None,
                use_smart_extraction=USE_SMART_EXTRACTION,
                use_llm_extraction=USE_LLM,
            )

            if result:
                representation, _ = result
                all_representations.append(representation)
                file_hashes.append(file_info["file_hash"])
    else:
        # Sequential processing
        for i, file_info in enumerate(file_infos):
            result = _ensure_profile_embedding(
                outputs=outputs,
                embeddings_meta=embeddings_meta,
                profile_collection=profile_collection,
                converter=converter,
                embedder=embedder,
                file_hash=file_info["file_hash"],
                file_path=file_info["file_path"],
                profile_id=None,
                use_smart_extraction=USE_SMART_EXTRACTION,
                use_llm_extraction=USE_LLM,
            )

            if result:
                representation, _ = result
                all_representations.append(representation)
                file_hashes.append(file_info["file_hash"])

            if progress_callback:
                progress_callback(i + 1, len(file_infos), file_info["file_path"])

    if not all_representations:
        return None

    # Combine representations
    combined_rep_embeddings: List[List[float]] = []
    combined_rep_chunks: List[str] = []
    all_mean_embeddings: List[List[float]] = []

    for repr in all_representations:
        combined_rep_embeddings.extend(repr.get("representative_embeddings", []))
        combined_rep_chunks.extend(repr.get("representative_chunks", []))
        if repr.get("mean_embedding"):
            all_mean_embeddings.append(repr["mean_embedding"])

    # Compute combined mean embedding
    combined_mean = _mean_vectors(all_mean_embeddings) if all_mean_embeddings else None

    return {
        "representative_embeddings": combined_rep_embeddings,
        "representative_chunks": combined_rep_chunks,
        "mean_embedding": combined_mean,
        "file_hashes": file_hashes,
        "document_count": len(all_representations),
    }


def process_profile_job(job_id: str) -> None:
    """
    Process a profile job with optimized pipeline.
    Uses parallel processing and multi-vector representation.
    """
    db = get_db()
    jobs = db["jobs"]
    profiles = db["company_profiles"]
    company_docs = db["company_documents"]
    outputs = db[PROFILE_OUTPUTS_COLLECTION]
    embeddings_meta = db[PROFILE_EMBEDDINGS_COLLECTION]

    job = jobs.find_one({"_id": ObjectId(job_id)})
    if not job:
        return

    logger.info(
        "LLM extraction status (profile_ingest): enabled=%s available=%s",
        USE_LLM,
        is_llm_extraction_available(),
    )

    profile_id = job["profile_id"]
    now = datetime.now(timezone.utc)

    jobs.update_one(
        {"_id": ObjectId(job_id)},
        {"$set": {"status": "running", "step": "processing", "progress": 5, "updated_at": now}},
    )

    converter = DocumentConverter()
    embedder = TenderEmbedder()
    profile_collection = get_chroma_collection(name=PROFILE_CHROMA_COLLECTION)

    docs = list(company_docs.find({"profile_id": profile_id, "docling_status": {"$ne": "failed"}}))
    if not docs:
        jobs.update_one(
            {"_id": ObjectId(job_id)},
            {"$set": {"status": "failed", "step": "processing", "error": "No documents", "updated_at": now}},
        )
        profiles.update_one(
            {"_id": profile_id},
            {"$set": {"status": "FAILED", "updated_at": now}},
        )
        return

    all_representations: List[Dict[str, Any]] = []
    file_hashes: List[str] = []
    total_docs = len(docs)

    # Process each document
    for i, doc in enumerate(docs):
        doc_id = doc["_id"]
        file_hash = doc.get("file_hash")
        file_path = doc.get("local_path")

        if not file_hash:
            company_docs.update_one(
                {"_id": doc_id},
                {"$set": {"docling_status": "failed", "updated_at": datetime.now(timezone.utc)}},
            )
            continue

        result = _ensure_profile_embedding(
            outputs=outputs,
            embeddings_meta=embeddings_meta,
            profile_collection=profile_collection,
            converter=converter,
            embedder=embedder,
            file_hash=file_hash,
            file_path=file_path,
            profile_id=profile_id,
            use_smart_extraction=USE_SMART_EXTRACTION,
            use_llm_extraction=USE_LLM,
        )

        if not result:
            company_docs.update_one(
                {"_id": doc_id},
                {"$set": {"docling_status": "failed", "updated_at": datetime.now(timezone.utc)}},
            )
            continue

        representation, _chunk_count = result
        all_representations.append(representation)
        file_hashes.append(file_hash)

        company_docs.update_one(
            {"_id": doc_id},
            {"$set": {"docling_status": "done", "updated_at": datetime.now(timezone.utc)}},
        )

        # Update progress
        pct = 5 + int(((i + 1) / max(1, total_docs)) * 85)
        jobs.update_one(
            {"_id": ObjectId(job_id)},
            {"$set": {"progress": pct, "updated_at": datetime.now(timezone.utc)}},
        )

    if not all_representations:
        jobs.update_one(
            {"_id": ObjectId(job_id)},
            {"$set": {"status": "failed", "step": "embedding", "error": "No embeddings created", "updated_at": now}},
        )
        profiles.update_one(
            {"_id": profile_id},
            {"$set": {"status": "FAILED", "updated_at": datetime.now(timezone.utc)}},
        )
        return

    # Combine all representations
    combined_rep_embeddings: List[List[float]] = []
    combined_rep_chunks: List[str] = []
    all_mean_embeddings: List[List[float]] = []

    for repr in all_representations:
        combined_rep_embeddings.extend(repr.get("representative_embeddings", []))
        combined_rep_chunks.extend(repr.get("representative_chunks", []))
        if repr.get("mean_embedding"):
            all_mean_embeddings.append(repr["mean_embedding"])

    # Compute final profile embedding (mean of all means for backwards compatibility)
    profile_embedding = _mean_vectors(all_mean_embeddings)
    if profile_embedding is None:
        jobs.update_one(
            {"_id": ObjectId(job_id)},
            {"$set": {"status": "failed", "step": "embedding", "error": "No embeddings", "updated_at": now}},
        )
        profiles.update_one(
            {"_id": profile_id},
            {"$set": {"status": "FAILED", "updated_at": datetime.now(timezone.utc)}},
        )
        return

    # Update profile with multi-vector data
    profiles.update_one(
        {"_id": profile_id},
        {
            "$set": {
                "status": "READY",
                "profile_embedding": profile_embedding,  # Backwards compatible
                "representative_embeddings": combined_rep_embeddings,
                "representative_chunks": combined_rep_chunks,
                "file_hashes": list(dict.fromkeys(file_hashes)),
                "updated_at": datetime.now(timezone.utc),
                "embedding_version": "v2_multi_vector",
            }
        },
    )

    jobs.update_one(
        {"_id": ObjectId(job_id)},
        {"$set": {"status": "done", "step": "ready", "progress": 100, "updated_at": datetime.now(timezone.utc)}},
    )

    logger.info(
        f"Profile job {job_id} completed: {len(file_hashes)} files, "
        f"{len(combined_rep_embeddings)} representative embeddings"
    )
