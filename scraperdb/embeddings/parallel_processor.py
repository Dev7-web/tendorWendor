# embeddings/parallel_processor.py
"""
Parallel PDF processing for multiple company profile documents.
Processes PDFs concurrently using ThreadPoolExecutor for 3-5x faster processing.
"""

from __future__ import annotations

import hashlib
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from docling.document_converter import DocumentConverter

try:
    from embeddings.smart_extractor import extract_relevant_sections, get_extraction_stats
    from embeddings.chunker import chunk_text
    from embeddings.tender_embedder import TenderEmbedder
except ModuleNotFoundError:
    from smart_extractor import extract_relevant_sections, get_extraction_stats
    from chunker import chunk_text
    from tender_embedder import TenderEmbedder

logger = logging.getLogger(__name__)

# Configuration
MAX_PARALLEL_PDFS = int(os.getenv("MAX_PARALLEL_PDFS", "4"))
CACHE_DIR = os.getenv("DOCLING_CACHE_DIR", "")
ENABLE_CACHE = os.getenv("ENABLE_DOCLING_CACHE", "true").lower() in ("true", "1", "yes")


def _compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of a file for caching."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def _serialize_docling_value(value: Any) -> Any:
    """Serialize Docling output values for storage."""
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


class DoclingCache:
    """Simple file-based cache for Docling outputs."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.enabled = ENABLE_CACHE and bool(cache_dir or CACHE_DIR)
        self.cache_dir = cache_dir or CACHE_DIR
        if self.enabled and self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self, file_hash: str) -> str:
        return os.path.join(self.cache_dir, f"{file_hash}.json")

    def get(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached Docling output if available."""
        if not self.enabled:
            return None
        cache_file = self._cache_path(file_hash)
        if not os.path.exists(cache_file):
            return None
        try:
            import json
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache for {file_hash}: {e}")
            return None

    def set(self, file_hash: str, data: Dict[str, Any]) -> None:
        """Cache Docling output."""
        if not self.enabled:
            return
        cache_file = self._cache_path(file_hash)
        try:
            import json
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save cache for {file_hash}: {e}")


# Global cache instance
_cache = DoclingCache()


def process_single_pdf(
    pdf_path: str,
    converter: Optional[DocumentConverter] = None,
    use_smart_extraction: bool = True,
    file_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process a single PDF file with Docling.

    Args:
        pdf_path: Path to the PDF file
        converter: Optional pre-initialized DocumentConverter
        use_smart_extraction: Whether to apply smart section extraction
        file_hash: Optional pre-computed file hash

    Returns:
        Dictionary containing extracted text, tables, sections, and metadata
    """
    if not os.path.exists(pdf_path):
        return {"error": f"File not found: {pdf_path}", "success": False}

    # Compute file hash for caching
    if file_hash is None:
        file_hash = _compute_file_hash(pdf_path)

    # Check cache first
    cached = _cache.get(file_hash)
    if cached:
        logger.info(f"Cache hit for {file_hash}")
        return cached

    # Initialize converter if not provided
    if converter is None:
        converter = DocumentConverter()

    try:
        logger.info(f"Processing PDF: {pdf_path}")
        start_time = datetime.now(timezone.utc)

        result = converter.convert(pdf_path)

        # Extract data from Docling result
        full_text = result.document.export_to_text()
        tables_value = getattr(result.document, "tables", None)
        sections_value = getattr(result.document, "sections", None)

        output = {
            "file_hash": file_hash,
            "file_path": pdf_path,
            "text": full_text,
            "tables": _serialize_docling_value(tables_value),
            "sections": _serialize_docling_value(sections_value),
            "extracted_at": start_time.isoformat(),
            "success": True,
        }

        # Apply smart extraction if enabled
        if use_smart_extraction:
            relevant_text = extract_relevant_sections(output)
            stats = get_extraction_stats(full_text, relevant_text)
            output["relevant_text"] = relevant_text
            output["extraction_stats"] = stats
            logger.info(
                f"Smart extraction: {stats['original_pages_approx']} pages -> "
                f"{stats['extracted_pages_approx']} pages ({stats['reduction_ratio']*100:.0f}% reduction)"
            )

        # Cache the result
        _cache.set(file_hash, output)

        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        output["processing_time_seconds"] = processing_time
        logger.info(f"Processed {pdf_path} in {processing_time:.2f}s")

        return output

    except Exception as e:
        logger.exception(f"Failed to process PDF {pdf_path}: {e}")
        return {
            "file_hash": file_hash,
            "file_path": pdf_path,
            "error": str(e),
            "success": False,
        }


def process_pdfs_parallel(
    pdf_paths: List[str],
    max_workers: Optional[int] = None,
    use_smart_extraction: bool = True,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Process multiple PDFs in parallel using ThreadPoolExecutor.

    Args:
        pdf_paths: List of PDF file paths to process
        max_workers: Maximum number of parallel workers (default: MAX_PARALLEL_PDFS)
        use_smart_extraction: Whether to apply smart section extraction
        progress_callback: Optional callback(completed, total, current_file)

    Returns:
        List of (path, result_dict) tuples
    """
    if not pdf_paths:
        return []

    workers = max_workers or MAX_PARALLEL_PDFS
    workers = min(workers, len(pdf_paths))  # Don't use more workers than files

    results: List[Tuple[str, Dict[str, Any]]] = []
    completed = 0
    total = len(pdf_paths)

    logger.info(f"Processing {total} PDFs with {workers} workers")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks
        future_to_path = {
            executor.submit(
                process_single_pdf,
                path,
                None,  # Each thread gets its own converter
                use_smart_extraction,
            ): path
            for path in pdf_paths
        }

        # Collect results as they complete
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            completed += 1

            try:
                result = future.result()
                results.append((path, result))
                if progress_callback:
                    progress_callback(completed, total, path)
            except Exception as e:
                logger.exception(f"Failed to process {path}: {e}")
                results.append((path, {"error": str(e), "success": False}))
                if progress_callback:
                    progress_callback(completed, total, path)

    # Sort results to maintain original order
    path_order = {path: i for i, path in enumerate(pdf_paths)}
    results.sort(key=lambda x: path_order.get(x[0], len(pdf_paths)))

    successful = sum(1 for _, r in results if r.get("success"))
    logger.info(f"Completed processing: {successful}/{total} successful")

    return results


def merge_pdf_outputs(
    outputs: List[Dict[str, Any]],
    use_relevant_text: bool = True
) -> Dict[str, Any]:
    """
    Merge outputs from multiple PDFs into a single combined output.

    Args:
        outputs: List of output dictionaries from process_single_pdf
        use_relevant_text: Whether to use smart-extracted text (if available)

    Returns:
        Merged output dictionary
    """
    successful_outputs = [o for o in outputs if o.get("success")]

    if not successful_outputs:
        return {
            "text": "",
            "tables": [],
            "file_hashes": [],
            "success": False,
            "error": "No successful outputs to merge",
        }

    combined_text_parts: List[str] = []
    combined_tables: List[Any] = []
    file_hashes: List[str] = []
    total_original_chars = 0
    total_extracted_chars = 0

    for output in successful_outputs:
        file_hash = output.get("file_hash", "")
        if file_hash:
            file_hashes.append(file_hash)

        # Choose which text to use
        if use_relevant_text and output.get("relevant_text"):
            text = output["relevant_text"]
        else:
            text = output.get("text", "")

        if text:
            combined_text_parts.append(text)

        # Collect tables
        tables = output.get("tables", [])
        if tables:
            combined_tables.extend(tables)

        # Track stats
        stats = output.get("extraction_stats", {})
        total_original_chars += stats.get("original_chars", len(output.get("text", "")))
        total_extracted_chars += stats.get("extracted_chars", len(text))

    return {
        "text": "\n\n---\n\n".join(combined_text_parts),
        "tables": combined_tables,
        "file_hashes": file_hashes,
        "document_count": len(successful_outputs),
        "total_original_chars": total_original_chars,
        "total_extracted_chars": total_extracted_chars,
        "reduction_ratio": round(1 - (total_extracted_chars / max(total_original_chars, 1)), 2),
        "success": True,
    }


def process_and_merge_pdfs(
    pdf_paths: List[str],
    max_workers: Optional[int] = None,
    use_smart_extraction: bool = True,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to process and merge multiple PDFs.

    Args:
        pdf_paths: List of PDF file paths
        max_workers: Maximum parallel workers
        use_smart_extraction: Whether to apply smart extraction
        progress_callback: Optional progress callback

    Returns:
        Merged output dictionary
    """
    results = process_pdfs_parallel(
        pdf_paths=pdf_paths,
        max_workers=max_workers,
        use_smart_extraction=use_smart_extraction,
        progress_callback=progress_callback,
    )

    outputs = [result for _, result in results]
    return merge_pdf_outputs(outputs, use_relevant_text=use_smart_extraction)
