# embeddings/index_tenders.py

from __future__ import annotations

import argparse
import logging
from typing import List, Optional
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime, timezone
import os

try:
    from embeddings.chunker import chunk_text
    from embeddings.tender_embedder import TenderEmbedder
    from embeddings.vector_store import get_chroma_collection, get_chroma_client
except ModuleNotFoundError:
    from chunker import chunk_text
    from tender_embedder import TenderEmbedder
    from vector_store import get_chroma_collection, get_chroma_client

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "tender_db")

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))

TENDER_COLLECTION_NAME = os.getenv("TENDER_CHROMA_COLLECTION", "tender_embeddings")

mongo = MongoClient(MONGO_URI)
db = mongo[DB_NAME]
docling_outputs = db["docling_outputs"]
tender_documents = db["tender_documents"]

collection = get_chroma_collection(name=TENDER_COLLECTION_NAME)
embedder = TenderEmbedder()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _combine_text_and_tables(doc: dict) -> str:
    """
    Combine text + table text (clean).
    Prefer table['text'] when available to avoid embedding noisy metadata.
    """
    text = doc.get("text", "") or ""
    tables = doc.get("tables", []) or []

    table_texts: List[str] = []
    for table in tables:
        if isinstance(table, dict):
            if "text" in table and isinstance(table["text"], str):
                table_texts.append(table["text"])
            else:
                # fallback: only stringify values that look textual
                for v in table.values():
                    if isinstance(v, str) and v.strip():
                        table_texts.append(v.strip())
        else:
            if isinstance(table, str) and table.strip():
                table_texts.append(table.strip())

    combined = "\n".join([text] + table_texts).strip()
    return combined


def _batch_items(items: List[str], batch_size: int) -> List[List[str]]:
    if batch_size <= 0:
        return [items]
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def index_pending_tenders(limit: int = 10) -> None:
    """
    Index ONLY tender docs from docling_outputs into tender_embeddings Chroma collection.
    """
    if CHUNK_SIZE <= 0:
        raise ValueError("CHUNK_SIZE must be > 0")

    query = {"doc_type": "tender", "indexed": {"$ne": True}}
    if limit and limit > 0:
        pending = docling_outputs.find(query, limit=limit)
    else:
        pending = docling_outputs.find(query)

    for doc in pending:
        document_oid = doc.get("document_id") or doc.get("_id")
        document_id = str(document_oid)
        tender_id = doc.get("tender_id")
        tender_id_str = str(tender_id) if tender_id is not None else None
        source = doc.get("source")

        tender_doc = tender_documents.find_one({"_id": document_oid})
        if not tender_doc:
            docling_outputs.update_one(
                {"_id": doc["_id"]},
                {
                    "$set": {
                        "indexed": True,
                        "index_status": "skipped",
                        "index_error": "tender_document_missing",
                        "indexed_at": datetime.now(timezone.utc),
                    }
                },
            )
            logger.info("Skipping expired/missing tender doc: %s", document_id)
            continue

        expires_at = tender_doc.get("expires_at")
        if expires_at and expires_at <= datetime.utcnow():
            docling_outputs.update_one(
                {"_id": doc["_id"]},
                {
                    "$set": {
                        "indexed": True,
                        "index_status": "skipped",
                        "index_error": "tender_document_expired",
                        "indexed_at": datetime.now(timezone.utc),
                    }
                },
            )
            logger.info("Skipping expired tender doc: %s", document_id)
            continue

        combined_text = _combine_text_and_tables(doc)
        if not combined_text:
            logger.warning("Skipping empty tender doc: %s", document_id)
            continue

        chunks = chunk_text(combined_text, max_chars=CHUNK_SIZE, overlap_chars=CHUNK_OVERLAP)
        if not chunks:
            logger.warning("No chunks created for tender doc: %s", document_id)
            continue

        logger.info("Indexing tender %s -> %d chunks", tender_id_str, len(chunks))

        try:
            for batch_index, batch in enumerate(_batch_items(chunks, BATCH_SIZE)):
                batch_embeddings = embedder.embed(batch)
                if len(batch_embeddings) != len(batch):
                    raise RuntimeError("Embedding count mismatch")

                start_index = batch_index * BATCH_SIZE

                ids = [f"tender:{document_id}:{i}" for i in range(start_index, start_index + len(batch))]

                metadatas = [
                    {
                        "doc_type": "tender",
                        "tender_id": tender_id_str,
                        "document_id": document_id,
                        "chunk_index": i,
                        "source": source,
                        "model_name": embedder.model_name,
                        "chunk_size": CHUNK_SIZE,
                        "chunk_overlap": CHUNK_OVERLAP,
                    }
                    for i in range(start_index, start_index + len(batch))
                ]

                collection.upsert(ids=ids, documents=batch, embeddings=batch_embeddings, metadatas=metadatas)

            docling_outputs.update_one(
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

            logger.info("Indexed tender doc successfully: %s", document_id)

        except Exception as exc:
            docling_outputs.update_one(
                {"_id": doc["_id"]},
                {
                    "$set": {
                        "indexed": False,
                        "index_error": str(exc),
                        "failed_at": datetime.now(timezone.utc),
                    }
                },
            )
            logger.exception("Failed to index tender doc %s", document_id)


def _reset_collection() -> None:
    global collection
    client = get_chroma_client()
    try:
        client.delete_collection(name=TENDER_COLLECTION_NAME)
    except Exception:
        pass
    collection = get_chroma_collection(name=TENDER_COLLECTION_NAME)


def rebuild_tenders(limit: Optional[int] = None) -> None:
    _reset_collection()
    docling_outputs.update_many(
        {"doc_type": "tender"},
        {"$set": {"indexed": False, "index_status": "rebuild"}},
    )
    index_pending_tenders(limit=limit or 0)


def index_tenders(limit: int = 10) -> None:
    index_pending_tenders(limit=limit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    if args.rebuild:
        rebuild_tenders(limit=args.limit)
        return

    index_tenders(limit=args.limit)


if __name__ == "__main__":
    main()
