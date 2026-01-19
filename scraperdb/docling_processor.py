from __future__ import annotations

from docling.document_converter import DocumentConverter
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime, timezone
import os
import argparse

load_dotenv()

mongo_uri = os.getenv("MONGO_URI") or "mongodb://localhost:27017"
db_name = os.getenv("DB_NAME", "tender_db")

client = MongoClient(mongo_uri)
db = client[db_name]

docling_outputs = db["docling_outputs"]
converter = DocumentConverter()


def _serialize_docling_value(value):
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


def process_pending_documents(collection_name: str, limit: int = 10):
    """
    Process PDFs that are not yet passed through Docling.
    Supports doc_type separation (tender/profile/etc).
    """
    documents = db[collection_name]

    pending_query = {"docling_status": "pending"}
    if not documents.find_one(pending_query, {"_id": 1}):
        print(f"No pending documents to process in {collection_name}.")
        return

    pending_docs = documents.find(pending_query, limit=limit)

    for doc in pending_docs:
        document_id = doc["_id"]
        pdf_path = doc.get("local_path")

        if not pdf_path:
            print(f" Missing local_path for document_id={document_id}")
            documents.update_one({"_id": document_id}, {"$set": {"docling_status": "failed"}})
            continue

        doc_type = doc.get("doc_type", "tender")  # IMPORTANT
        tender_id = doc.get("tender_id") if doc_type == "tender" else None
        profile_id = doc.get("profile_id") if doc_type == "profile" else None
        source = doc.get("source")

        # If already processed, mark done and skip
        if docling_outputs.find_one({"document_id": document_id}, {"_id": 1}):
            documents.update_one({"_id": document_id}, {"$set": {"docling_status": "done"}})
            print(f"Skipping already processed doc: {pdf_path}")
            continue

        print(f"📄 Docling ({doc_type}) → {pdf_path}")

        try:
            result = converter.convert(pdf_path)

            tables_value = getattr(result.document, "tables", None)
            sections_value = getattr(result.document, "sections", None)

            docling_outputs.update_one(
                {"document_id": document_id},
                {
                    "$set": {
                        "doc_type": doc_type,           # ✅ key fix
                        "tender_id": tender_id,
                        "profile_id": profile_id,
                        "source": source,
                        "document_id": document_id,
                        "text": result.document.export_to_text(),
                        "tables": _serialize_docling_value(tables_value),
                        "sections": _serialize_docling_value(sections_value),
                        "extracted_at": datetime.now(timezone.utc),
                        "docling_version": "v1",
                        # reset index flags on fresh extract
                        "indexed": False,
                    },
                    "$unset": {
                        "indexed_at": "",
                        "chunk_count": "",
                        "index_error": "",
                        "failed_at": "",
                    },
                },
                upsert=True,
            )

            documents.update_one({"_id": document_id}, {"$set": {"docling_status": "done"}})
            print("✅ Docling success")

        except Exception as e:
            documents.update_one({"_id": document_id}, {"$set": {"docling_status": "failed"}})
            print("❌ Docling failed:", e)


def docling_process(collection_name: str, limit: int = 10) -> None:
    process_pending_documents(collection_name=collection_name, limit=limit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default=os.getenv("DOCS_COLLECTION", "tender_documents"))
    parser.add_argument("--limit", type=int, default=int(os.getenv("DOCLING_LIMIT", "10")))
    args = parser.parse_args()

    docling_process(collection_name=args.collection, limit=args.limit)


if __name__ == "__main__":
    main()
