from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime, timedelta
import os

load_dotenv()

mongo_uri = os.getenv("MONGO_URI") or "mongodb://localhost:27017"
client = MongoClient(mongo_uri)
db = client["tender_db"]
collection = db["tender_documents"]

TTL_DAYS = int(os.getenv("TENDER_TTL_DAYS", "15"))


def _ensure_ttl_index() -> None:
    try:
        collection.create_index(
            [("expires_at", 1)],
            expireAfterSeconds=0,
            name="expires_at_ttl",
        )
    except Exception:
        pass


_ensure_ttl_index()

def upsert_pdf_metadata(data):
    """
    One PDF per tender
    Safe for re-runs
    """

    filter_query = {
        "$or": [
            {
                "tender_id": data["tender_id"],
                "document_name": data["document_name"]
            },
            {
                "source": data["source"],
                "tender_ref_no": data["tender_ref_no"]
            }
        ]
    }

    insert_docling_status = data.get("docling_status", "pending")
    update_payload = {**data}
    update_payload.pop("docling_status", None)

    expires_at = datetime.utcnow() + timedelta(days=TTL_DAYS)
    update_data = {
        "$set": {
            **update_payload,
            "updated_at": datetime.utcnow()
        },
        "$setOnInsert": {
            "created_at": datetime.utcnow(),
            "docling_status": insert_docling_status,
            "expires_at": expires_at,
        }
    }

    collection.update_one(
        filter_query,
        update_data,
        upsert=True
    )
