from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime, timedelta
import os

load_dotenv()

mongo_uri = os.getenv("MONGO_URI") or "mongodb://localhost:27017"
client = MongoClient(mongo_uri)
db = client["tender_db"]
collection = db["raw_tenders"]

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

def upsert_tender(data):
    """
    One tender = one document
    Stable across re-scrapes
    """

    filter_query = {
        "source": data["source"],
        "tender_ref_no": data["tender_ref_no"]
    }

    expires_at = datetime.utcnow() + timedelta(days=TTL_DAYS)
    update_data = {
        "$set": {
            **data,
            "updated_at": datetime.utcnow()
        },
        "$setOnInsert": {
            "created_at": datetime.utcnow(),
            "expires_at": expires_at,
        }
    }

    result = collection.update_one(
        filter_query,
        update_data,
        upsert=True
    )

    if result.upserted_id:
        return result.upserted_id

    doc = collection.find_one(filter_query, {"_id": 1})
    return doc["_id"]
