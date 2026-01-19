# api/services/mongo.py

import os
from pathlib import Path
from pymongo import MongoClient
from dotenv import load_dotenv

# Find the .env file relative to this file's location (in scraperdb/)
_this_dir = Path(__file__).resolve().parent  # api/services/
_project_root = _this_dir.parent.parent  # scraperdb/
_env_file = _project_root / ".env"
load_dotenv(_env_file)

_MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
_DB_NAME = "tender_db"

_client = None

def get_db():
    global _client
    if _client is None:
        _client = MongoClient(_MONGO_URI)
    return _client[_DB_NAME]
