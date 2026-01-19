# api/services/jobs.py

from __future__ import annotations

import threading
import queue
from datetime import datetime, timezone
from bson import ObjectId

from api.services.mongo import get_db
from api.services.profile_ingest import process_profile_job
from api.services.match import process_match_job

_job_queue: "queue.Queue[str]" = queue.Queue()
_worker_started = False
_worker_lock = threading.Lock()

def enqueue_job(job_id: str) -> None:
    _job_queue.put(job_id)

def start_worker() -> None:
    global _worker_started
    with _worker_lock:
        if _worker_started:
            return
        t = threading.Thread(target=_worker_loop, daemon=True)
        t.start()
        _worker_started = True

def _worker_loop():
    while True:
        job_id = _job_queue.get()
        try:
            db = get_db()
            jobs = db["jobs"]
            job = jobs.find_one({"_id": ObjectId(job_id)})
            job_type = (job or {}).get("type", "profile_ingest")

            if job_type == "match_search":
                process_match_job(job_id)
            elif job_type == "profile_ingest":
                process_profile_job(job_id)
            else:
                jobs.update_one(
                    {"_id": ObjectId(job_id)},
                    {
                        "$set": {
                            "status": "failed",
                            "step": "error",
                            "error": f"Unknown job type: {job_type}",
                            "updated_at": datetime.now(timezone.utc),
                        }
                    },
                )
        except Exception as exc:
            # Last-resort job failure update
            db = get_db()
            jobs = db["jobs"]
            try:
                jobs.update_one(
                    {"_id": ObjectId(job_id)},
                    {
                        "$set": {
                            "status": "failed",
                            "step": "error",
                            "error": str(exc),
                            "updated_at": datetime.now(timezone.utc),
                        }
                    },
                )
            except Exception:
                pass
        finally:
            _job_queue.task_done()
