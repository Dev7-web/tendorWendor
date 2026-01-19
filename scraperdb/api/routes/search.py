# api/routes/search.py

from __future__ import annotations

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import Optional
from datetime import datetime, timezone
import hashlib
import os

from api.services.mongo import get_db
from api.services.jobs import enqueue_job
from api.services.match import match_search as run_match_search

router = APIRouter()

PROFILE_SYNC_MAX_MB = float(os.getenv("PROFILE_SYNC_MAX_MB", "5"))


@router.post("/search/match")
async def match_search(
    profile_pdf: UploadFile = File(...),
    query: Optional[str] = Form(None),
    top_k: int = Form(5),
    location: Optional[str] = Form(None),
    dept: Optional[str] = Form(None),
    deadline: Optional[str] = Form(None),
):
    content = await profile_pdf.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty upload")
    if not (profile_pdf.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF allowed")

    file_hash = hashlib.sha256(content).hexdigest()

    base_dir = os.path.join("data", "pdfs", "PROFILE_SEARCH", file_hash)
    os.makedirs(base_dir, exist_ok=True)
    filename = os.path.basename(profile_pdf.filename or "profile.pdf")
    save_path = os.path.join(base_dir, filename)
    if not os.path.exists(save_path):
        with open(save_path, "wb") as out:
            out.write(content)

    filters = {"location": location, "dept": dept, "deadline": deadline}
    size_mb = len(content) / (1024 * 1024)

    safe_top_k = max(1, int(top_k or 5))

    if size_mb <= PROFILE_SYNC_MAX_MB:
        results = run_match_search(
            file_hash=file_hash,
            file_path=save_path,
            query=query,
            filters=filters,
            top_k=safe_top_k,
        )
        if results is None:
            raise HTTPException(status_code=500, detail="Profile processing failed")
        return {"mode": "sync", "file_hash": file_hash, "results": results}

    db = get_db()
    jobs = db["jobs"]
    now = datetime.now(timezone.utc)

    job = {
        "type": "match_search",
        "file_hash": file_hash,
        "file_path": save_path,
        "query": query,
        "filters": filters,
        "top_k": safe_top_k,
        "status": "queued",
        "step": "queued",
        "progress": 0,
        "error": None,
        "created_at": now,
        "updated_at": now,
    }
    job_res = jobs.insert_one(job)
    job_id = str(job_res.inserted_id)
    enqueue_job(job_id)

    return {"mode": "async", "file_hash": file_hash, "job_id": job_id}
