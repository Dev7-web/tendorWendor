# api/routes/ui.py

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import FileResponse
from pathlib import Path

router = APIRouter(include_in_schema=False)


@router.get("/")
def ui():
    root = Path(__file__).resolve().parents[1]
    return FileResponse(root / "static" / "index.html")
