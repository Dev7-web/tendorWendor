# embeddings/chunker.py

from __future__ import annotations

from typing import List
import re


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _tail_overlap(text: str, overlap_chars: int) -> str:
    """Take last overlap_chars but try not to cut mid-word."""
    if overlap_chars <= 0 or not text:
        return ""
    tail = text[-overlap_chars:]
    # Move to next whitespace boundary to avoid mid-word cut
    if " " in tail:
        tail = tail[tail.find(" ") + 1 :]
    return tail.strip()


def chunk_text(text: str, max_chars: int = 500, overlap_chars: int = 80) -> List[str]:
    """
    Production chunker:
    - Prefers paragraph boundaries
    - If a paragraph is too long, splits by sentences
    - Adds a small overlap to preserve context across chunks
    """
    if not text or max_chars <= 0:
        return []

    # Normalize whitespace
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: List[str] = []
    current = ""

    def flush():
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    def add_piece(piece: str):
        nonlocal current
        piece = piece.strip()
        if not piece:
            return

        if not current:
            current = piece
            return

        if len(current) + 1 + len(piece) <= max_chars:
            current += " " + piece
        else:
            # flush current chunk and start new with overlap
            prev = current
            flush()
            overlap = _tail_overlap(prev, overlap_chars)
            current = (overlap + " " + piece).strip() if overlap else piece

    for para in paragraphs:
        # If para itself is too long, break it down by sentences
        if len(para) > max_chars:
            sentences = _SENTENCE_SPLIT.split(para)
            for s in sentences:
                # if even a sentence is huge, hard-split it
                if len(s) > max_chars:
                    start = 0
                    while start < len(s):
                        part = s[start : start + max_chars]
                        add_piece(part)
                        start += max_chars
                else:
                    add_piece(s)
        else:
            add_piece(para)

    flush()
    return chunks
