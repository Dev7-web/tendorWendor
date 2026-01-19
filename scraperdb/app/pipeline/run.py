from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Callable, Optional

from docling_processor import docling_process
from embeddings.index_tenders import index_tenders
from main import run_scrape

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"

logger = logging.getLogger("pipeline")


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def _try_remove_stale_lock(lock_path: str, max_age_seconds: int) -> bool:
    if max_age_seconds <= 0:
        return False
    try:
        age = time.time() - os.path.getmtime(lock_path)
    except OSError:
        return False
    if age <= max_age_seconds:
        return False
    try:
        os.remove(lock_path)
        return True
    except OSError:
        return False


def _create_lock_file(lock_path: str, max_age_seconds: int) -> bool:
    lock_dir = os.path.dirname(lock_path)
    if lock_dir:
        os.makedirs(lock_dir, exist_ok=True)

    if os.path.exists(lock_path):
        if not _try_remove_stale_lock(lock_path, max_age_seconds):
            return False

    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False

    with os.fdopen(fd, "w") as handle:
        payload = {
            "pid": os.getpid(),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        handle.write(json.dumps(payload))
    return True


def _remove_lock_file(lock_path: str) -> None:
    try:
        if lock_path and os.path.exists(lock_path):
            os.remove(lock_path)
    except OSError:
        logger.warning("Failed to remove lock file: %s", lock_path)


def _run_step(name: str, action: Callable[[], None], dry_run: bool) -> None:
    if dry_run:
        logger.info("DRY RUN: %s (skipped)", name)
        return
    logger.info("Starting step: %s", name)
    action()
    logger.info("Completed step: %s", name)


def run_pipeline(
    *,
    source: str,
    limit: int,
    dry_run: bool,
    lock_file: Optional[str],
    lock_max_age_seconds: int,
) -> int:
    _setup_logging()

    if source.lower() != "mha":
        logger.error("Unsupported source: %s (only 'mha' is wired)", source)
        return 2

    if lock_file:
        if not _create_lock_file(lock_file, lock_max_age_seconds):
            logger.warning("Pipeline already running (lock exists): %s", lock_file)
            return 0

    try:
        logger.info("Pipeline start (source=%s, limit=%s)", source, limit)
        _run_step("scrape", run_scrape, dry_run)
        _run_step(
            "docling_process",
            lambda: docling_process(collection_name="tender_documents", limit=limit),
            dry_run,
        )
        _run_step("index_tenders", lambda: index_tenders(limit=limit), dry_run)
        logger.info("Pipeline complete")
    finally:
        if lock_file:
            _remove_lock_file(lock_file)

    return 0


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="mha")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--lock-file", default=os.getenv("PIPELINE_LOCK_FILE", "data/pipeline.lock"))
    parser.add_argument("--lock-max-age", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    return run_pipeline(
        source=args.source,
        limit=args.limit,
        dry_run=args.dry_run,
        lock_file=args.lock_file,
        lock_max_age_seconds=args.lock_max_age,
    )


if __name__ == "__main__":
    sys.exit(main())
