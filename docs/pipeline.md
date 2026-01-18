# Clean Pipeline

This document summarizes the optimized pipeline additions and provides a clean,
end-to-end flow for TenderVendor.

## Pipeline Additions (Optimized Components)

- Smart section extraction for profile PDFs to reduce processing size.
  - `embeddings/smart_extractor.py`
- Parallel PDF processing with Docling + optional file cache.
  - `embeddings/parallel_processor.py`
  - `embeddings/cache.py`
- Multi-vector profile representation (representative embeddings + mean/max).
  - `embeddings/profile_embeddings.py`
  - `api/services/profile_ingest.py`
  - `embeddings/index_profiles.py`
- Optional LLM extraction for structured profile data + filter metadata.
  - `embeddings/llm_extractor.py`
- Hybrid search with keyword boosting + score fusion.
  - `api/services/hybrid_search.py`
  - `api/services/search.py`
- Async profile ingest + match job worker.
  - `api/services/jobs.py`
  - `api/services/profile_ingest.py`
  - `api/services/match.py`
- Pipeline runner + lock file support for tenders.
  - `app/pipeline/run.py`
  - `docs/cron.md`
- Pipeline configuration toggles.
  - `scraperdb/.env.example`

## Clean Pipeline Flow

### A) Tender Ingestion Pipeline

1. Scrape tenders.
   - `main.py` (calls `scrapers/mha/mha_scraper.py`)
2. Docling extraction for tender documents.
   - `docling_processor.py` -> `docling_outputs` (doc_type="tender")
3. Chunk + embed + index tenders in Chroma.
   - `embeddings/index_tenders.py` -> `tender_embeddings`
4. Search uses tender embeddings (legacy or hybrid via profile data).
   - `api/services/search.py`

Run:
```
python -m app.pipeline run --lock-file data/pipeline.lock --limit 50
```

### B) Profile Ingestion + Matching Pipeline

1. Upload profile PDFs.
   - `api/routes/profiles.py` creates `company_documents` + `company_profiles`
2. Enqueue job and process asynchronously.
   - `api/services/jobs.py` -> `api/services/profile_ingest.py`
3. Docling + smart extraction + optional cache.
   - `embeddings/parallel_processor.py`
4. Chunk + embed + multi-vector representation.
   - `embeddings/profile_embeddings.py`
5. Persist to Mongo + Chroma, update profile status.
   - `api/services/profile_ingest.py`
6. Search uses hybrid multi-vector logic if available.
   - `api/services/search.py`
   - `api/services/hybrid_search.py`

Batch indexing (if needed):
```
python embeddings/index_profiles.py --limit 10
```

### C) Maintenance / Rebuild

Monthly rebuild of tender embeddings:
```
python embeddings/index_tenders.py --rebuild --limit 0
```

Reindex all profiles to multi-vector format:
```
python embeddings/index_profiles.py --reindex
```
