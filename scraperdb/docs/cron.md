# Cron Setup (Pipeline + Rebuild)

## Pipeline (every 30 minutes)

```
*/30 * * * * cd /path/to/scraperdb/scraperdb && /usr/bin/python -m app.pipeline run --lock-file data/pipeline.lock >> logs/pipeline.log 2>&1
```

Notes:
- `--lock-file` prevents overlapping runs.
- Create `logs/` ahead of time.

## Rebuild Chroma (monthly)

```
0 3 1 * * cd /path/to/scraperdb/scraperdb && /usr/bin/python embeddings/index_tenders.py --rebuild --limit 0 >> logs/rebuild.log 2>&1
```
