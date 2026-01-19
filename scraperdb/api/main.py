# api/main.py

from fastapi import FastAPI
from api.routes.profiles import router as profiles_router
from api.routes.search import router as search_router
from api.routes.ui import router as ui_router
from api.services.jobs import start_worker

app = FastAPI(title="ScraperDB API", version="0.1")

app.include_router(profiles_router)
app.include_router(search_router)
app.include_router(ui_router)

@app.on_event("startup")
def _startup():
    #  Starts the in-process background worker thread
    start_worker()
