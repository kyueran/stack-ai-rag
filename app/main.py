from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes.ingest import router as ingest_router
from app.api.routes.query import router as query_router
from app.api.routes.ui import router as ui_router
from app.core.config import get_settings
from app.core.runtime import get_database

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    get_database().initialize()
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(ingest_router)
app.include_router(query_router)
app.include_router(ui_router)
app.mount(
    "/ui/static",
    StaticFiles(directory=str(Path(__file__).resolve().parent / "ui" / "static")),
    name="ui-static",
)


@app.get("/healthz", tags=["health"])
def healthcheck() -> dict[str, str]:
    return {"status": "ok", "environment": settings.app_env}
