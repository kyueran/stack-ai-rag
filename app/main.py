from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes.ingest import router as ingest_router
from app.core.config import get_settings
from app.core.runtime import get_database

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    get_database().initialize()
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(ingest_router)


@app.get("/healthz", tags=["health"])
def healthcheck() -> dict[str, str]:
    return {"status": "ok", "environment": settings.app_env}
