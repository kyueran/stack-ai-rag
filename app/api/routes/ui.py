from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.api.routes.ingest import ingest_files
from app.api.routes.query import query_knowledge_base
from app.models.query import QueryRequest

router = APIRouter(tags=["ui"])
FilesParam = Annotated[list[UploadFile], File(...)]

_templates = Jinja2Templates(directory=str(Path(__file__).resolve().parents[2] / "ui" / "templates"))


@router.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return _templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"title": "StackAI RAG Chat"},
    )


@router.post("/ui/ingest", response_class=HTMLResponse)
async def ui_ingest(request: Request, files: FilesParam) -> HTMLResponse:
    result = await ingest_files(files)
    status_class = {
        "ok": "ok",
        "partial_success": "partial",
        "error": "error",
    }.get(result.status, "idle")
    return _templates.TemplateResponse(
        request=request,
        name="partials/ingest_status.html",
        context={
            "status_class": status_class,
            "result": result,
        },
    )


@router.post("/ui/query", response_class=HTMLResponse)
def ui_query(
    request: Request,
    query: str = Form(...),
    output_format: Literal["paragraph", "list", "table"] = Form("paragraph"),
) -> HTMLResponse:
    response = query_knowledge_base(QueryRequest(query=query, output_format=output_format))
    return _templates.TemplateResponse(
        request=request,
        name="partials/chat_turn.html",
        context={
            "user_query": query,
            "response": response,
        },
    )
