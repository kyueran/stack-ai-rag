import re
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from app.api.routes.ingest import ingest_files
from app.api.routes.query import query_knowledge_base
from app.core.config import get_settings
from app.core.runtime import get_concept_service
from app.models.query import QueryRequest
from app.ui.answer_format import build_answer_view

router = APIRouter(tags=["ui"])
FilesParam = Annotated[list[UploadFile], File(...)]
DOCUMENT_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")

_templates = Jinja2Templates(directory=str(Path(__file__).resolve().parents[2] / "ui" / "templates"))


@router.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return _templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"title": "StackAI RAG Chat"},
    )


@router.get("/ui/concepts", response_class=HTMLResponse)
def ui_concepts(
    request: Request,
    document_id: str | None = None,
    top_n: int = 30,
) -> HTMLResponse:
    concept_service = get_concept_service()
    total_documents, concepts = concept_service.get_concepts(document_id=document_id or None, top_n=top_n)
    documents = concept_service.get_document_options()
    return _templates.TemplateResponse(
        request=request,
        name="partials/concepts_panel.html",
        context={
            "concepts": concepts,
            "documents": documents,
            "selected_document_id": document_id or "",
            "top_n": top_n,
            "total_documents": total_documents,
        },
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
) -> HTMLResponse:
    response = query_knowledge_base(QueryRequest(query=query))
    answer_view = build_answer_view(response.answer, response.citations, response.answer_format)
    return _templates.TemplateResponse(
        request=request,
        name="partials/chat_turn.html",
        context={
            "user_query": query,
            "response": response,
            "answer_view": answer_view,
        },
    )


@router.get("/ui/document/{document_id}")
def ui_document(document_id: str) -> FileResponse:
    if not DOCUMENT_ID_PATTERN.fullmatch(document_id):
        raise HTTPException(status_code=404, detail="Document not found")

    raw_dir = get_settings().data_dir / "pdfs" / "raw"
    matches = sorted(raw_dir.glob(f"{document_id}_*.pdf"))
    if not matches:
        raise HTTPException(status_code=404, detail="Document not found")

    pdf_path = matches[0]
    return FileResponse(path=pdf_path, media_type="application/pdf", filename=pdf_path.name)
