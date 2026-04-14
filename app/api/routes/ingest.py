from typing import Annotated, Literal

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import get_settings
from app.models.ingest import IngestFileResult, IngestResponse
from app.services.ingestion import build_rejected_result, persist_raw_pdf, validate_pdf_upload

router = APIRouter(prefix="/api/v1", tags=["ingestion"])
FilesParam = Annotated[list[UploadFile], File(...)]


@router.post("/ingest", response_model=IngestResponse)
async def ingest_files(files: FilesParam) -> IngestResponse:
    settings = get_settings()
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    if len(files) > settings.max_files_per_upload:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum allowed is {settings.max_files_per_upload}",
        )

    results: list[IngestFileResult] = []

    for upload in files:
        file_bytes = await upload.read()
        validation_error = validate_pdf_upload(upload, len(file_bytes), settings)
        if validation_error:
            results.append(build_rejected_result(upload, validation_error))
            continue

        document_id, _ = persist_raw_pdf(file_bytes, upload.filename or "upload.pdf", settings)
        results.append(
            IngestFileResult(
                filename=upload.filename or "upload.pdf",
                document_id=document_id,
                status="accepted",
                bytes_received=len(file_bytes),
            )
        )

    accepted_count = sum(1 for item in results if item.status == "accepted")
    rejected_count = len(results) - accepted_count
    status: Literal["ok", "partial_success", "error"]
    if accepted_count and rejected_count:
        status = "partial_success"
    elif accepted_count:
        status = "ok"
    else:
        status = "error"

    return IngestResponse(
        status=status,
        accepted_count=accepted_count,
        rejected_count=rejected_count,
        files=results,
    )
