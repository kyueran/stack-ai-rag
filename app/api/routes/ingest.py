from typing import Annotated, Literal

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import get_settings
from app.core.runtime import get_ingestion_repository, get_mistral_client
from app.db.repositories import ChunkRecord, EmbeddingRecord
from app.models.ingest import IngestFileResult, IngestResponse
from app.services.chunking import chunk_pages
from app.services.ingestion import (
    build_rejected_result,
    cleanup_document_artifacts,
    persist_chunks,
    persist_extracted_pages,
    persist_raw_pdf,
    validate_pdf_upload,
)
from app.services.pdf_extract import extract_pdf_pages

router = APIRouter(prefix="/api/v1", tags=["ingestion"])
FilesParam = Annotated[list[UploadFile], File(...)]


@router.post("/ingest", response_model=IngestResponse)
async def ingest_files(files: FilesParam) -> IngestResponse:
    settings = get_settings()
    ingestion_repository = get_ingestion_repository()
    mistral_client = get_mistral_client()
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
        validation_error = validate_pdf_upload(upload, file_bytes, settings)
        if validation_error:
            results.append(build_rejected_result(upload, validation_error))
            continue

        document_id, raw_path = persist_raw_pdf(file_bytes, upload.filename or "upload.pdf", settings)
        try:
            extraction = extract_pdf_pages(raw_path)
            if not extraction.success:
                results.append(
                    IngestFileResult(
                        filename=upload.filename or "upload.pdf",
                        status="rejected",
                        bytes_received=len(file_bytes),
                        document_id=document_id,
                        extraction_error=extraction.error,
                        warnings=extraction.warnings,
                        error="PDF extraction failed",
                    )
                )
                cleanup_document_artifacts(document_id, raw_path, settings)
                continue

            if extraction.page_count > settings.max_pdf_pages:
                results.append(
                    IngestFileResult(
                        filename=upload.filename or "upload.pdf",
                        status="rejected",
                        bytes_received=len(file_bytes),
                        document_id=document_id,
                        page_count=extraction.page_count,
                        error=f"PDF exceeds max page limit of {settings.max_pdf_pages}",
                    )
                )
                cleanup_document_artifacts(document_id, raw_path, settings)
                continue

            persist_extracted_pages(
                document_id=document_id,
                source_filename=upload.filename or "upload.pdf",
                page_payload=[
                    {"page_number": page.page_number, "text": page.text, "char_count": page.char_count}
                    for page in extraction.pages
                ],
                settings=settings,
            )
            chunks = chunk_pages(
                document_id=document_id,
                pages=extraction.pages,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
            persist_chunks(
                document_id=document_id,
                source_filename=upload.filename or "upload.pdf",
                chunks=[
                    {
                        "chunk_id": chunk.chunk_id,
                        "text": chunk.text,
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end,
                        "char_count": chunk.char_count,
                    }
                    for chunk in chunks
                ],
                settings=settings,
            )
            embedding_response = mistral_client.embed_texts([chunk.text for chunk in chunks])
            embeddings = [
                EmbeddingRecord(
                    chunk_id=chunk.chunk_id,
                    model=embedding_response.model,
                    vector=vector,
                )
                for chunk, vector in zip(chunks, embedding_response.vectors, strict=True)
            ]
            ingestion_repository.ingest_document_atomic(
                document_id=document_id,
                filename=upload.filename or "upload.pdf",
                byte_size=len(file_bytes),
                page_count=extraction.page_count,
                chunk_count=len(chunks),
                text_char_count=extraction.text_char_count,
                chunks=[
                    ChunkRecord(
                        chunk_id=chunk.chunk_id,
                        page_start=chunk.page_start,
                        page_end=chunk.page_end,
                        char_count=chunk.char_count,
                        text=chunk.text,
                    )
                    for chunk in chunks
                ],
                embeddings=embeddings,
            )
        except Exception as exc:
            cleanup_document_artifacts(document_id, raw_path, settings)
            error_code = exc.__class__.__name__
            results.append(
                IngestFileResult(
                    filename=upload.filename or "upload.pdf",
                    status="rejected",
                    bytes_received=len(file_bytes),
                    document_id=document_id,
                    error=f"Ingestion failed and was rolled back safely ({error_code})",
                    extraction_error=str(exc)[:220],
                )
            )
            continue
        results.append(
            IngestFileResult(
                filename=upload.filename or "upload.pdf",
                document_id=document_id,
                status="accepted",
                bytes_received=len(file_bytes),
                page_count=extraction.page_count,
                chunk_count=len(chunks),
                text_char_count=extraction.text_char_count,
                warnings=extraction.warnings,
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
