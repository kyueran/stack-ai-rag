import json
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.core.config import Settings
from app.models.ingest import IngestFileResult

ALLOWED_CONTENT_TYPES = {"application/pdf", "application/x-pdf"}
PDF_MAGIC_PREFIX = b"%PDF-"


def _document_id() -> str:
    return uuid4().hex


def _safe_filename(filename: str | None) -> str:
    candidate = (filename or "upload.pdf").strip().replace("/", "_").replace("\\", "_")
    return candidate or "upload.pdf"


def validate_pdf_upload(file: UploadFile, file_bytes: bytes, settings: Settings) -> str | None:
    file_size = len(file_bytes)
    filename = _safe_filename(file.filename)
    if not filename.lower().endswith(".pdf"):
        return "Only .pdf files are allowed"

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        return f"Invalid content type: {file.content_type or 'unknown'}"

    if file_size == 0:
        return "File is empty"

    if file_size > settings.max_upload_bytes:
        return f"File exceeds max size of {settings.max_upload_mb}MB"

    if not file_bytes.startswith(PDF_MAGIC_PREFIX):
        return "File does not match PDF signature"

    return None


def persist_raw_pdf(file_bytes: bytes, original_filename: str, settings: Settings) -> tuple[str, Path]:
    document_id = _document_id()
    raw_dir = settings.data_dir / "pdfs" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    output_name = f"{document_id}_{_safe_filename(original_filename)}"
    output_path = raw_dir / output_name
    output_path.write_bytes(file_bytes)
    return document_id, output_path


def build_rejected_result(file: UploadFile, reason: str) -> IngestFileResult:
    return IngestFileResult(
        filename=_safe_filename(file.filename),
        status="rejected",
        error=reason,
    )


def persist_extracted_pages(
    document_id: str,
    source_filename: str,
    page_payload: list[dict[str, object]],
    settings: Settings,
) -> Path:
    extraction_dir = settings.data_dir / "indexes" / "extracted"
    extraction_dir.mkdir(parents=True, exist_ok=True)
    target = extraction_dir / f"{document_id}.json"
    payload = {
        "document_id": document_id,
        "source_filename": source_filename,
        "pages": page_payload,
    }
    target.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return target


def persist_chunks(
    document_id: str,
    source_filename: str,
    chunks: list[dict[str, object]],
    settings: Settings,
) -> Path:
    chunk_dir = settings.data_dir / "indexes" / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    target = chunk_dir / f"{document_id}.json"
    payload = {
        "document_id": document_id,
        "source_filename": source_filename,
        "chunks": chunks,
    }
    target.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return target


def cleanup_document_artifacts(document_id: str, raw_path: Path, settings: Settings) -> None:
    extraction_path = settings.data_dir / "indexes" / "extracted" / f"{document_id}.json"
    chunk_path = settings.data_dir / "indexes" / "chunks" / f"{document_id}.json"
    for candidate in (raw_path, extraction_path, chunk_path):
        if candidate.exists():
            candidate.unlink()


def clear_all_ingestion_artifacts(settings: Settings) -> int:
    removed_files = 0
    candidate_directories = (
        settings.data_dir / "pdfs" / "raw",
        settings.data_dir / "indexes" / "extracted",
        settings.data_dir / "indexes" / "chunks",
    )
    for directory in candidate_directories:
        if not directory.exists():
            continue
        for candidate in directory.glob("*"):
            if candidate.is_file():
                candidate.unlink()
                removed_files += 1
    return removed_files
