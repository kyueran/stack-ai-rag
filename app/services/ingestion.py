from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.core.config import Settings
from app.models.ingest import IngestFileResult

ALLOWED_CONTENT_TYPES = {"application/pdf", "application/x-pdf"}


def _document_id() -> str:
    return uuid4().hex


def _safe_filename(filename: str | None) -> str:
    candidate = (filename or "upload.pdf").strip().replace("/", "_").replace("\\", "_")
    return candidate or "upload.pdf"


def validate_pdf_upload(file: UploadFile, file_size: int, settings: Settings) -> str | None:
    filename = _safe_filename(file.filename)
    if not filename.lower().endswith(".pdf"):
        return "Only .pdf files are allowed"

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        return f"Invalid content type: {file.content_type or 'unknown'}"

    if file_size == 0:
        return "File is empty"

    if file_size > settings.max_upload_bytes:
        return f"File exceeds max size of {settings.max_upload_mb}MB"

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
