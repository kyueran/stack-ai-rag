from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


@dataclass(frozen=True)
class ExtractedPage:
    page_number: int
    text: str
    char_count: int


@dataclass(frozen=True)
class PdfExtractionResult:
    success: bool
    pages: list[ExtractedPage]
    page_count: int
    text_char_count: int
    warnings: list[str]
    error: str | None = None


def extract_pdf_pages(pdf_path: Path) -> PdfExtractionResult:
    warnings: list[str] = []

    try:
        reader = PdfReader(str(pdf_path))
    except PdfReadError as exc:
        return PdfExtractionResult(
            success=False,
            pages=[],
            page_count=0,
            text_char_count=0,
            warnings=[],
            error=f"Unreadable PDF: {exc}",
        )
    except Exception as exc:  # pragma: no cover - defensive catch for parser edge-cases
        return PdfExtractionResult(
            success=False,
            pages=[],
            page_count=0,
            text_char_count=0,
            warnings=[],
            error=f"PDF parse failure: {exc}",
        )

    if reader.is_encrypted:
        warnings.append("PDF is encrypted; extraction quality may be degraded.")

    pages: list[ExtractedPage] = []
    total_chars = 0

    for page_index, page in enumerate(reader.pages, start=1):
        try:
            page_text = (page.extract_text() or "").strip()
        except Exception as exc:  # pragma: no cover - parser failures vary by PDF structure
            warnings.append(f"Failed to extract page {page_index}: {exc}")
            page_text = ""

        if not page_text:
            warnings.append(f"No extractable text on page {page_index}.")

        char_count = len(page_text)
        total_chars += char_count
        pages.append(ExtractedPage(page_number=page_index, text=page_text, char_count=char_count))

    if not pages:
        return PdfExtractionResult(
            success=False,
            pages=[],
            page_count=0,
            text_char_count=0,
            warnings=warnings,
            error="PDF has no pages.",
        )

    return PdfExtractionResult(
        success=True,
        pages=pages,
        page_count=len(pages),
        text_char_count=total_chars,
        warnings=warnings,
    )
