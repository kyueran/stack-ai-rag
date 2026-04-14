import re
from dataclasses import dataclass
from hashlib import sha1

from app.services.pdf_extract import ExtractedPage

SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    document_id: str
    text: str
    page_start: int
    page_end: int
    char_count: int


def _split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    return [segment.strip() for segment in SENTENCE_BOUNDARY_PATTERN.split(normalized) if segment.strip()]


def _chunk_id(document_id: str, page_start: int, page_end: int, offset: int, text: str) -> str:
    digest = sha1(f"{document_id}:{page_start}:{page_end}:{offset}:{text}".encode()).hexdigest()
    return digest[:16]


def _with_overlap(base: str, overlap_text: str) -> str:
    if not overlap_text:
        return base
    if base.startswith(overlap_text):
        return base
    return f"{overlap_text} {base}".strip()


def chunk_pages(
    document_id: str,
    pages: list[ExtractedPage],
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    all_chunks: list[TextChunk] = []

    for page in pages:
        sentences = _split_sentences(page.text)
        if not sentences:
            continue

        current = ""
        sentence_start = 0

        for sentence_idx, sentence in enumerate(sentences):
            proposed = f"{current} {sentence}".strip() if current else sentence
            if len(proposed) <= chunk_size:
                current = proposed
                continue

            if current:
                overlap_text = current[-chunk_overlap:].strip() if chunk_overlap else ""
                chunk_text = current
                offset = max(sentence_start, 0)
                all_chunks.append(
                    TextChunk(
                        chunk_id=_chunk_id(document_id, page.page_number, page.page_number, offset, chunk_text),
                        document_id=document_id,
                        text=chunk_text,
                        page_start=page.page_number,
                        page_end=page.page_number,
                        char_count=len(chunk_text),
                    )
                )
                current = _with_overlap(sentence, overlap_text)
                sentence_start = sentence_idx
            else:
                current = sentence[:chunk_size]
                sentence_start = sentence_idx

        if current:
            offset = max(sentence_start, 0)
            all_chunks.append(
                TextChunk(
                    chunk_id=_chunk_id(document_id, page.page_number, page.page_number, offset, current),
                    document_id=document_id,
                    text=current,
                    page_start=page.page_number,
                    page_end=page.page_number,
                    char_count=len(current),
                )
            )

    return all_chunks
