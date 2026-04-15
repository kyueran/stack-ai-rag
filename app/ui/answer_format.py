from __future__ import annotations

import re
from typing import Literal

from markupsafe import Markup, escape

from app.models.query import Citation

SourcePattern = re.compile(r"\[source:([a-zA-Z0-9_-]+)(?:\s+pages?\s+(\d+)-(\d+))?\]")


def build_answer_view(answer: str, citations: list[Citation], output_format: Literal["paragraph", "list", "table"]) -> dict[str, object]:
    normalized = _normalize_text(answer)
    citation_lookup = {citation.chunk_id: citation for citation in citations}

    if output_format == "list":
        items = _parse_list_items(normalized)
        return {
            "mode": "list",
            "list_items": [_linkify_sources(item, citation_lookup) for item in items],
            "paragraphs": [],
            "table": None,
        }

    if output_format == "table":
        table = _parse_markdown_table(normalized)
        if table:
            headers, rows = table
            return {
                "mode": "table",
                "table": {
                    "headers": [escape(header) for header in headers],
                    "rows": [[_linkify_sources(cell, citation_lookup) for cell in row] for row in rows],
                },
                "list_items": [],
                "paragraphs": [],
            }

    paragraphs = _parse_paragraphs(normalized)
    return {
        "mode": "paragraph",
        "paragraphs": [_linkify_sources(paragraph, citation_lookup) for paragraph in paragraphs],
        "list_items": [],
        "table": None,
    }


def _normalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    # Repair common compacted bullet formatting from LLM outputs.
    normalized = re.sub(r"(?<!\n)\s-\s(?=\*\*|[A-Za-z0-9])", "\n- ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized


def _parse_list_items(text: str) -> list[str]:
    lines = [line.strip() for line in text.split("\n")]
    items: list[str] = []
    current: list[str] = []

    for line in lines:
        if not line:
            continue
        if line.startswith("- "):
            if current:
                items.append(" ".join(current).strip())
            current = [line[2:].strip()]
            continue

        if current:
            current.append(line)
        else:
            current = [line]

    if current:
        items.append(" ".join(current).strip())

    return items or [text]


def _parse_paragraphs(text: str) -> list[str]:
    paragraphs = [segment.strip() for segment in re.split(r"\n\s*\n", text) if segment.strip()]
    return paragraphs or [text]


def _parse_markdown_table(text: str) -> tuple[list[str], list[list[str]]] | None:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    table_lines = [line for line in lines if line.startswith("|") and line.endswith("|")]
    if len(table_lines) < 2:
        return None

    headers = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
    separator = [cell.strip() for cell in table_lines[1].strip("|").split("|")]
    if not all(cell and set(cell) <= {"-", ":"} for cell in separator):
        return None

    rows: list[list[str]] = []
    for row_line in table_lines[2:]:
        rows.append([cell.strip() for cell in row_line.strip("|").split("|")])

    return headers, rows


def _linkify_sources(text: str, citation_lookup: dict[str, Citation]) -> Markup:
    chunks: list[Markup] = []
    last = 0

    for match in SourcePattern.finditer(text):
        chunks.append(Markup(escape(text[last : match.start()])))

        chunk_id = match.group(1)
        citation = citation_lookup.get(chunk_id)
        if citation is not None:
            page_start = citation.page_start
            page_end = citation.page_end
            document_id = citation.document_id
        else:
            page_start = int(match.group(2) or 1)
            page_end = int(match.group(3) or page_start)
            document_id = None

        if document_id:
            href = f"/ui/document/{document_id}?page={page_start}"
            label = f"source:{chunk_id} p{page_start}-{page_end}"
            chunks.append(
                Markup(
                    f'<a class="source-link" href="{escape(href)}" target="_blank" rel="noopener noreferrer">{escape(label)}</a>'
                )
            )
        else:
            chunks.append(Markup(escape(match.group(0))))

        last = match.end()

    chunks.append(Markup(escape(text[last:])))
    return Markup("".join(str(piece) for piece in chunks))
