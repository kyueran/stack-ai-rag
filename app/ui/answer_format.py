from __future__ import annotations

import re
from typing import Literal

from markupsafe import Markup, escape

from app.models.query import Citation

RichSourcePattern = re.compile(r"\[source:([a-zA-Z0-9_-]+)(?:\s+pages?\s+(\d+)-(\d+))?\]")
SourceGroupPattern = re.compile(r"\[(?:\s*source:[a-zA-Z0-9_-]+\s*,\s*)*source:[a-zA-Z0-9_-]+\s*\]")
StandaloneSourcePattern = re.compile(r"source:([a-zA-Z0-9_-]+)")
SourceTokenPattern = re.compile(
    r"\[(?:\s*source:[a-zA-Z0-9_-]+\s*,\s*)*source:[a-zA-Z0-9_-]+\s*\]"
    r"|\[source:[a-zA-Z0-9_-]+(?:\s+pages?\s+\d+-\d+)?\]"
    r"|source:[a-zA-Z0-9_-]+"
)
SourceIdPattern = re.compile(r"source:([a-zA-Z0-9_-]+)")


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

    for match in SourceTokenPattern.finditer(text):
        chunks.append(_format_markdown_segment(text[last : match.start()]))

        token = match.group(0)
        if SourceGroupPattern.fullmatch(token):
            chunks.append(_render_source_group(token, citation_lookup))
        elif rich_match := RichSourcePattern.fullmatch(token):
            chunk_id = rich_match.group(1)
            page_start = int(rich_match.group(2) or 1)
            page_end = int(rich_match.group(3) or page_start)
            chunks.append(
                _render_source_reference(
                    chunk_id=chunk_id,
                    citation_lookup=citation_lookup,
                    fallback_page_start=page_start,
                    fallback_page_end=page_end,
                )
            )
        elif standalone_match := StandaloneSourcePattern.fullmatch(token):
            chunks.append(
                _render_source_reference(
                    chunk_id=standalone_match.group(1),
                    citation_lookup=citation_lookup,
                )
            )
        else:
            chunks.append(_format_markdown_segment(token))

        last = match.end()

    chunks.append(_format_markdown_segment(text[last:]))
    return Markup("".join(str(piece) for piece in chunks))


def _render_source_group(text: str, citation_lookup: dict[str, Citation]) -> Markup:
    chunk_ids = [match.group(1) for match in SourceIdPattern.finditer(text)]
    if not chunk_ids:
        return _format_markdown_segment(text)
    links = [
        _render_source_reference(chunk_id=chunk_id, citation_lookup=citation_lookup, css_class="source-link source-chip")
        for chunk_id in chunk_ids
    ]
    return Markup(f'<span class="source-group">{" ".join(str(link) for link in links)}</span>')


def _render_source_reference(
    chunk_id: str,
    citation_lookup: dict[str, Citation],
    *,
    fallback_page_start: int | None = None,
    fallback_page_end: int | None = None,
    css_class: str = "source-link",
) -> Markup:
    page_start: int | None
    page_end: int | None
    document_id: str | None
    citation = citation_lookup.get(chunk_id)
    if citation is not None:
        page_start = citation.page_start
        page_end = citation.page_end
        document_id = citation.document_id
    else:
        page_start = fallback_page_start
        page_end = fallback_page_end or fallback_page_start
        document_id = None

    source_name = citation.source_filename if citation is not None and citation.source_filename else chunk_id
    label = f"source:{source_name}"
    if page_start is not None and page_end is not None:
        label = f"{label} p{page_start}-{page_end}"

    if document_id:
        href = f"/ui/document/{document_id}?page={page_start}"
        return Markup(
            f'<a class="{escape(css_class)}" href="{escape(href)}" target="_blank" rel="noopener noreferrer">{escape(label)}</a>'
        )
    return Markup(f'<span class="{escape(css_class)} source-plain">{escape(label)}</span>')


def _format_markdown_segment(text: str) -> Markup:
    escaped = str(escape(text))
    escaped = re.sub(r"\*\*([^\*\n][^\n]*?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", escaped)
    escaped = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", escaped)
    return Markup(escaped)
