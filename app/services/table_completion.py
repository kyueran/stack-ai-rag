from __future__ import annotations

import re
from dataclasses import dataclass

from app.services.retrieval import RetrievedChunk

SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
TABLE_LINE_PATTERN = re.compile(r"^\s*\|.*\|\s*$")
EXHAUSTIVE_QUERY_HINT_PATTERN = re.compile(r"\b(all|each|every|per|by|across)\b", re.IGNORECASE)
DIMENSION_PATTERNS = (
    re.compile(r"\bper\s+([a-zA-Z][a-zA-Z0-9_-]{1,40})s?\b", re.IGNORECASE),
    re.compile(r"\bby\s+([a-zA-Z][a-zA-Z0-9_-]{1,40})s?\b", re.IGNORECASE),
    re.compile(r"\bfor each\s+([a-zA-Z][a-zA-Z0-9_-]{1,40})s?\b", re.IGNORECASE),
    re.compile(r"\beach\s+([a-zA-Z][a-zA-Z0-9_-]{1,40})s?\b", re.IGNORECASE),
    re.compile(r"\bacross\s+([a-zA-Z][a-zA-Z0-9_-]{1,40})s?\b", re.IGNORECASE),
)
STOPWORD_LABELS = {"a", "an", "the", "this", "that", "these", "those"}


@dataclass(frozen=True)
class _CategoryEvidence:
    label: str
    sentence: str
    chunk: RetrievedChunk


@dataclass(frozen=True)
class _TableBlock:
    start: int
    end: int
    lines: list[str]
    column_count: int
    text: str


def ensure_exhaustive_table_coverage(
    query: str,
    answer: str,
    evidence: list[RetrievedChunk],
) -> str:
    if not answer.strip() or not evidence:
        return answer
    if not EXHAUSTIVE_QUERY_HINT_PATTERN.search(query):
        return answer

    dimension = _infer_dimension(query)
    if not dimension:
        return answer

    table = _find_first_table_block(answer)
    if table is None or table.column_count < 2:
        return answer

    category_evidence = _extract_category_evidence(evidence, dimension)
    if len(category_evidence) < 2:
        return answer

    missing = [
        entry
        for entry in category_evidence.values()
        if not _category_is_present_in_table(table.text, entry.label, dimension)
    ]
    if not missing:
        return answer

    added_rows = [_build_claim_evidence_row(entry) for entry in missing]
    return _append_rows_to_table(answer, table, added_rows)


def _infer_dimension(query: str) -> str | None:
    for pattern in DIMENSION_PATTERNS:
        match = pattern.search(query)
        if match:
            return match.group(1).lower().rstrip("s")
    return None


def _find_first_table_block(answer: str) -> _TableBlock | None:
    lines = answer.splitlines()
    start = -1
    for idx, line in enumerate(lines):
        if TABLE_LINE_PATTERN.match(line):
            start = idx
            break
    if start < 0:
        return None

    end = start
    while end + 1 < len(lines) and TABLE_LINE_PATTERN.match(lines[end + 1]):
        end += 1
    if end - start < 2:
        return None

    table_lines = lines[start : end + 1]
    headers = _parse_table_cells(table_lines[0])
    separators = _parse_table_cells(table_lines[1])
    if len(headers) < 2 or len(headers) != len(separators):
        return None
    if not all(_is_separator_cell(cell) for cell in separators):
        return None

    return _TableBlock(
        start=start,
        end=end,
        lines=table_lines,
        column_count=len(headers),
        text="\n".join(table_lines).lower(),
    )


def _parse_table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_separator_cell(cell: str) -> bool:
    return bool(cell) and set(cell) <= {"-", ":"}


def _extract_category_evidence(
    evidence: list[RetrievedChunk],
    dimension: str,
) -> dict[str, _CategoryEvidence]:
    categories: dict[str, _CategoryEvidence] = {}
    hyphen_pattern = re.compile(rf"\b([a-zA-Z0-9][a-zA-Z0-9_-]*)-{re.escape(dimension)}s?\b", re.IGNORECASE)
    spaced_pattern = re.compile(rf"\b([a-zA-Z0-9][a-zA-Z0-9_-]*)\s+{re.escape(dimension)}s?\b", re.IGNORECASE)

    for chunk in evidence:
        for sentence in SENTENCE_SPLIT_PATTERN.split(chunk.text):
            cleaned = _clean_sentence(sentence)
            if not cleaned:
                continue
            for pattern in (hyphen_pattern, spaced_pattern):
                for match in pattern.finditer(cleaned):
                    label = match.group(1).strip(" ,.;:").lower()
                    if not label or label in STOPWORD_LABELS:
                        continue
                    if label not in categories:
                        categories[label] = _CategoryEvidence(
                            label=label,
                            sentence=cleaned,
                            chunk=chunk,
                        )

    return categories


def _clean_sentence(sentence: str) -> str:
    return re.sub(r"\s+", " ", sentence).strip()


def _category_is_present_in_table(table_text: str, label: str, dimension: str) -> bool:
    explicit_pattern = re.compile(
        rf"\b{re.escape(label)}(?:-|\s+){re.escape(dimension)}s?\b",
        re.IGNORECASE,
    )
    if explicit_pattern.search(table_text):
        return True
    return bool(re.search(rf"\b{re.escape(label)}\b", table_text, re.IGNORECASE))


def _build_claim_evidence_row(entry: _CategoryEvidence) -> str:
    citation = f"[source:{entry.chunk.chunk_id} pages {entry.chunk.page_start}-{entry.chunk.page_end}]"
    claim = _escape_table_cell(entry.sentence)
    evidence = _escape_table_cell(f"{entry.sentence} {citation}")
    return f"| {claim} | {evidence} |"


def _escape_table_cell(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value).strip()
    return normalized.replace("|", "\\|")


def _append_rows_to_table(answer: str, table: _TableBlock, rows: list[str]) -> str:
    lines = answer.splitlines()
    updated = [*lines[: table.end + 1], *rows, *lines[table.end + 1 :]]
    return "\n".join(updated)

