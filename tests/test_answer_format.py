from typing import cast

from app.models.query import Citation
from app.ui.answer_format import build_answer_view


def test_build_answer_view_list_and_source_linkification() -> None:
    citations = [
        Citation(
            chunk_id="abc123",
            document_id="0123456789abcdef0123456789abcdef",
            source_filename="policy.pdf",
            page_start=3,
            page_end=3,
            score=0.9,
            snippet="sample",
        )
    ]
    answer = "- First point [source:abc123 pages 3-3] - Second point"
    view = build_answer_view(answer, citations, "list")

    assert view["mode"] == "list"
    list_items = cast(list[object], view["list_items"])
    first_item = str(list_items[0])
    assert "/ui/document/0123456789abcdef0123456789abcdef?page=3" in first_item
    assert "source:policy.pdf p3-3" in first_item


def test_build_answer_view_paragraph_mode_preserves_blocks() -> None:
    citations: list[Citation] = []
    answer = "Paragraph one.\n\nParagraph two."
    view = build_answer_view(answer, citations, "paragraph")

    assert view["mode"] == "paragraph"
    paragraphs = cast(list[object], view["paragraphs"])
    assert len(paragraphs) == 2


def test_build_answer_view_renders_inline_markdown() -> None:
    citations: list[Citation] = []
    answer = "This has **bold** and *italic* and `code`."
    view = build_answer_view(answer, citations, "paragraph")

    paragraphs = cast(list[object], view["paragraphs"])
    paragraph = str(paragraphs[0])
    assert "<strong>bold</strong>" in paragraph
    assert "<em>italic</em>" in paragraph
    assert "<code>code</code>" in paragraph


def test_build_answer_view_formats_grouped_sources() -> None:
    citations = [
        Citation(
            chunk_id="abc123",
            document_id="0123456789abcdef0123456789abcdef",
            page_start=2,
            page_end=2,
            score=0.9,
            snippet="sample",
        ),
        Citation(
            chunk_id="def456",
            document_id="fedcba9876543210fedcba9876543210",
            page_start=5,
            page_end=6,
            score=0.88,
            snippet="sample two",
        ),
    ]
    answer = "Summary [source:abc123,source:def456]."
    view = build_answer_view(answer, citations, "paragraph")

    paragraphs = cast(list[object], view["paragraphs"])
    paragraph = str(paragraphs[0])
    assert "source-chip" in paragraph
    assert "/ui/document/0123456789abcdef0123456789abcdef?page=2" in paragraph
    assert "/ui/document/fedcba9876543210fedcba9876543210?page=5" in paragraph
