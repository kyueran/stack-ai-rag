from typing import cast

from app.models.query import Citation
from app.ui.answer_format import build_answer_view


def test_build_answer_view_list_and_source_linkification() -> None:
    citations = [
        Citation(
            chunk_id="abc123",
            document_id="0123456789abcdef0123456789abcdef",
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


def test_build_answer_view_paragraph_mode_preserves_blocks() -> None:
    citations: list[Citation] = []
    answer = "Paragraph one.\n\nParagraph two."
    view = build_answer_view(answer, citations, "paragraph")

    assert view["mode"] == "paragraph"
    paragraphs = cast(list[object], view["paragraphs"])
    assert len(paragraphs) == 2
