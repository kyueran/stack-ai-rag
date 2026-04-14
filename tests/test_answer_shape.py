from app.services.answer_shape import select_output_format


def test_select_output_format_returns_table_for_comparison_queries() -> None:
    fmt = select_output_format("Compare README against GP-GOMEA benchmarks", "knowledge_lookup")
    assert fmt == "table"


def test_select_output_format_returns_list_for_list_queries() -> None:
    fmt = select_output_format("List the key points from the document", "knowledge_lookup")
    assert fmt == "list"


def test_select_output_format_returns_paragraph_for_simple_question() -> None:
    fmt = select_output_format("What is README?", "knowledge_lookup")
    assert fmt == "paragraph"


def test_select_output_format_returns_paragraph_for_chitchat() -> None:
    fmt = select_output_format("hi", "chitchat")
    assert fmt == "paragraph"
