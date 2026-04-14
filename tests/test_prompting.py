from app.services.prompting import build_system_prompt


def test_prompt_template_switches_by_output_shape() -> None:
    paragraph_prompt = build_system_prompt("knowledge_lookup", "paragraph")
    list_prompt = build_system_prompt("knowledge_lookup", "list")
    table_prompt = build_system_prompt("knowledge_lookup", "table")

    assert "paragraph" in paragraph_prompt.lower()
    assert "bullet" in list_prompt.lower()
    assert "table" in table_prompt.lower()
