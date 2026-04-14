from app.services.pdf_extract import sanitize_extracted_text


def test_sanitize_extracted_text_removes_surrogates_and_nulls() -> None:
    raw = f"alpha{chr(0xD800)}beta\x00gamma"
    cleaned = sanitize_extracted_text(raw)

    assert cleaned == "alphabeta gamma"
    assert "\x00" not in cleaned
    assert all(not (0xD800 <= ord(ch) <= 0xDFFF) for ch in cleaned)
