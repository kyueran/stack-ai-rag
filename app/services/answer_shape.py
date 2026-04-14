import re

from app.services.prompting import OutputFormat

LIST_HINT_PATTERN = re.compile(
    r"\b(list|steps?|key points?|top\s+\d+|bullets?|enumerate|outline|summarize|summary)\b",
    re.IGNORECASE,
)
TABLE_HINT_PATTERN = re.compile(
    r"\b(compare|comparison|versus|vs\.?|table|benchmark|metrics?|scores?|columns?|trade-?offs?|differences?)\b",
    re.IGNORECASE,
)


def select_output_format(query: str, intent: str) -> OutputFormat:
    if intent != "knowledge_lookup":
        return "paragraph"

    normalized = query.strip()
    if TABLE_HINT_PATTERN.search(normalized):
        return "table"

    if LIST_HINT_PATTERN.search(normalized):
        return "list"

    return "paragraph"
