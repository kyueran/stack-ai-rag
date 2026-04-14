from typing import Literal

from app.services.retrieval import RetrievedChunk

OutputFormat = Literal["paragraph", "list", "table"]


def build_system_prompt(intent: str, output_format: OutputFormat) -> str:
    format_instruction = {
        "paragraph": "Respond as a concise paragraph with citations.",
        "list": "Respond as a concise bullet list. Each bullet must be grounded in evidence.",
        "table": "Respond as a compact markdown table with columns: Claim | Evidence.",
    }[output_format]

    if intent == "knowledge_lookup":
        return (
            "You are a retrieval-grounded assistant. "
            "Use only provided evidence, cite source IDs, and refuse unsupported claims. "
            f"{format_instruction}"
        )

    if intent == "chitchat":
        return "You are a concise and friendly assistant."

    return "You are a safe assistant. Refuse disallowed requests clearly and briefly."


def build_user_prompt(query: str, evidence: list[RetrievedChunk], output_format: OutputFormat) -> str:
    context_lines = [
        f"[source:{item.chunk_id} pages {item.page_start}-{item.page_end}] {item.text}"
        for item in evidence
    ]
    return (
        f"Question: {query}\n"
        f"Requested format: {output_format}\n"
        "Evidence:\n"
        + "\n".join(context_lines)
        + "\n\nInstructions: Cite source IDs inline like [source:chunk-id]."
    )
