from __future__ import annotations

from app.services.retrieval import RetrievedChunk
from app.services.tokenizer import tokenize

QUERY_NOISE_TOKENS = {
    "what",
    "which",
    "who",
    "when",
    "where",
    "why",
    "how",
    "tell",
    "say",
    "show",
    "give",
    "provide",
    "list",
    "table",
    "compare",
    "summarize",
    "summary",
    "explain",
    "exact",
    "details",
    "detail",
    "information",
    "about",
    "regarding",
    "per",
    "by",
    "each",
    "every",
    "all",
    "across",
    "does",
    "do",
    "did",
    "can",
    "could",
    "would",
    "should",
    "please",
    "me",
    "us",
    "our",
    "their",
    "there",
    "any",
}


def query_evidence_coverage(query: str, evidence: list[RetrievedChunk]) -> float:
    if not evidence:
        return 0.0

    query_terms = _query_signal_terms(query)
    if not query_terms:
        return 1.0

    evidence_terms: set[str] = set()
    for chunk in evidence:
        for token in tokenize(chunk.text):
            evidence_terms.update(_token_variants(token))

    matched = 0
    for term in query_terms:
        if any(variant in evidence_terms for variant in _token_variants(term)):
            matched += 1
    return matched / len(query_terms)


def _query_signal_terms(query: str) -> list[str]:
    unique_terms = sorted(set(tokenize(query)))
    filtered = [term for term in unique_terms if len(term) >= 3 and term not in QUERY_NOISE_TOKENS]
    if filtered:
        return filtered
    return [term for term in unique_terms if len(term) >= 3]


def _token_variants(token: str) -> set[str]:
    variants = {token}
    if len(token) > 4 and token.endswith("ies"):
        variants.add(token[:-3] + "y")
    if len(token) > 4 and token.endswith("ing"):
        variants.add(token[:-3])
        variants.add(token[:-3] + "e")
    if len(token) > 3 and token.endswith("ed"):
        variants.add(token[:-2])
        variants.add(token[:-1])
    if len(token) > 3 and token.endswith("es"):
        variants.add(token[:-2])
    if len(token) > 2 and token.endswith("s"):
        variants.add(token[:-1])
    return {variant for variant in variants if len(variant) >= 2}
