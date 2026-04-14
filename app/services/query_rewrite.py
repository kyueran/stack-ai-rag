import re
from dataclasses import dataclass

from app.services.tokenizer import tokenize

FILLER_PATTERN = re.compile(r"\b(please|could you|can you|would you|tell me|explain)\b", re.IGNORECASE)
ABBREVIATION_EXPANSIONS = {
    "rag": "retrieval augmented generation",
    "llm": "large language model",
    "api": "application programming interface",
}


@dataclass(frozen=True)
class QueryRewriteResult:
    original_query: str
    rewritten_query: str
    expansion_terms: list[str]


class QueryRewriter:
    def rewrite(self, query: str) -> QueryRewriteResult:
        cleaned = self._normalize(query)
        tokens = tokenize(cleaned)

        expansion_terms: list[str] = []
        for token in tokens:
            if token in ABBREVIATION_EXPANSIONS:
                expansion_terms.extend(tokenize(ABBREVIATION_EXPANSIONS[token]))

        rewritten_tokens = tokens + [term for term in expansion_terms if term not in tokens]
        rewritten_query = " ".join(rewritten_tokens) if rewritten_tokens else cleaned

        return QueryRewriteResult(
            original_query=query,
            rewritten_query=rewritten_query,
            expansion_terms=expansion_terms,
        )

    def _normalize(self, query: str) -> str:
        no_filler = FILLER_PATTERN.sub(" ", query)
        compact = re.sub(r"\s+", " ", no_filler)
        return compact.strip().lower()
