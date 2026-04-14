import re

from app.services.retrieval import RetrievedChunk
from app.services.tokenizer import tokenize

SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


class EvidenceChecker:
    def __init__(self, sentence_support_threshold: float = 0.2) -> None:
        self.sentence_support_threshold = sentence_support_threshold

    def filter_answer(self, answer: str, evidence: list[RetrievedChunk]) -> tuple[str, list[str]]:
        sentences = [segment.strip() for segment in SENTENCE_SPLIT_PATTERN.split(answer) if segment.strip()]
        if not sentences:
            return answer.strip(), []

        evidence_tokens = set()
        for chunk in evidence:
            evidence_tokens.update(tokenize(chunk.text))

        kept_sentences: list[str] = []
        unsupported: list[str] = []

        for sentence in sentences:
            sentence_tokens = set(tokenize(sentence))
            if not sentence_tokens:
                kept_sentences.append(sentence)
                continue

            overlap = sentence_tokens & evidence_tokens
            support = len(overlap) / max(len(sentence_tokens), 1)
            if support >= self.sentence_support_threshold:
                kept_sentences.append(sentence)
            else:
                unsupported.append(sentence)

        filtered_answer = " ".join(kept_sentences).strip()
        return filtered_answer, unsupported
