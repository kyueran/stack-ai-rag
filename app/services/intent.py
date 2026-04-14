import re
from dataclasses import dataclass
from typing import Literal

IntentLabel = Literal["chitchat", "knowledge_lookup", "refusal"]

CHITCHAT_PATTERNS = [
    re.compile(r"^(hi|hello|hey|yo|good\s(morning|afternoon|evening))\b", re.IGNORECASE),
    re.compile(r"^(thanks|thank you|cool|awesome)\b", re.IGNORECASE),
]

REFUSAL_PATTERNS = [
    re.compile(r"\b(ssn|social security|credit card|cvv|password|private key)\b", re.IGNORECASE),
    re.compile(r"\b(hack|bypass|exploit|malware|phishing)\b", re.IGNORECASE),
]


@dataclass(frozen=True)
class IntentResult:
    intent: IntentLabel
    should_search: bool
    reason: str


class IntentRouter:
    def detect(self, query: str) -> IntentResult:
        normalized = query.strip()
        if not normalized:
            return IntentResult(intent="chitchat", should_search=False, reason="empty_query")

        if any(pattern.search(normalized) for pattern in CHITCHAT_PATTERNS):
            return IntentResult(intent="chitchat", should_search=False, reason="social_or_greeting")

        if any(pattern.search(normalized) for pattern in REFUSAL_PATTERNS):
            return IntentResult(intent="refusal", should_search=False, reason="policy_sensitive_request")

        return IntentResult(intent="knowledge_lookup", should_search=True, reason="knowledge_request")
