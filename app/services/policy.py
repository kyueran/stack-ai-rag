import re
from dataclasses import dataclass

PII_PATTERN = re.compile(
    r"\b(ssn|social security|credit card|cvv|bank account|routing number|passport|driver\s?license)\b",
    re.IGNORECASE,
)
LEGAL_MEDICAL_PATTERN = re.compile(
    r"\b(lawyer|legal advice|sue|liability|contract|diagnose|medical advice|treatment|prescription|symptom)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PolicyDecision:
    refuse: bool
    refusal_reason: str | None = None
    disclaimer: str | None = None


class QueryPolicyEngine:
    def evaluate(self, query: str) -> PolicyDecision:
        if PII_PATTERN.search(query):
            return PolicyDecision(refuse=True, refusal_reason="pii_request")

        if LEGAL_MEDICAL_PATTERN.search(query):
            return PolicyDecision(
                refuse=False,
                disclaimer=(
                    "This response is informational and not a substitute for professional "
                    "legal or medical advice."
                ),
            )

        return PolicyDecision(refuse=False)
