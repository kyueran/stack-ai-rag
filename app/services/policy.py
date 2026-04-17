import re
from dataclasses import dataclass

PII_PATTERN = re.compile(
    r"\b(ssn|social security|credit card|cvv|bank account|routing number|passport|driver\s?license)\b",
    re.IGNORECASE,
)
LEGAL_PATTERN = re.compile(
    r"\b(lawyer|legal advice|sue|liability|contract)\b",
    re.IGNORECASE,
)
MEDICAL_ADVICE_PATTERN = re.compile(
    r"\b("
    r"diagnos(?:e|is)|medical advice|treatment|prescription|medication|medicine|dosage|dose|symptom"
    r"|what should i take|what medicine should i take|what medication should i take"
    r")\b",
    re.IGNORECASE,
)
URGENT_MEDICAL_PATTERN = re.compile(
    r"\b("
    r"chest pain|shortness of breath|difficulty breathing|trouble breathing|heart attack|stroke"
    r")\b",
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

        if URGENT_MEDICAL_PATTERN.search(query):
            return PolicyDecision(
                refuse=True,
                refusal_reason="medical_advice_request",
                disclaimer=(
                    "I can't provide medical advice. Chest pain or shortness of breath can be an emergency. "
                    "Call 911 now or go to the nearest emergency department."
                ),
            )

        if MEDICAL_ADVICE_PATTERN.search(query):
            return PolicyDecision(
                refuse=True,
                refusal_reason="medical_advice_request",
                disclaimer=(
                    "I can't provide medical advice or medication recommendations. "
                    "Please contact a licensed clinician. If symptoms are severe or worsening, seek urgent care now."
                ),
            )

        if LEGAL_PATTERN.search(query):
            return PolicyDecision(
                refuse=True,
                refusal_reason="legal_advice_request",
                disclaimer=(
                    "I can't provide legal advice. Please consult a licensed attorney for guidance specific "
                    "to your situation."
                ),
            )

        return PolicyDecision(refuse=False)
