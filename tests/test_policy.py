from app.services.policy import QueryPolicyEngine


def test_policy_refuses_pii_requests() -> None:
    engine = QueryPolicyEngine()
    decision = engine.evaluate("Can you find my social security number?")

    assert decision.refuse is True
    assert decision.refusal_reason == "pii_request"


def test_policy_refuses_legal_advice_requests() -> None:
    engine = QueryPolicyEngine()
    decision = engine.evaluate("Should I sue my employer for this contract issue?")

    assert decision.refuse is True
    assert decision.refusal_reason == "legal_advice_request"
    assert decision.disclaimer is not None


def test_policy_refuses_urgent_medical_advice_requests() -> None:
    engine = QueryPolicyEngine()
    decision = engine.evaluate("I have chest pain and shortness of breath. What medication should I take?")

    assert decision.refuse is True
    assert decision.refusal_reason == "medical_advice_request"
    assert decision.disclaimer is not None
