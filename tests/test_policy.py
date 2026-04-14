from app.services.policy import QueryPolicyEngine


def test_policy_refuses_pii_requests() -> None:
    engine = QueryPolicyEngine()
    decision = engine.evaluate("Can you find my social security number?")

    assert decision.refuse is True
    assert decision.refusal_reason == "pii_request"


def test_policy_adds_legal_medical_disclaimer() -> None:
    engine = QueryPolicyEngine()
    decision = engine.evaluate("Should I sue my employer for this contract issue?")

    assert decision.refuse is False
    assert decision.disclaimer is not None
