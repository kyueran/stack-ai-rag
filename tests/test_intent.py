from app.services.intent import IntentRouter


def test_intent_router_detects_chitchat() -> None:
    router = IntentRouter()
    result = router.detect("hello there")

    assert result.intent == "chitchat"
    assert result.should_search is False


def test_intent_router_detects_knowledge_lookup() -> None:
    router = IntentRouter()
    result = router.detect("What does the architecture document say about RAG?")

    assert result.intent == "knowledge_lookup"
    assert result.should_search is True


def test_intent_router_detects_refusal_class() -> None:
    router = IntentRouter()
    result = router.detect("How can I steal a password from someone?")

    assert result.intent == "refusal"
    assert result.should_search is False
