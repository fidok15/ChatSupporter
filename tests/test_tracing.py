import pytest

import tracing


def test_anonymizer_masks_pii():
    raw = {
        "messages": [
            {"role": "user", "content": "Zamówienie gpu-7f3k-9qxa, mail Jan.Kowalski@example.com, ul. Długa 5, 80-001 Gdańsk"},
            {"role": "assistant", "tool_calls": [{"function": {"arguments": '{"name": "Jan Kowalski", "address": "ul. Długa 5"}'}}]},
        ]
    }
    masked = str(tracing.ANONYMIZE(raw))
    for secret in ["gpu-7f3k-9qxa", "Jan.Kowalski@example.com", "Długa", "80-001", "Jan Kowalski"]:
        assert secret not in masked, secret
    assert "<EMAIL>" in masked and "<ORDER>" in masked


def test_tracing_is_off_by_default():
    if tracing.ENABLED:
        pytest.skip("LANGSMITH_TRACING=true in this shell")
    fn = lambda: 1  # noqa: E731
    assert tracing.traced(fn) is fn
    tracing.tag_blocked("x")  # no-op, must not raise
