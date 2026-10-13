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


def test_anonymizer_dicts_keys_and_addresses():
    a = tracing.ANONYMIZE
    assert "Jan K" not in str(a({"name": "Jan K"}))
    assert "Jan K" not in str(a({"customer_name": "Jan K"}))
    assert "create_order" in str(a({"function": {"name": "create_order"}}))
    for addr in ["Aleje Jerozolimskie 5/3", "ulica Długa 5", "plac Zbawiciela 1"]:
        out = str(a({"content": f"wyślij na {addr}, Warszawa"}))
        assert addr.split()[1] not in out, addr
    call = {"function": {"name": "create_order", "arguments": '{"name": "Jan", "address": "Długa 5", "email": "a@b.pl"}'}}
    out = str(a(call))
    assert "create_order" in out and "Jan" not in out and "Długa" not in out and "a@b.pl" not in out
