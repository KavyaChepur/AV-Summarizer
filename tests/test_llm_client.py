import pytest

from errors import LLMError
from llm import client

RPM = Exception("429 RESOURCE_EXHAUSTED. Please retry in 3.5s. 'retryDelay': '3s'")
DAILY = Exception("429 RESOURCE_EXHAUSTED. Please retry in 14h14m7.07s.")
NOT_FOUND = Exception("404 NOT_FOUND. models/x is no longer available")


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(client.time, "sleep", sleeps.append)
    return sleeps


def _flaky(*outcomes):
    """A call that raises/returns the given outcomes in order."""
    it = iter(outcomes)

    def call():
        o = next(it)
        if isinstance(o, Exception):
            raise o
        return o
    return call


def test_retry_delay_parsing():
    assert client._retry_delay(RPM) == 3.5
    assert client._retry_delay(DAILY) == pytest.approx(14 * 3600 + 14 * 60 + 7.07)
    assert client._retry_delay(Exception("boom")) is None


def test_waits_out_per_minute_limit(no_sleep):
    assert client._with_retries(_flaky(RPM, RPM, "ok"), retries=1, what="x") == "ok"
    assert no_sleep == [4.5, 4.5]


def test_daily_quota_fails_fast(no_sleep):
    with pytest.raises(LLMError, match="resets in about 14h 14m"):
        client._with_retries(_flaky(DAILY, "ok"), retries=3, what="x")
    assert no_sleep == []


def test_unknown_model_fails_fast():
    with pytest.raises(LLMError, match="isn't available for your API key"):
        client._with_retries(_flaky(NOT_FOUND, "ok"), retries=3, what="x")


def test_other_errors_retry_then_give_up(no_sleep):
    assert client._with_retries(_flaky(Exception("net"), "ok"), retries=2, what="x") == "ok"
    with pytest.raises(LLMError, match="failed after 2 attempts: net"):
        client._with_retries(_flaky(Exception("net"), Exception("net")), retries=2, what="x")


# ---------------------------------------------------------------- model fallback
BUSY = Exception("503 UNAVAILABLE. This model is currently experiencing high demand.")


@pytest.fixture
def models(monkeypatch):
    monkeypatch.setattr(client.config, "GEMINI_MODEL", "main")
    monkeypatch.setattr(client.config, "GEMINI_FALLBACK_MODELS", ["fb1", "fb2"])
    monkeypatch.setattr(client, "_active", (None, 0.0))


def _per_model(**outcomes):
    """make_call for _with_fallback: each model gets its own outcome sequence."""
    calls = []
    its = {m: iter(o) for m, o in outcomes.items()}

    def make_call(model):
        def call():
            calls.append(model)
            o = next(its[model])
            if isinstance(o, Exception):
                raise o
            return o
        return call
    return make_call, calls


def test_overloaded_model_switches_to_fallback(models):
    make_call, calls = _per_model(main=[BUSY, BUSY], fb1=["ok"])
    assert client._with_fallback(make_call, retries=3, what="x") == "ok"
    assert calls == ["main", "main", "fb1"]
    # the next request starts on the model that worked
    assert client.model_chain() == ["fb1", "main", "fb2"]


def test_overload_retried_once_on_same_model(models):
    make_call, calls = _per_model(main=[BUSY, "ok"])
    assert client._with_fallback(make_call, retries=3, what="x") == "ok"
    assert calls == ["main", "main"]
    assert client.model_chain()[0] == "main"


def test_quota_and_missing_model_skip_to_next(models):
    make_call, calls = _per_model(main=[DAILY], fb1=[NOT_FOUND], fb2=["ok"])
    assert client._with_fallback(make_call, retries=3, what="x") == "ok"
    assert calls == ["main", "fb1", "fb2"]


def test_all_busy_waits_and_tries_another_round(models, no_sleep):
    make_call, calls = _per_model(main=[BUSY] * 2 + ["ok"], fb1=[BUSY] * 2, fb2=[BUSY] * 2)
    assert client._with_fallback(make_call, retries=3, what="x") == "ok"
    assert client.OVERLOAD_ROUND_WAIT in no_sleep


def test_all_models_fail_gives_clear_error(models):
    make_call, _ = _per_model(main=[BUSY] * 9, fb1=[BUSY] * 9, fb2=[BUSY] * 9)
    with pytest.raises(LLMError, match="failed on every model"):
        client._with_fallback(make_call, retries=3, what="x")


def test_invalid_key_does_not_fall_back(models):
    make_call, calls = _per_model(main=[Exception("400 API_KEY_INVALID")], fb1=["ok"])
    with pytest.raises(LLMError, match="rejected the API key"):
        client._with_fallback(make_call, retries=3, what="x")
    assert calls == ["main"]
