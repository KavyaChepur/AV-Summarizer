"""The only module that talks to the LLM (Gemini via LangChain).

Every agent goes through `ask`, `ask_many` or `ask_structured`, so switching to
another provider (e.g. ChatOllama) only means changing `get_llm`.

Error handling:
  * per-minute rate limits (Google says "retry in N s", N small) -> wait and retry
  * invalid key -> fail at once with a clear message
  * model overloaded (503), daily quota, unknown model -> switch to the next model
    in config.GEMINI_FALLBACK_MODELS; if every model is overloaded, wait and
    go round again
  * anything else (network, empty answer) -> retry with backoff
"""
import os
import re
import threading
import time
from functools import lru_cache
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, TypeVar

from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

import config
from errors import LLMError

T = TypeVar("T", bound=BaseModel)
Prompt = Any  # str | list[BaseMessage] | PromptValue

MAX_RATE_LIMIT_WAIT = 90      # seconds we're willing to wait for a per-minute limit
MAX_RATE_LIMIT_RETRIES = 4
OVERLOAD_RETRIES = 2          # tries on one overloaded model before switching
OVERLOAD_ROUNDS = 3           # passes over the whole model list when all are busy
OVERLOAD_ROUND_WAIT = 20      # seconds between those passes
STICKY_SECONDS = 600          # keep using a fallback this long before retrying the main model


class ModelUnavailable(LLMError):
    """This model can't serve the request now; another model might.
    `transient` is True for overload (worth retrying later), False for
    quota/missing model (skip it for the rest of this request)."""

    def __init__(self, message: str, transient: bool):
        super().__init__(message)
        self.transient = transient


# The model that last worked; later calls start there instead of re-hitting a
# model that was just overloaded.
_lock = threading.Lock()
_active: Tuple[Optional[str], float] = (None, 0.0)


def model_chain() -> List[str]:
    """Models to try, in order: the one that last worked, then the main model,
    then the fallbacks (no duplicates)."""
    with _lock:
        active, since = _active
    if active and time.monotonic() - since > STICKY_SECONDS:
        active = None
    chain = [active] if active else []
    for m in [config.GEMINI_MODEL, *config.GEMINI_FALLBACK_MODELS]:
        if m not in chain:
            chain.append(m)
    return chain


def _set_active(model: str) -> None:
    global _active
    with _lock:
        if _active[0] != model:
            _active = (None, 0.0) if model == config.GEMINI_MODEL \
                else (model, time.monotonic())


@lru_cache(maxsize=16)
def get_llm(temperature: float = 0.3, model: Optional[str] = None) -> ChatGoogleGenerativeAI:
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise LLMError("GOOGLE_API_KEY is not set. Add it to your .env file.")
    # max_retries=0: retries are handled below, where we can read Google's wait time.
    return ChatGoogleGenerativeAI(model=model or config.GEMINI_MODEL, google_api_key=key,
                                  temperature=temperature, max_retries=0)


def message_text(msg: BaseMessage) -> str:
    """Plain text from a chat message whose content may be a string or blocks."""
    content = msg.content
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


# ---------------------------------------------------------------- errors
def _is_quota_error(e: Exception) -> bool:
    s = str(e)
    return "RESOURCE_EXHAUSTED" in s or " 429" in s or s.startswith("429")


def _is_key_error(e: Exception) -> bool:
    s = str(e)
    return any(k in s for k in ("API_KEY_INVALID", "API key not valid", "PERMISSION_DENIED"))


def _is_model_missing(e: Exception) -> bool:
    s = str(e)
    return "NOT_FOUND" in s or "is not found for API version" in s


def _is_overloaded(e: Exception) -> bool:
    s = str(e)
    return "UNAVAILABLE" in s or "503" in s or "overloaded" in s.lower() \
        or "high demand" in s.lower()


def _is_fatal(e: Exception) -> bool:
    return _is_key_error(e) or _is_model_missing(e)


def _retry_delay(e: Exception) -> Optional[float]:
    """Seconds Google asks us to wait, if it says so."""
    s = str(e)
    m = re.search(r"retry in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)s", s, re.I)
    if m:
        return int(m.group(1) or 0) * 3600 + int(m.group(2) or 0) * 60 + float(m.group(3))
    m = re.search(r"retryDelay['\"]?:\s*['\"](\d+(?:\.\d+)?)s", s)
    return float(m.group(1)) if m else None


def _fmt_wait(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    return f"{h}h {rem // 60}m" if h else (f"{rem // 60}m" if rem >= 60 else f"{rem}s")


def describe_error(e: Optional[Exception], model: Optional[str] = None) -> str:
    """Short, readable version of a Gemini/LangChain error."""
    if e is None:
        return "unknown error"
    if isinstance(e, LLMError):
        return str(e)
    model = model or config.GEMINI_MODEL
    s = str(e)
    if _is_quota_error(e):
        msg = f"Gemini quota exceeded for '{model}'"
        delay = _retry_delay(e)
        if delay:
            msg += f"; it resets in about {_fmt_wait(delay)}"
        return msg + (". Wait, choose another model with GEMINI_MODEL in .env, "
                      "or enable billing for your API key.")
    if "API_KEY_INVALID" in s or "API key not valid" in s or "PERMISSION_DENIED" in s:
        return "Gemini rejected the API key. Check GOOGLE_API_KEY in your .env file."
    if _is_overloaded(e):
        return (f"Gemini ('{model}') is overloaded right now (503). "
                "This is usually temporary; try again in a minute.")
    if _is_model_missing(e):
        return (f"The model '{model}' isn't available for your API key. "
                "Set GEMINI_MODEL in .env to a current model (e.g. gemini-3.5-flash).")
    first = s.strip().splitlines()[0] if s.strip() else type(e).__name__
    return first[:300]


def _notify(message: str) -> None:
    """Show a live message in the UI/CLI if we're running inside the graph."""
    try:
        from langgraph.config import get_stream_writer
        get_stream_writer()({"agent": "llm", "message": message, "progress": None})
    except Exception:
        pass


def _with_retries(call: Callable[[], Any], *, retries: int, what: str,
                  model: Optional[str] = None) -> Any:
    """Run `call` against one model; it returns a result or raises. Returns the
    first good result. Raises ModelUnavailable when another model should be tried."""
    last: Optional[Exception] = None
    attempt = rate_waits = overloads = 0
    while attempt < retries:
        try:
            return call()
        except ModelUnavailable:
            raise
        except LLMError as e:   # e.g. empty answer: retry normally
            last = e
        except Exception as e:
            if _is_key_error(e):
                raise LLMError(describe_error(e, model)) from e
            if _is_model_missing(e):
                raise ModelUnavailable(describe_error(e, model), transient=False) from e
            if _is_quota_error(e):
                delay = _retry_delay(e)
                if delay is None or delay > MAX_RATE_LIMIT_WAIT \
                        or rate_waits >= MAX_RATE_LIMIT_RETRIES:
                    raise ModelUnavailable(describe_error(e, model), transient=False) from e
                rate_waits += 1
                _notify(f"Gemini rate limit reached; waiting {delay:.0f}s…")
                time.sleep(delay + 1)
                continue  # doesn't count as a failed attempt
            if _is_overloaded(e):
                overloads += 1
                if overloads >= OVERLOAD_RETRIES:
                    raise ModelUnavailable(describe_error(e, model), transient=True) from e
            last = e
        attempt += 1
        if attempt < retries:
            time.sleep(3 * 2 ** (attempt - 1))  # 3 s, 6 s, 12 s…
    raise LLMError(f"{what} failed after {retries} attempts: {describe_error(last, model)}")


def _with_fallback(make_call: Callable[[str], Callable[[], Any]], *, retries: int,
                   what: str) -> Any:
    """Run the request on the first model in model_chain() that can serve it.
    Overloaded models are retried in later rounds; models that are out of quota
    or missing are skipped for the rest of this request."""
    models = model_chain()
    skipped: set = set()
    errors: List[str] = []
    for rnd in range(OVERLOAD_ROUNDS):
        for i, model in enumerate(models):
            if model in skipped:
                continue
            try:
                result = _with_retries(make_call(model), retries=retries, what=what,
                                       model=model)
            except ModelUnavailable as e:
                errors.append(str(e))
                if not e.transient:
                    skipped.add(model)
                nxt = next((m for m in models[i + 1:] if m not in skipped), None)
                if nxt:
                    _notify(f"'{model}' unavailable; switching to '{nxt}'…")
                continue
            _set_active(model)
            return result
        if len(skipped) == len(models):
            break
        if rnd < OVERLOAD_ROUNDS - 1:
            _notify(f"All Gemini models are busy; waiting {OVERLOAD_ROUND_WAIT}s and "
                    "trying again…")
            time.sleep(OVERLOAD_ROUND_WAIT)
    raise LLMError(f"{what} failed on every model ({', '.join(models)}). "
                   f"Last error: {errors[-1] if errors else 'unknown error'}")


# ---------------------------------------------------------------- public API
def ask(prompt: Prompt, *, temperature: float = 0.3, retries: int = 3) -> str:
    """One LLM call that returns non-empty text, or raises LLMError."""
    def make_call(model: str):
        llm = get_llm(temperature, model)

        def call():
            text = message_text(llm.invoke(prompt)).strip()
            if not text:
                raise LLMError("the model returned no text (it may have been blocked)")
            return text
        return call

    return _with_fallback(make_call, retries=retries, what="Gemini request")


def ask_many(prompts: Sequence[Prompt], *, temperature: float = 0.3,
             concurrency: Optional[int] = None) -> List[str]:
    """Run several prompts in parallel. Any that fail are retried one by one
    (which also waits out per-minute rate limits)."""
    if not prompts:
        return []
    model = model_chain()[0]
    llm = get_llm(temperature, model)
    raw = llm.batch(list(prompts),
                    config={"max_concurrency": concurrency or config.MAP_CONCURRENCY},
                    return_exceptions=True)
    results = []
    for prompt, r in zip(prompts, raw):
        if isinstance(r, Exception) and _is_key_error(r):
            raise LLMError(describe_error(r, model)) from r
        # Anything else (overload, quota, missing model) is retried by `ask`,
        # which switches models as needed.
        text = "" if isinstance(r, Exception) else message_text(r).strip()
        results.append(text or ask(prompt, temperature=temperature))
    return results


def ask_structured(schema: Type[T], prompt: Prompt, *, retries: int = 3) -> T:
    """LLM call whose answer is parsed into a Pydantic model."""
    def make_call(model: str):
        llm = get_llm(0.0, model).with_structured_output(schema)

        def call():
            result = llm.invoke(prompt)
            if not isinstance(result, schema):
                raise LLMError("the model did not return the expected structure")
            return result
        return call

    return _with_fallback(make_call, retries=retries, what="Gemini structured request")
