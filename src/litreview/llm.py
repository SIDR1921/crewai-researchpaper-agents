"""LLM access for the whole project.

Every call goes through CrewAI's :class:`crewai.LLM`, which is a thin wrapper over
LiteLLM, so one model string works for Groq, Gemini and a local Ollama.

Two things are centralised here:

* **Usage logging.** A LiteLLM success/failure callback is installed once, so
  *every* completion -- whether issued by a CrewAI agent or by our own
  :func:`complete` helper -- appends a row to ``results/llm_calls.jsonl`` with
  prompt/completion tokens, wall-clock latency and cost. Nothing has to opt in.
* **Rate limiting.** Free tiers are small (Groq is ~30 requests/min). A token
  bucket plus exponential backoff on 429/5xx keeps us inside them.
"""
from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

from .config import SETTINGS, Settings, paths

_LOG_LOCK = threading.Lock()
_INSTALLED = False
_LOG_PATH: Path | None = None
_TAG_STACK: list[str] = []


def _current_tag() -> str:
    return _TAG_STACK[-1] if _TAG_STACK else "untagged"


# ------------------------------------------------------------------ usage log

def _usage_of(response: Any) -> tuple[int, int]:
    if response is None:
        return 0, 0
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None and (isinstance(response, dict) or hasattr(response, "prompt_tokens")):
        usage = response
    if usage is None:
        return 0, 0
    get = usage.get if isinstance(usage, dict) else lambda k, d=0: getattr(usage, k, d)
    return int(get("prompt_tokens", 0) or 0), int(get("completion_tokens", 0) or 0)


def _cost_of(response: Any, model: str, n_in: int, n_out: int) -> float:
    try:
        import litellm
        c = litellm.completion_cost(completion_response=response)
        if c is not None:
            return float(c)
    except Exception:
        pass
    from .config import PROVIDERS
    bare = model.split("/")[-1]
    for name, preset in PROVIDERS.items():
        known = {preset["model"], preset["small_model"]}
        if model.startswith(name + "/") or bare in {k.split("/")[-1] for k in known}:
            return (n_in * preset["price_in"] + n_out * preset["price_out"]) / 1e6
    return 0.0


def _append(row: dict) -> None:
    if _LOG_PATH is None:
        return
    with _LOG_LOCK:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_PATH.open("a") as f:
            f.write(json.dumps(row) + "\n")


def _on_success(kwargs, response, start_time, end_time):  # pragma: no cover - callback
    """LiteLLM callback, kept for any direct ``litellm.completion`` use."""
    model = kwargs.get("model", "?")
    n_in, n_out = _usage_of(response)
    _append({
        "ts": time.time(),
        "tag": _current_tag(),
        "model": model,
        "source": "litellm",
        "ok": True,
        "prompt_tokens": n_in,
        "completion_tokens": n_out,
        "total_tokens": n_in + n_out,
        "latency_s": round((end_time - start_time).total_seconds()
                           if hasattr(end_time, "timestamp") else float(end_time - start_time), 3),
        "cost_usd": round(_cost_of(response, model, n_in, n_out), 8),
    })


def _on_failure(kwargs, response, start_time, end_time):  # pragma: no cover - callback
    _append({
        "ts": time.time(),
        "tag": _current_tag(),
        "model": kwargs.get("model", "?"),
        "source": "litellm",
        "ok": False,
        "error": str(response)[:400],
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        "latency_s": None, "cost_usd": 0.0,
    })


#: call_id -> {"t0", "tag"}. The tag is captured when the call *starts*, because
#: the bus delivers events on a worker thread: by the time the completion event
#: arrives the caller may already have left its ``tagged`` block, and the row
#: would be filed under the wrong stage.
_CALL_STARTS: dict[str, dict] = {}


def _install_crewai_listener() -> bool:
    """Log from CrewAI's event bus.

    CrewAI 1.x reaches several providers through native clients that never call
    ``litellm.completion``, so a LiteLLM callback alone silently records nothing.
    Its event bus fires for every path, which is why the usage log is built from
    it rather than from LiteLLM. ``litellm`` callbacks stay installed only for the
    rare direct call, and rows carry a ``source`` field so the two can be told
    apart if they ever both fire.
    """
    try:
        from crewai.events import crewai_event_bus
        from crewai.events.types.llm_events import (LLMCallCompletedEvent, LLMCallFailedEvent,
                                                    LLMCallStartedEvent)
    except Exception:  # pragma: no cover - crewai optional at import time
        return False

    @crewai_event_bus.on(LLMCallStartedEvent)
    def _started(source, event):  # noqa: ANN001
        cid = getattr(event, "call_id", None)
        if cid:
            _CALL_STARTS[str(cid)] = {"t0": time.time(), "tag": _current_tag()}

    @crewai_event_bus.on(LLMCallCompletedEvent)
    def _completed(source, event):  # noqa: ANN001
        cid = str(getattr(event, "call_id", "") or "")
        started = _CALL_STARTS.pop(cid, None) or {}
        t0 = started.get("t0")
        model = str(getattr(event, "model", "?") or "?")
        n_in, n_out = _usage_of(event)
        if (n_in, n_out) == (0, 0):
            n_in, n_out = _usage_of(getattr(event, "usage", None))
        if (n_in, n_out) == (0, 0):
            n_in, n_out = _estimate_tokens(getattr(event, "messages", None),
                                           getattr(event, "response", None), model)
        _append({
            "ts": time.time(),
            "tag": started.get("tag") or _current_tag(),
            "model": model,
            "source": "crewai",
            "call_type": str(getattr(event, "call_type", "") or ""),
            "ok": True,
            "prompt_tokens": n_in,
            "completion_tokens": n_out,
            "total_tokens": n_in + n_out,
            "latency_s": round(time.time() - t0, 3) if t0 else None,
            "cost_usd": round(_cost_of(None, model, n_in, n_out), 8),
        })

    @crewai_event_bus.on(LLMCallFailedEvent)
    def _failed(source, event):  # noqa: ANN001
        cid = str(getattr(event, "call_id", "") or "")
        started = _CALL_STARTS.pop(cid, None) or {}
        t0 = started.get("t0")
        _append({
            "ts": time.time(),
            "tag": started.get("tag") or _current_tag(),
            "model": str(getattr(event, "model", "?") or "?"),
            "source": "crewai",
            "ok": False,
            "error": str(getattr(event, "error", ""))[:400],
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "latency_s": round(time.time() - t0, 3) if t0 else None,
            "cost_usd": 0.0,
        })

    return True


def _estimate_tokens(messages, response, model: str) -> tuple[int, int]:
    """Fallback when a provider reports no usage: count with LiteLLM's tokenizer.

    Estimated rows are still honest about being tokens, and the alternative --
    recording zero -- would understate the cost of exactly the providers that are
    hardest to budget for.
    """
    try:
        import litellm
        n_in = litellm.token_counter(model=model, messages=messages or [])
        n_out = litellm.token_counter(model=model, text=str(response or ""))
        return int(n_in), int(n_out)
    except Exception:
        text_in = " ".join(str(m.get("content", "")) for m in (messages or [])
                           if isinstance(m, dict))
        return len(text_in) // 4, len(str(response or "")) // 4


def install_call_logger(path: str | Path | None = None) -> Path:
    """Install the usage logger once. Returns the log path.

    The log is built from CrewAI's event bus rather than from a LiteLLM callback,
    because CrewAI 1.x reaches several providers through native clients that never
    touch LiteLLM; a callback-only logger records nothing for those and the cost
    table in the report would quietly read zero.
    """
    global _INSTALLED, _LOG_PATH
    _LOG_PATH = Path(path) if path else paths().results / "llm_calls.jsonl"
    if not _INSTALLED:
        try:
            import litellm
            litellm.drop_params = True        # providers differ on supported params
            litellm.suppress_debug_info = True
        except ImportError:
            pass
        _install_crewai_listener()
        _INSTALLED = True
    return _LOG_PATH


@contextmanager
def tagged(tag: str):
    """Label every call made inside the block, so usage can be split by stage."""
    _TAG_STACK.append(tag)
    try:
        yield
    finally:
        _TAG_STACK.pop()


def usage_summary(path: str | Path | None = None) -> dict:
    """Aggregate ``llm_calls.jsonl`` into totals overall, per tag and per model."""
    p = Path(path) if path else (_LOG_PATH or paths().results / "llm_calls.jsonl")
    rows = [json.loads(l) for l in p.open()] if p.exists() else []
    def agg(rs: list[dict]) -> dict:
        lat = [r["latency_s"] for r in rs if r.get("latency_s") is not None]
        return {
            "calls": len(rs),
            "failed": sum(1 for r in rs if not r.get("ok")),
            "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in rs),
            "completion_tokens": sum(r.get("completion_tokens", 0) for r in rs),
            "total_tokens": sum(r.get("total_tokens", 0) for r in rs),
            "cost_usd": round(sum(r.get("cost_usd", 0.0) for r in rs), 6),
            "latency_s_mean": round(sum(lat) / len(lat), 3) if lat else None,
            "latency_s_total": round(sum(lat), 1) if lat else None,
        }
    keys = lambda f: sorted({r.get(f, "?") for r in rows})
    return {
        "overall": agg(rows),
        "by_tag": {k: agg([r for r in rows if r.get("tag") == k]) for k in keys("tag")},
        "by_model": {k: agg([r for r in rows if r.get("model") == k]) for k in keys("model")},
    }


# --------------------------------------------------------------- rate limiting

class RateLimiter:
    """Token bucket on requests/min and tokens/min, shared across threads.

    Reservations are made up front from an estimate, which means a batch whose
    estimate is a large fraction of the per-minute budget packs badly: two
    reservations of 3,900 do not fit inside 7,400, so only one call runs a minute
    even though the real usage would have allowed nearly two. Prefer several small
    batches over one large one; :func:`litreview.relations.label.label_contexts`
    is tuned on that basis.
    """

    def __init__(self, rpm: int, tpm: int) -> None:
        self.rpm, self.tpm = rpm, tpm
        self._reqs: deque[float] = deque()
        self._toks: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()

    def acquire(self, est_tokens: int = 800) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._reqs and now - self._reqs[0] > 60:
                    self._reqs.popleft()
                while self._toks and now - self._toks[0][0] > 60:
                    self._toks.popleft()
                tok_sum = sum(t for _, t in self._toks)
                if len(self._reqs) < self.rpm and tok_sum + est_tokens <= self.tpm:
                    self._reqs.append(now)
                    self._toks.append((now, est_tokens))
                    return
                waits = [60 - (now - self._reqs[0]) if self._reqs else 0.0,
                         60 - (now - self._toks[0][0]) if self._toks else 0.0]
            time.sleep(max(0.25, min(w for w in waits if w > 0) if any(w > 0 for w in waits) else 0.25))


_LIMITERS: dict[str, RateLimiter] = {}


def limiter_for(settings: Settings = SETTINGS) -> RateLimiter:
    if settings.provider not in _LIMITERS:
        _LIMITERS[settings.provider] = RateLimiter(settings.preset["rpm"], settings.preset["tpm"])
    return _LIMITERS[settings.provider]


# ------------------------------------------------------------------ LLM access

_LLM_CACHE: dict[tuple, Any] = {}


def get_llm(size: str = "large", settings: Settings = SETTINGS, **kw):
    """A configured :class:`crewai.LLM`. ``size`` is ``large`` or ``small``."""
    from crewai import LLM

    install_call_logger()
    model = settings.model if size == "large" else settings.small_model
    key = (model, settings.temperature, settings.max_tokens, tuple(sorted(kw.items())))
    if key in _LLM_CACHE:
        return _LLM_CACHE[key]

    opts: dict[str, Any] = {
        "model": model,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        **kw,
    }
    if settings.provider == "ollama":
        opts["base_url"] = settings.ollama_base_url
    elif settings.api_key:
        opts["api_key"] = settings.api_key
        os.environ.setdefault(settings.preset["api_key_env"], settings.api_key)
    _LLM_CACHE[key] = LLM(**opts)
    return _LLM_CACHE[key]


RETRYABLE = re.compile(r"429|rate.?limit|overload|timeout|timed out|50[0-9]|temporarily", re.I)


def complete(
    prompt: str,
    system: str | None = None,
    size: str = "large",
    settings: Settings = SETTINGS,
    attempts: int = 5,
    est_tokens: int = 1200,
    **kw,
) -> str:
    """One chat completion with rate limiting and exponential backoff."""
    llm = get_llm(size=size, settings=settings, **kw)
    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
    last: Exception | None = None
    for i in range(attempts):
        limiter_for(settings).acquire(est_tokens)
        try:
            return llm.call(messages)
        except Exception as e:  # noqa: BLE001
            last = e
            if i == attempts - 1 or not RETRYABLE.search(str(e)):
                raise
            time.sleep(min(60, 2 ** i * 2) + random.uniform(0, 1.5))
    raise RuntimeError(f"LLM call failed after {attempts} attempts") from last


JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_json(text: str) -> Any:
    """Best-effort JSON out of an LLM reply (fenced block, or first {...}/[...])."""
    if m := JSON_BLOCK.search(text):
        text = m.group(1)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        s, e = text.find(open_c), text.rfind(close_c)
        if s != -1 and e > s:
            try:
                return json.loads(text[s:e + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON found in reply: {text[:300]!r}")


def complete_json(prompt: str, system: str | None = None, attempts: int = 3, **kw) -> Any:
    """Ask for JSON and parse it, re-prompting on malformed output."""
    err = ""
    for _ in range(attempts):
        text = complete(prompt + err, system=system, **kw)
        try:
            return parse_json(text)
        except ValueError as e:
            err = f"\n\nYour previous reply was not valid JSON ({e}). Reply with JSON only."
    raise ValueError("model never produced valid JSON")


def probe(settings: Settings = SETTINGS) -> dict:
    """Smoke-test the configured provider. Used by notebooks' config cell."""
    t0 = time.time()
    with tagged("probe"):
        try:
            reply = complete("Reply with the single word: ready", settings=settings, max_tokens=16)
            return {"ok": True, "provider": settings.provider, "model": settings.model,
                    "reply": reply.strip()[:40], "latency_s": round(time.time() - t0, 2)}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "provider": settings.provider, "model": settings.model,
                    "error": str(e)[:300]}
