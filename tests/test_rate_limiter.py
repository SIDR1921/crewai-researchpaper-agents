"""The rate limiter, including the output-token budget providers meter separately.

Every test drives a fake clock that ``sleep`` advances, so a limiter that decides
to wait is observable without the test actually waiting.
"""
import pytest

from litreview.llm import RateLimiter


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    slept = []

    def _sleep(s):
        slept.append(s)
        now[0] += max(s, 0.01)
        if len(slept) > 500:                     # a limiter that never releases
            raise AssertionError("rate limiter did not release")

    monkeypatch.setattr("litreview.llm.time.monotonic", lambda: now[0])
    monkeypatch.setattr("litreview.llm.time.sleep", _sleep)
    return {"now": now, "slept": slept}


def test_three_calls_fit_when_nothing_is_constrained(clock):
    rl = RateLimiter(rpm=10, tpm=10_000, otpm=10_000)
    for _ in range(3):
        rl.acquire(100, 10)
    assert clock["slept"] == [], "nothing should have waited"


def test_requests_per_minute_is_enforced(clock):
    rl = RateLimiter(rpm=2, tpm=10_000, otpm=10_000)
    rl.acquire(100, 10)
    rl.acquire(100, 10)
    assert clock["slept"] == []
    rl.acquire(100, 10)                          # third exceeds rpm
    assert clock["slept"], "a third request inside the minute must wait"
    assert clock["now"][0] >= 1060, "it should wait out the window"


def test_total_token_budget_blocks(clock):
    rl = RateLimiter(rpm=100, tpm=1000, otpm=10_000)
    rl.acquire(900, 10)
    rl.acquire(900, 10)                          # 1800 > 1000
    assert clock["slept"], "the total-token budget must be enforced"


def test_output_token_budget_blocks_even_when_totals_fit(clock):
    """Groq allows 8,000 total but only 1,000 completion tokens a minute.

    Tracking totals alone let a second call straight through and earned a 429.
    """
    rl = RateLimiter(rpm=100, tpm=100_000, otpm=500)
    rl.acquire(2000, 400)
    rl.acquire(2000, 400)                        # totals fine; 800 > 500 output
    assert clock["slept"], "output budget must be enforced independently"


def test_window_slides_so_old_usage_is_forgiven(clock):
    rl = RateLimiter(rpm=1, tpm=100, otpm=100)
    rl.acquire(100, 100)
    clock["now"][0] += 61                        # the whole window has passed
    rl.acquire(100, 100)
    assert clock["slept"] == [], "usage older than a minute must not count"


def test_default_output_estimate_is_derived_from_total(clock):
    rl = RateLimiter(rpm=100, tpm=100_000, otpm=100_000)
    rl.acquire(800)                              # no explicit output estimate
    assert rl._out and rl._out[0][1] == 200      # a quarter of the total
