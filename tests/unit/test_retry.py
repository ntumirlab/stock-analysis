"""core.retry.call_with_retry 的單元測試。"""

import pytest

from core.retry import call_with_retry


class _Script:
    """依序丟出例外或回傳值的假函式，記錄呼叫次數。"""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def test_first_try_success_does_not_sleep():
    sleeps = []
    result, failures = call_with_retry(_Script("ok"), (10, 30, 60), sleep=sleeps.append)
    assert result == "ok"
    assert failures == []
    assert sleeps == []


def test_retries_with_given_delays_until_success():
    sleeps = []
    first, second = TimeoutError("t1"), ConnectionError("c2")
    fn = _Script(first, second, "ok")

    result, failures = call_with_retry(fn, (10, 30, 60), sleep=sleeps.append)

    assert result == "ok"
    assert failures == [first, second]
    assert sleeps == [10, 30]
    assert fn.calls == 3


def test_any_exception_type_is_retried():
    # 永豐端異常會丟哪種例外無法事先確定，故不分類一律重試
    fn = _Script(ValueError("bad"), RuntimeError("500"), "ok")
    result, failures = call_with_retry(fn, (0, 0, 0), sleep=lambda s: None)
    assert result == "ok"
    assert len(failures) == 2


def test_exhausted_retries_reraise_last_exception():
    sleeps = []
    last = TimeoutError("t4")
    fn = _Script(TimeoutError("t1"), TimeoutError("t2"), TimeoutError("t3"), last)

    with pytest.raises(TimeoutError) as excinfo:
        call_with_retry(fn, (10, 30, 60), sleep=sleeps.append)

    assert excinfo.value is last
    # 總嘗試 = len(delays) + 1；最後一次失敗後不再等待
    assert fn.calls == 4
    assert sleeps == [10, 30, 60]


def test_keyboard_interrupt_is_not_retried():
    fn = _Script(KeyboardInterrupt(), "ok")
    with pytest.raises(KeyboardInterrupt):
        call_with_retry(fn, (0, 0, 0), sleep=lambda s: None)
    assert fn.calls == 1
