"""暫時性失敗的重試（純邏輯，無外部依賴）。

供 utils/authentication.py 的 shioaji 登入使用：永豐伺服器偶發逾時
（如 2026-10-01 08:10 的 `api/v1/auth/error_tracking` TimeoutError）會讓
一天唯一一次的下單整批失敗，而這類錯誤通常數秒到數分鐘內就恢復。
"""

import logging
import time
from typing import Callable, List, Optional, Sequence, Tuple, TypeVar

T = TypeVar("T")


def call_with_retry(
    fn: Callable[[], T],
    delays: Sequence[float],
    *,
    label: str = "operation",
    logger: Optional[logging.Logger] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Tuple[T, List[Exception]]:
    """呼叫 fn()，失敗時依 delays 依序等待後重試。

    總嘗試次數為 len(delays) + 1。任何 Exception 都會重試（不分類），
    KeyboardInterrupt 等 BaseException 不攔截。

    Returns:
        (fn 的回傳值, 成功前每次失敗的例外)。後者為空表示第一次就成功。

    Raises:
        重試用盡時，原樣拋出最後一次的例外（保留其 traceback）。
    """
    logger = logger or logging.getLogger(__name__)
    failures: List[Exception] = []
    total = len(delays) + 1

    for attempt in range(1, total + 1):
        try:
            return fn(), failures
        except Exception as exc:
            failures.append(exc)
            if attempt == total:
                logger.error(f"{label} 第 {attempt}/{total} 次失敗，放棄: {exc!r}")
                raise
            delay = delays[attempt - 1]
            logger.warning(
                f"{label} 第 {attempt}/{total} 次失敗，{delay} 秒後重試: {exc!r}"
            )
            sleep(delay)

    raise AssertionError("unreachable")
