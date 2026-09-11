"""
頁面資料快取與每日更新
======================
每個頁面在 import 時建立一個 ``PageCache``：啟動時計算一次，之後由
``start_daily_refresh`` 的執行緒每天固定時間（``REFRESH_AT``，台北時間）重算。
失敗只記 log、不重試（與 docker/crontab 的 job 一致；重試會撞上 21:55 起的回測時段），
隔天同一時間再算；需要立即重算時重啟服務即可（啟動時會算一次）。

為什麼是固定時間而不是偵測新資料：資料為日 K，且 FinLab 各資料集是陸續更新的
（價格先到、法人與分點稍晚），若一有新資料就重算，會出現「價格是今天、籌碼還是昨天」
的中間狀態，最新一日的名單會少一截。21:45 前顯示前一交易日、21:45 後顯示當日，語意最單純。

callback 讀取時請先 ``cache = page_cache.data`` 取得一份快照，之後全部從該快照讀，
避免背景執行緒在讀取途中換掉快取而混用新舊資料。

限制：gunicorn 必須維持 ``-w 1``（單一 process；執行緒數可用 ``-k gthread --threads N`` 調整）。
快取存在 process 記憶體，若開多個 process，每個 process 各有一份快取、各自重算，
其中一個失敗時使用者會輪流看到新舊兩種資料。
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from datetime import time as dtime

from alan_dashboard.theme import TZ

logger = logging.getLogger(__name__)

# 每日重算時間（台北）：FinLab 日資料（含分點）於 19:00 前更新完畢；
# 排在 docker/crontab 的回測（21:55 起）之前，避免兩個容器同時佔用記憶體
REFRESH_AT = dtime(21, 45)
# 顯示於各頁「訊號日」卡片，讓使用者知道何時會換成當日資料
REFRESH_LABEL = f'每日 {REFRESH_AT:%H:%M} 自動更新'

_REGISTRY: list['PageCache'] = []


class PageCache:
    """單一頁面的記憶體快取。

    Args:
        name: 頁面名稱（僅用於 log）
        load_inputs: 無參數函式，回傳 build 所需的 inputs（內含 FinLab data.get）
        build: 接收 inputs、回傳快取 dict 的函式；本類別會補上 ``'updated'``
    """

    def __init__(self, name, load_inputs, build):
        self.name = name
        self._load_inputs = load_inputs
        self._build = build
        self._lock = threading.Lock()
        self.data: dict | None = None
        _REGISTRY.append(self)

    def refresh(self) -> None:
        """重新拉取 FinLab 資料並重算；以鎖序列化，同時觸發時逐一執行。"""
        with self._lock:
            t0 = time.monotonic()
            built = self._build(self._load_inputs())
            built['updated'] = datetime.now(TZ).strftime('%Y-%m-%d %H:%M')
            self.data = built  # 換掉整份引用，讀取端已取得的快照不受影響
            logger.info('[%s] cache rebuilt in %.1fs', self.name, time.monotonic() - t0)


def _seconds_until(at: dtime) -> float:
    now = datetime.now(TZ)
    target = now.replace(hour=at.hour, minute=at.minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


def _refresh_loop(at: dtime) -> None:
    while True:
        time.sleep(_seconds_until(at))
        for cache in _REGISTRY:
            try:
                cache.refresh()
            except Exception:
                logger.exception('[%s] daily refresh failed; keeping previous cache until next run',
                                 cache.name)


def start_daily_refresh(at: dtime = REFRESH_AT) -> None:
    """啟動背景執行緒，每天 ``at``（台北時間）對所有已註冊的快取執行 ``refresh()``。"""
    thread = threading.Thread(target=_refresh_loop, args=(at,),
                              name='dashboard-cache-refresh', daemon=True)
    thread.start()
