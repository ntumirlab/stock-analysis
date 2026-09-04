"""GoldenAI 下單 adapter 的週期日期運算與清單新鮮度檢查（純邏輯，CI 可測）。

從 strategy_class/golden_ai_order_adapter.py 抽出。這裡管的是實盤下單的
買賣日排程，改動前後必須通過 tests/unit/test_trading_cycles.py 的全部案例。
weekday 皆為 pandas dayofweek 慣例（週一=0）。
"""

import logging
from typing import List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

Cycle = Tuple[pd.Timestamp, pd.Timestamp]


def compute_cycles(anchor, buy_weekday: int, sell_weekday: int,
                   hold_weeks: int, until) -> List[Cycle]:
    """從錨點鋪出各週期的 (買入日, 賣出日)，直到 until（含）之後的第一個買入日為止。

    賣出日 = 買入日之後（含當天）的第 hold_weeks 個賣出 weekday；
    下一個買入日 = 賣出日之後的第一個買入 weekday。日期落在假日時
    由 finlab 訊號對齊處理，行程表本身不位移。
    """
    anchor = pd.Timestamp(anchor).normalize()
    entry = anchor + pd.Timedelta(days=(buy_weekday - anchor.dayofweek) % 7)
    sell_offset = (sell_weekday - buy_weekday) % 7
    cycles = []
    while entry <= until:
        exit_d = entry + pd.Timedelta(days=sell_offset + 7 * (hold_weeks - 1))
        cycles.append((entry, exit_d))
        step = (buy_weekday - exit_d.dayofweek) % 7
        entry = exit_d + pd.Timedelta(days=step if step else 7)
    return cycles


def compute_historical_cycles(index: pd.DatetimeIndex, buy_weekday: int,
                              sell_weekday: int, hold_weeks: int,
                              before: pd.Timestamp) -> List[Cycle]:
    """錨點之前的歷史週期（資料驅動、僅完整週期，且整個週期須在 before 之前結束）。

    僅供報告延續性：finlab Report 會把全為 1.0 的 creturn 截斷成空序列，
    導致 Portfolio 建構時 iloc[0] 崩潰，所以報告視窗內必須有可定價的既往交易。
    今日目標持股永遠由錨點行程表（compute_cycles）決定，歷史週期不影響。
    """
    dow = index.dayofweek
    all_buy_days = index[dow == buy_weekday]
    all_sell_days = index[dow == sell_weekday]
    cycles = []
    current_entry = all_buy_days[0] if len(all_buy_days) > 0 else None
    while current_entry is not None and current_entry < before:
        sell_days_after = all_sell_days[all_sell_days >= current_entry]
        if len(sell_days_after) < hold_weeks:
            break
        current_exit = sell_days_after[hold_weeks - 1]
        if current_exit >= before:
            break
        cycles.append((current_entry, current_exit))
        next_buy_days = all_buy_days[all_buy_days > current_exit]
        current_entry = next_buy_days[0] if len(next_buy_days) > 0 else None
    return cycles


def build_tranche_specs(cycle_start_date, num_tranches: int,
                        invest_ratio: float = 1.0) -> List[Tuple[str, pd.Timestamp, float]]:
    """滾動 tranche 規格：[(名稱, 錨點, 權重)]。

    第 k 個 tranche 的錨點 = cycle_start_date + 7k 天，權重 = invest_ratio / num_tranches。
    週頻進場 + 持有 num_tranches 週時，各 tranche 週期恰好無縫涵蓋每個買入 weekday。
    名稱是 PortfolioSyncManager state 的 key，實倉啟用後不可再改。
    """
    start = pd.Timestamp(cycle_start_date).normalize()
    weight = invest_ratio / num_tranches
    return [(f"tranche_{k + 1}", start + pd.Timedelta(days=7 * k), weight)
            for k in range(num_tranches)]


def find_current_cycle(cycles: List[Cycle], today: pd.Timestamp) -> Optional[Cycle]:
    """回傳 today 所在的 (買入日, 賣出日)，不在任何週期內則回傳 None。"""
    return next(((e, x) for e, x in cycles if e <= today <= x), None)


def missing_list_sunday(cycles: List[Cycle], today: pd.Timestamp,
                        list_sundays) -> Optional[pd.Timestamp]:
    """當期進場該用的那個週日根本沒有清單時回傳它，否則 None。

    `check_recommendation_freshness` 看不到這種缺席：它比的是「最新清單」對「當期
    該用的週日」，而週中才發布的清單會對齊到**下一個**週日、讓判斷式通過。實際資料
    裡四次缺清單（weekly 2025-12-14、2026-01-11，monthly 2025-10-12、2026-01-11）
    全是這一種——清單在進場日當天（週一）才發，那一輪於是空手，而且一聲不響。

    只在缺席的那個週日所屬的整週內回報。那一週是清單還補得進來的時間窗（日期記成
    該週日入庫，每日 sync 就會補進場）；過了就確定空手，4 週策略等於 25% 資金空一輪，
    每天再喊也改變不了。

    不拋例外：拋出去會擋掉整支 job、連其他 tranche 的賣出都做不成，而這種情況本來
    就沒東西可買，擋下來沒有意義。`check_recommendation_freshness` 已經按同一個理由
    改成不拋（見那支的說明），兩邊現在一致＝缺清單只讓當期空手，其餘 tranche 照常。
    `list_sundays` 是對齊後的清單週日（見 `align_to_sunday`），與 `_create_df` 的
    `weekly_batches` 同一套鍵。
    """
    current = find_current_cycle(cycles, today)
    if current is None:
        return None
    expected_sunday = owning_sunday([current[0]])[0]
    if expected_sunday in {pd.Timestamp(s).normalize() for s in list_sundays}:
        return None
    if owning_sunday([today])[0] != expected_sunday:
        return None
    return expected_sunday


def align_to_sunday(date: pd.Timestamp) -> pd.Timestamp:
    """週日留在當天；週一～週六對齊到下一個週日（與推薦清單批次對齊規則一致）。"""
    return date + pd.Timedelta(days=6 - date.weekday())


def owning_sunday(dates) -> pd.DatetimeIndex:
    """每一天用的是哪一份清單＝往前最近的週日（週日算自己）。`align_to_sunday` 的反向。

    `_create_df` 把清單日的內容 resample('D').ffill() 往後鋪，所以「這天的 position
    是哪個週日的清單」就是這個函式。用途是找出「那個週日根本沒有清單」的日子——
    那幾天鋪過來的是更早的清單，進場等於買一份從未發布過的名單。
    """
    idx = pd.DatetimeIndex(dates)
    return idx - pd.to_timedelta((idx.dayofweek + 1) % 7, unit='D')


def check_recommendation_freshness(cycles: List[Cycle], today: pd.Timestamp,
                                   latest_rec_date: Optional[str]) -> None:
    """當期缺少進場日該用的週日清單就記警告；DB 一份清單都沒有才拋 RuntimeError。

    **過期清單不再擋下單。**這支原本的工作是「不要默默拿過期清單買」，而
    `GoldenAITWStrategyBase._create_df` 現在會把沒有清單的那一週整週歸零
    （見那裡的註解），過期清單買進在結構上已經不可能發生——缺清單的那一份 tranche
    自己空手，其餘照常。

    留著拋就只剩壞處：`OrderExecutor.run_strategy_and_sync` 對每支策略沒有
    try/except，任何一份 tranche 拋出去就是整支 job 中止，那天所有 tranche 的
    **賣出**也一起做不成。實測缺 2026-09-06 清單時，tranche_2 會從週一一路拋到週六，
    其中週五正是 tranche_3 的賣出日。而且不能改成「catch 起來跳過那一份」——
    finlab 的 `_prune_removed_strategies` 會把不在 Portfolio 裡的策略整個 pop 掉，
    等於把那份 tranche 的持股全部賣出，比中止更糟。

    `latest_rec_date is None` 仍然拋：一份清單都沒有時 `_create_df` 連 position
    都建不起來（空 records 會在 pivot 前 KeyError），與其讓它爆在下游不如講清楚。
    這種情況也不可能有持股，擋下來沒有賣單的代價。

    today 不在任何週期內（錨點前、週期交界的週末）則不檢查。
    """
    current = find_current_cycle(cycles, today)
    if current is None:
        return
    entry = current[0]
    expected_sunday = owning_sunday([entry])[0]
    if latest_rec_date is None:
        raise RuntimeError(
            f"DB 無推薦清單：當前週期（買入日 {entry:%Y-%m-%d}）"
            f"需要 {expected_sunday:%Y-%m-%d} 的清單，不下單"
        )
    aligned = align_to_sunday(pd.to_datetime(latest_rec_date))
    if aligned < expected_sunday:
        logger.warning(
            f"推薦清單過期：最新清單日期 {latest_rec_date}（對齊週日 {aligned:%Y-%m-%d}），"
            f"當前週期買入日 {entry:%Y-%m-%d} 應使用 {expected_sunday:%Y-%m-%d} 的清單。"
            f"本輪空手，其餘 tranche 照常"
        )
