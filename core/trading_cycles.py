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


def find_current_cycle(cycles: List[Cycle], today: pd.Timestamp) -> Optional[Cycle]:
    """回傳 today 所在的 (買入日, 賣出日)，不在任何週期內則回傳 None。"""
    return next(((e, x) for e, x in cycles if e <= today <= x), None)


def align_to_sunday(date: pd.Timestamp) -> pd.Timestamp:
    """週日留在當天；週一～週六對齊到下一個週日（與推薦清單批次對齊規則一致）。"""
    return date + pd.Timedelta(days=6 - date.weekday())


def check_recommendation_freshness(cycles: List[Cycle], today: pd.Timestamp,
                                   latest_rec_date: Optional[str]) -> None:
    """進行中的週期若缺少進場日應使用的週日清單，拋 RuntimeError 擋下下單。

    清單晚入庫時每日 sync 會自動補進場，寧可晚買也不要默默用過期清單買。
    today 不在任何週期內（錨點前、週期交界的週末）則不檢查。
    """
    current = find_current_cycle(cycles, today)
    if current is None:
        return
    entry = current[0]
    expected_sunday = entry - pd.Timedelta(days=(entry.dayofweek - 6) % 7)
    if latest_rec_date is None:
        raise RuntimeError(
            f"DB 無推薦清單：當前週期（買入日 {entry:%Y-%m-%d}）"
            f"需要 {expected_sunday:%Y-%m-%d} 的清單，不下單"
        )
    rec_date = pd.to_datetime(latest_rec_date)
    aligned = align_to_sunday(rec_date)
    if aligned < expected_sunday:
        raise RuntimeError(
            f"推薦清單過期：最新清單日期 {latest_rec_date}（對齊週日 {aligned:%Y-%m-%d}），"
            f"當前週期買入日 {entry:%Y-%m-%d} 應使用 {expected_sunday:%Y-%m-%d} 的清單，不下單"
        )
