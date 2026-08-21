"""core/backtest_window 的單元測試：回測視窗起點對齊到空手交易日。

固定情境重現週策略 `final_position` 的形狀：週一進場、週五出場，經過 `.shift(-1)`
之後持倉落在「週日 ~ 週三」，週四到週六空手。2026-05-04 / 05-11 / 05-18 / 05-25
都是週一。
"""

import pandas as pd
import pytest

from core.backtest_window import flat_trading_days, snap_cutoff_to_flat_trading_day

HELD_WEEKS = ["2026-05-03", "2026-05-10", "2026-05-17", "2026-05-24"]  # 各週的週日


def _position(held_sundays=HELD_WEEKS, start="2026-05-01", end="2026-06-06"):
    """持倉 = 每個週日起連續四天（週日~週三），其餘空手。"""
    idx = pd.date_range(start, end, freq="D")
    held = set()
    for sun in held_sundays:
        held.update(pd.date_range(sun, periods=4, freq="D"))
    return pd.DataFrame({"2330": [d in held for d in idx]}, index=idx)


def _trading_days(position, holidays=()):
    """週一~週五扣掉指定休市日。"""
    idx = position.index[position.index.dayofweek < 5]
    return idx.difference(pd.DatetimeIndex(list(holidays)))


# 切點落在持倉中間、週末，都要退到前一個空手交易日 2026-05-08（週五）。
# 週四 05-14 與週五 05-15 本身就空手且是交易日，不該被動到。
@pytest.mark.parametrize("cutoff,expected", [
    ("2026-05-08", "2026-05-08"),  # 週五，空手交易日 → 不動
    ("2026-05-09", "2026-05-08"),  # 週六，空手但非交易日
    ("2026-05-10", "2026-05-08"),  # 週日，持倉（shift 後帶著週一的訊號）
    ("2026-05-11", "2026-05-08"),  # 週一，持倉中
    ("2026-05-12", "2026-05-08"),  # 週二，持倉中
    ("2026-05-13", "2026-05-08"),  # 週三，持倉中
    ("2026-05-14", "2026-05-14"),  # 週四，空手交易日 → 不動
    ("2026-05-15", "2026-05-15"),  # 週五，空手交易日 → 不動
])
def test_snap_retreats_to_flat_trading_day(cutoff, expected):
    pos = _position()
    got = snap_cutoff_to_flat_trading_day(pos, cutoff, _trading_days(pos))
    assert got == pd.Timestamp(expected)


@pytest.mark.parametrize("cutoff", pd.date_range("2026-05-08", "2026-05-30", freq="D"))
def test_snapped_cutoff_is_always_flat_and_tradable(cutoff):
    """不變式：不論切點落在星期幾，結果一定是空手的交易日。"""
    pos = _position()
    tds = _trading_days(pos)
    got = snap_cutoff_to_flat_trading_day(pos, cutoff, tds)
    assert got in tds
    assert not pos.loc[got].any()
    assert got <= cutoff


def test_holiday_friday_retreats_further_to_thursday():
    """週五休市時不能停在週五 —— 它不是交易日，會讓首筆延後一天進場。"""
    pos = _position()
    tds = _trading_days(pos, holidays=["2026-05-08"])
    got = snap_cutoff_to_flat_trading_day(pos, "2026-05-13", tds)
    assert got == pd.Timestamp("2026-05-07")  # 週四


def test_without_trading_days_falls_back_to_calendar_days():
    """沒有交易日曆時只能判空手，會停在週六 —— 這正是必須傳入交易日曆的理由。"""
    pos = _position()
    got = snap_cutoff_to_flat_trading_day(pos, "2026-05-13", trading_days=None)
    assert got == pd.Timestamp("2026-05-09")  # 週六


def test_no_flat_trading_day_before_cutoff_keeps_whole_frame():
    """切點之前整段都在持倉 → 回傳起點，不裁切（寧可視窗長也不留假交易）。"""
    idx = pd.date_range("2026-05-04", "2026-05-20", freq="D")
    pos = pd.DataFrame({"2330": [True] * len(idx)}, index=idx)
    got = snap_cutoff_to_flat_trading_day(pos, "2026-05-13", idx[idx.dayofweek < 5])
    assert got == pd.Timestamp("2026-05-04")


def test_empty_position_returns_cutoff_unchanged():
    empty = pd.DataFrame(index=pd.DatetimeIndex([]))
    assert snap_cutoff_to_flat_trading_day(empty, "2026-05-13", None) == pd.Timestamp("2026-05-13")


def test_flat_trading_days_excludes_weekends_and_held_days():
    pos = _position()
    got = flat_trading_days(pos, _trading_days(pos))
    assert pd.Timestamp("2026-05-07") in got   # 週四，空手
    assert pd.Timestamp("2026-05-08") in got   # 週五，空手
    assert pd.Timestamp("2026-05-09") not in got  # 週六
    assert pd.Timestamp("2026-05-11") not in got  # 週一，持倉中
    assert all(d.dayofweek < 5 for d in got)


def test_slicing_at_snapped_cutoff_keeps_the_signal_row():
    """回歸重點：裁切後 frame 第一列必須是空手的交易日，下一個交易日才是持倉起點。

    這一列就是 finlab 用來判「訊號在前一個交易日、隔週一進場」的依據；
    少了它首筆會延後一天進場（實測 period 從 4 掉到 3）。
    """
    pos = _position()
    tds = _trading_days(pos)
    cutoff = snap_cutoff_to_flat_trading_day(pos, "2026-05-13", tds)
    sliced = pos[pos.index >= cutoff]
    assert not sliced.iloc[0].any()
    assert sliced.index[0] == pd.Timestamp("2026-05-08")
    first_held = sliced.index[sliced.any(axis=1).to_numpy()][0]
    assert first_held == pd.Timestamp("2026-05-10")  # 週日列，對應週一進場
