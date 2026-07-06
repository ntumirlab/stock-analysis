"""core/price_frames 的單元測試：開收盤混合價與資料集 index 偏斜的對齊。"""

import pandas as pd

from core.price_frames import mix_open_close


def _frames(open_dates, close_dates):
    open_df = pd.DataFrame(
        {"2330": [100.0 + i for i in range(len(open_dates))]},
        index=pd.DatetimeIndex(open_dates),
    )
    close_df = pd.DataFrame(
        {"2330": [200.0 + i for i in range(len(close_dates))]},
        index=pd.DatetimeIndex(close_dates),
    )
    return open_df, close_df


def test_monday_uses_open_others_use_close():
    # 2026-07-06 是週一、07-07 是週二
    dates = ["2026-07-06", "2026-07-07"]
    open_df, close_df = _frames(dates, dates)
    mixed = mix_open_close(open_df, close_df, buy_weekday=0)
    assert mixed.loc["2026-07-06", "2330"] == 100.0  # 週一取開盤
    assert mixed.loc["2026-07-07", "2330"] == 201.0  # 週二取收盤


def test_close_has_extra_day_than_open():
    # 回歸案例（2026-07-06 盤後實測炸過）：收盤後 adj_close 先更新到今天、
    # adj_open 還停在前一交易日 → boolean mask 長度不一致 IndexError。
    # 修正後取交集，行為等同資料尚未更新的早晨。
    open_df, close_df = _frames(
        ["2026-07-02", "2026-07-03"],
        ["2026-07-02", "2026-07-03", "2026-07-06"],
    )
    mixed = mix_open_close(open_df, close_df, buy_weekday=0)
    assert list(mixed.index) == list(pd.DatetimeIndex(["2026-07-02", "2026-07-03"]))


def test_column_mismatch_takes_intersection():
    open_df = pd.DataFrame(
        {"2330": [100.0], "6669": [50.0]},
        index=pd.DatetimeIndex(["2026-07-07"]),
    )
    close_df = pd.DataFrame(
        {"2330": [200.0]},
        index=pd.DatetimeIndex(["2026-07-07"]),
    )
    mixed = mix_open_close(open_df, close_df, buy_weekday=0)
    assert list(mixed.columns) == ["2330"]
    assert mixed.loc["2026-07-07", "2330"] == 200.0  # 週二取收盤
