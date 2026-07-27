"""已實現損益抓取區間的純邏輯測試。

重疊回抓的天數與日期正規化是「重複入庫 → 損益灌水」的第一道防線，
配合 DAO 的唯一索引一起看（見 test_profit_loss_dao.py）。
"""

import datetime

import pytest

from core.profit_loss_range import (
    DEFAULT_INITIAL_LOOKBACK_DAYS,
    DEFAULT_OVERLAP_DAYS,
    normalize_trade_date,
    resolve_fetch_range,
)

TODAY = datetime.date(2026, 7, 27)


@pytest.mark.parametrize("raw,expected", [
    ("2026-07-31", "2026-07-31"),
    ("2026/07/31", "2026-07-31"),
    ("20260731", "2026-07-31"),
    ("2026-07-31 09:10:00", "2026-07-31"),
    (datetime.date(2026, 7, 31), "2026-07-31"),
    (datetime.datetime(2026, 7, 31, 9, 10), "2026-07-31"),
    (None, None),
])
def test_normalize_trade_date_accepts_broker_variants(raw, expected):
    assert normalize_trade_date(raw) == expected


def test_first_fetch_uses_initial_lookback():
    begin, end = resolve_fetch_range(TODAY, latest_trade_date=None)

    assert end == TODAY
    assert begin == TODAY - datetime.timedelta(days=DEFAULT_INITIAL_LOOKBACK_DAYS)


def test_routine_fetch_overlaps_backwards_from_latest():
    # 重疊的用意：昨天的 20:30 job 掛掉時，今天這趟要能把昨天的平倉補回來
    begin, end = resolve_fetch_range(TODAY, latest_trade_date="2026-07-24")

    assert begin == datetime.date(2026, 7, 24) - datetime.timedelta(days=DEFAULT_OVERLAP_DAYS)
    assert end == TODAY


def test_routine_fetch_accepts_date_object_as_latest():
    begin, _ = resolve_fetch_range(TODAY, latest_trade_date=datetime.date(2026, 7, 24))

    assert begin == datetime.date(2026, 7, 19)


def test_explicit_range_is_respected_for_backfill():
    begin, end = resolve_fetch_range(
        TODAY,
        latest_trade_date="2026-07-24",
        begin_date=datetime.date(2025, 5, 1),
        end_date=datetime.date(2026, 4, 30),
    )

    assert begin == datetime.date(2025, 5, 1)
    assert end == datetime.date(2026, 4, 30)


def test_inverted_range_raises():
    with pytest.raises(ValueError):
        resolve_fetch_range(
            TODAY,
            begin_date=datetime.date(2026, 7, 27),
            end_date=datetime.date(2026, 7, 1),
        )
