"""ProfitLossDAO 測試（真實 SQLite on tmp file）。

冪等性是這張表的核心保證：fetcher 每天重抓一段重疊區間，若唯一索引失效，
同一筆平倉會被重複累加成假的獲利。
"""

import datetime
import decimal
import enum

import pytest

from dao.profit_loss_dao import ProfitLossDAO


class _Cond(enum.Enum):
    Cash = "Cash"


TS = datetime.datetime(2026, 7, 31, 20, 30, 0)


@pytest.fixture
def dao(tmp_path):
    return ProfitLossDAO(db_path=str(tmp_path / "test.db"))


def _record(**overrides):
    record = {
        "trade_date": "2026-07-31",
        "stock_id": "2330",
        "stock_name": "台積電",
        "quantity": 0.002,
        "price": 1450.0,
        "pnl": 120.5,
        "pr_ratio": 4.34,
        "cond": "Cash",
        "dseq": "A0001",
        "seqno": "000123",
        "raw_data": {"code": "2330", "quantity": 2},
    }
    record.update(overrides)
    return record


def test_insert_and_query_roundtrip(dao):
    inserted = dao.insert_profit_loss(1, [_record()], fetch_timestamp=TS)

    assert inserted == 1
    rows = dao.get_profit_loss(1, datetime.date(2026, 7, 1), datetime.date(2026, 7, 31))
    assert len(rows) == 1
    assert rows[0]["stock_id"] == "2330"
    assert rows[0]["pnl"] == 120.5
    assert rows[0]["pr_ratio"] == 4.34


def test_duplicate_record_is_ignored(dao):
    dao.insert_profit_loss(1, [_record()], fetch_timestamp=TS)
    inserted = dao.insert_profit_loss(1, [_record()], fetch_timestamp=TS)

    assert inserted == 0
    rows = dao.get_profit_loss(1, datetime.date(2026, 7, 1), datetime.date(2026, 7, 31))
    assert len(rows) == 1


def test_same_stock_different_trade_is_kept(dao):
    # 同一天同一檔可能有多筆平倉（不同委託），dseq/seqno 不同就是不同筆
    dao.insert_profit_loss(1, [_record()], fetch_timestamp=TS)
    inserted = dao.insert_profit_loss(
        1, [_record(dseq="A0002", seqno="000124", pnl=-40.0)], fetch_timestamp=TS
    )

    assert inserted == 1
    rows = dao.get_profit_loss(1, datetime.date(2026, 7, 1), datetime.date(2026, 7, 31))
    assert sorted(row["pnl"] for row in rows) == [-40.0, 120.5]


def test_accounts_are_isolated(dao):
    dao.insert_profit_loss(1, [_record()], fetch_timestamp=TS)
    dao.insert_profit_loss(2, [_record()], fetch_timestamp=TS)

    rows = dao.get_profit_loss(2, datetime.date(2026, 7, 1), datetime.date(2026, 7, 31))
    assert len(rows) == 1


def test_query_filters_by_date_range(dao):
    dao.insert_profit_loss(1, [
        _record(trade_date="2026-06-30", dseq="B1"),
        _record(trade_date="2026-07-31", dseq="B2"),
    ], fetch_timestamp=TS)

    rows = dao.get_profit_loss(1, datetime.date(2026, 7, 1), datetime.date(2026, 7, 31))
    assert [row["trade_date"] for row in rows] == ["2026-07-31"]


def test_latest_trade_date(dao):
    assert dao.get_latest_trade_date(1) is None

    dao.insert_profit_loss(1, [
        _record(trade_date="2026-06-30", dseq="B1"),
        _record(trade_date="2026-07-31", dseq="B2"),
    ], fetch_timestamp=TS)

    assert dao.get_latest_trade_date(1) == "2026-07-31"


def test_raw_data_with_enum_and_decimal_does_not_raise(dao):
    # 同 inventory_dao 的 default=str 回歸鎖：shioaji 物件欄位含 enum / Decimal
    raw = {"cond": _Cond.Cash, "price": decimal.Decimal("1450.5")}
    dao.insert_profit_loss(1, [_record(raw_data=raw)], fetch_timestamp=TS)

    rows = dao.get_profit_loss(1, datetime.date(2026, 7, 1), datetime.date(2026, 7, 31))
    assert len(rows) == 1


def test_none_timestamp_raises(dao):
    with pytest.raises(ValueError):
        dao.insert_profit_loss(1, [_record()], fetch_timestamp=None)
