"""BalanceDAO 測試（真實 SQLite on tmp file）。"""

import datetime
import pytest

from dao.balance_dao import BalanceDAO


@pytest.fixture
def dao(tmp_path):
    return BalanceDAO(db_path=str(tmp_path / "test.db"))


TS = datetime.datetime(2026, 7, 2, 20, 30, 0)

BALANCE = {
    "bank_balance": 139322.0,
    "settlements": 0.0,
    "adjusted_bank_balance": 139322.0,
    "market_value": 0.0,
    "total_assets": 139322.0,
}


def test_insert_and_get_latest_roundtrip(dao):
    dao.insert_balance(1, BALANCE, fetch_timestamp=TS)

    latest = dao.get_latest_balance(1)
    assert latest["bank_balance"] == 139322.0
    assert latest["total_assets"] == 139322.0
    assert latest["fetch_timestamp"] == "2026-07-02 20:30:00"


def test_get_latest_returns_newest(dao):
    dao.insert_balance(1, BALANCE, fetch_timestamp=TS)
    newer = dict(BALANCE, total_assets=150000.0)
    dao.insert_balance(1, newer, fetch_timestamp=TS + datetime.timedelta(days=1))

    assert dao.get_latest_balance(1)["total_assets"] == 150000.0


def test_none_timestamp_raises(dao):
    with pytest.raises(ValueError):
        dao.insert_balance(1, BALANCE, fetch_timestamp=None)


def test_balance_history_date_range(dao):
    dao.insert_balance(1, BALANCE, fetch_timestamp=TS)
    dao.insert_balance(1, BALANCE, fetch_timestamp=TS + datetime.timedelta(days=10))

    history = dao.get_balance_history(
        1, start_date=TS.date(), end_date=(TS + datetime.timedelta(days=1)).date())
    assert len(history) == 1


def test_isolated_by_account(dao):
    dao.insert_balance(1, BALANCE, fetch_timestamp=TS)
    assert dao.get_latest_balance(2) is None
