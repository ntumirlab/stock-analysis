"""OrderDAO 測試（真實 SQLite on tmp file）。"""

import datetime
import pytest

from dao.order_dao import OrderDAO


@pytest.fixture
def dao(tmp_path):
    return OrderDAO(db_path=str(tmp_path / "test.db"))


TS = datetime.datetime(2026, 7, 6, 8, 10, 0)


def _order(stock_id="2330", action="BUY"):
    return {
        "action": action,
        "stock_id": stock_id,
        "stock_name": "台積電",
        "quantity": 1.0,
        "limit_price": 1000.0,
        "extra_bid_pct": 0.0,
        "order_condition": "CASH",
    }


def test_insert_and_query_roundtrip(dao):
    dao.insert_order_logs([_order("2330"), _order("2317", "SELL")], account_id=1, order_timestamp=TS)

    orders = dao.get_orders_by_account_and_date(1, TS.date())
    assert len(orders) == 2
    assert {o["stock_id"] for o in orders} == {"2330", "2317"}
    assert orders[0]["order_timestamp"] == "2026-07-06 08:10:00"


def test_view_only_flag_persisted(dao):
    dao.insert_order_logs([_order()], account_id=1, order_timestamp=TS, view_only=True)
    orders = dao.get_orders_by_account_and_date(1, TS.date())
    assert orders[0]["view_only"] == 1


def test_orders_isolated_by_account(dao):
    dao.insert_order_logs([_order("2330")], account_id=1, order_timestamp=TS)
    dao.insert_order_logs([_order("2317")], account_id=2, order_timestamp=TS)

    assert {o["stock_id"] for o in dao.get_orders_by_account_and_date(1, TS.date())} == {"2330"}
    assert {o["stock_id"] for o in dao.get_orders_by_account_and_date(2, TS.date())} == {"2317"}


def test_orders_isolated_by_date(dao):
    other_day = datetime.datetime(2026, 7, 7, 8, 10, 0)
    dao.insert_order_logs([_order("2330")], account_id=1, order_timestamp=TS)
    dao.insert_order_logs([_order("2317")], account_id=1, order_timestamp=other_day)

    assert {o["stock_id"] for o in dao.get_orders_by_account_and_date(1, TS.date())} == {"2330"}


def test_none_timestamp_raises(dao):
    with pytest.raises(ValueError):
        dao.insert_order_logs([_order()], account_id=1, order_timestamp=None)


def test_available_years_months_days(dao):
    dao.insert_order_logs([_order()], account_id=1, order_timestamp=TS)
    assert dao.get_available_years(1) == ["2026"]
    assert dao.get_available_months(1, "2026") == ["07"]
    assert dao.get_available_days(1, "2026", "07") == ["06"]
