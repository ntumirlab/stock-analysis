"""InventoryDAO 測試（真實 SQLite on tmp file）。

raw_data 含 enum / Decimal 的案例鎖住 default=str 修復：shioaji position
物件的欄位型別 json 化失敗曾是 20:30 帳務抓取的潛在炸點。
"""

import datetime
import decimal
import enum

import pytest

from dao.inventory_dao import InventoryDAO


class _Action(enum.Enum):
    Buy = "Buy"


TS = datetime.datetime(2026, 7, 6, 20, 30, 0)


@pytest.fixture
def dao(tmp_path):
    return InventoryDAO(db_path=str(tmp_path / "test.db"))


def _item(**overrides):
    item = {
        "stock_id": "2330",
        "stock_name": "台積電",
        "quantity": 0.002,
        "last_price": 1450.0,
        "pnl": 120.5,
        "raw_data": {"code": "2330", "quantity": 2},
    }
    item.update(overrides)
    return item


def test_insert_and_query_roundtrip(dao):
    dao.insert_inventory_data(1, [_item()], fetch_timestamp=TS)

    rows = dao.get_inventories_by_account_and_date(1, TS.date())
    assert len(rows) == 1
    row = rows[0]
    assert row["stock_id"] == "2330"
    assert row["quantity"] == 0.002
    assert row["raw_data"] == {"code": "2330", "quantity": 2}


def test_raw_data_with_enum_and_decimal_does_not_raise(dao):
    # 回歸鎖（2026-07-06 default=str 修復）：shioaji position 的 __dict__
    # 含 enum（direction）與 Decimal，json.dumps 預設會 TypeError
    raw = {"direction": _Action.Buy, "price": decimal.Decimal("1450.5")}
    dao.insert_inventory_data(1, [_item(raw_data=raw)], fetch_timestamp=TS)

    rows = dao.get_inventories_by_account_and_date(1, TS.date())
    assert rows[0]["raw_data"]["direction"] == "_Action.Buy"
    assert rows[0]["raw_data"]["price"] == "1450.5"


def test_none_timestamp_raises(dao):
    with pytest.raises(ValueError):
        dao.insert_inventory_data(1, [_item()], fetch_timestamp=None)


def test_query_filters_by_date_and_account(dao):
    dao.insert_inventory_data(1, [_item()], fetch_timestamp=TS)
    dao.insert_inventory_data(1, [_item(stock_id="2059")],
                              fetch_timestamp=TS + datetime.timedelta(days=1))

    same_day = dao.get_inventories_by_account_and_date(1, TS.date())
    assert [r["stock_id"] for r in same_day] == ["2330"]

    other_account = dao.get_inventories_by_account_and_date(2, TS.date())
    assert other_account == []
