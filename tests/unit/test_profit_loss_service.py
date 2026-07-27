"""ProfitLossService 彙總測試。

重點在報酬率的分母：用「實際投入成本」而非帳戶總資產。帳戶資金頁的月度
熱力圖用總資產當分母，遇到出入金會顯示假報酬（junting 帳戶 2026-04 出金後
總資產從 11.8 萬掉到 69 元即為一例），這裡的算法必須不受出入金影響。
"""

import datetime

import pytest

from dao.inventory_dao import InventoryDAO
from dao.profit_loss_dao import ProfitLossDAO
from service.profit_loss_service import ProfitLossService

REALIZED_TS = datetime.datetime(2026, 7, 31, 20, 30, 0)
INVENTORY_TS = datetime.datetime(2026, 7, 31, 20, 30, 0)
START = datetime.date(2026, 7, 1)
END = datetime.date(2026, 7, 31)


@pytest.fixture
def service(tmp_path):
    db_path = str(tmp_path / "test.db")
    return ProfitLossService(db_path=db_path)


@pytest.fixture
def daos(service, tmp_path):
    return service.profit_loss_dao, service.inventory_dao


def _realized(**overrides):
    record = {
        "trade_date": "2026-07-31",
        "stock_id": "2330",
        "stock_name": "台積電",
        "quantity": 0.002,
        "price": 1450.0,
        "pnl": 100.0,
        "pr_ratio": 5.0,       # → 成本 2000
        "cond": "Cash",
        "dseq": "A1",
        "seqno": "S1",
        "raw_data": {},
    }
    record.update(overrides)
    return record


def _holding(**overrides):
    item = {
        "stock_id": "2454",
        "stock_name": "聯發科",
        "quantity": 0.003,
        "last_price": 1100.0,
        "pnl": -300.0,
        "raw_data": {"code": "2454", "quantity": 3, "price": 1200.0},  # 成本 3600
    }
    item.update(overrides)
    return item


def test_realized_ratio_uses_broker_pr_ratio_for_cost(service, daos):
    pnl_dao, _ = daos
    pnl_dao.insert_profit_loss(1, [_realized()], fetch_timestamp=REALIZED_TS)

    summary = service.get_summary(1, START, END)

    assert summary["realized"]["pnl"] == 100.0
    assert summary["realized"]["cost"] == pytest.approx(2000.0)
    assert summary["realized"]["ratio"] == pytest.approx(5.0)
    assert summary["realized"]["count"] == 1


def test_realized_cost_falls_back_to_price_times_shares(service, daos):
    # pr_ratio 為 0（券商未提供）時退回 price × 股數，屬不含費用的近似值
    pnl_dao, _ = daos
    pnl_dao.insert_profit_loss(
        1, [_realized(pr_ratio=0.0, price=50.0, quantity=1.0, pnl=500.0)],
        fetch_timestamp=REALIZED_TS
    )

    summary = service.get_summary(1, START, END)

    assert summary["realized"]["cost"] == pytest.approx(50.0 * 1000)
    assert summary["realized"]["ratio"] == pytest.approx(1.0)


def test_unrealized_ratio_uses_cost_price_from_raw_data(service, daos):
    _, inventory_dao = daos
    inventory_dao.insert_inventory_data(1, [_holding()], fetch_timestamp=INVENTORY_TS)

    summary = service.get_summary(1, START, END)

    assert summary["unrealized"]["pnl"] == -300.0
    assert summary["unrealized"]["cost"] == pytest.approx(3600.0)
    assert summary["unrealized"]["ratio"] == pytest.approx(-300.0 / 3600.0 * 100)


def test_total_combines_both_sides(service, daos):
    pnl_dao, inventory_dao = daos
    pnl_dao.insert_profit_loss(1, [_realized()], fetch_timestamp=REALIZED_TS)
    inventory_dao.insert_inventory_data(1, [_holding()], fetch_timestamp=INVENTORY_TS)

    summary = service.get_summary(1, START, END)

    assert summary["total"]["pnl"] == pytest.approx(-200.0)
    assert summary["total"]["cost"] == pytest.approx(5600.0)
    assert summary["total"]["ratio"] == pytest.approx(-200.0 / 5600.0 * 100)


def test_zero_cost_yields_none_ratio_not_zero(service):
    # 沒有任何資料時分母為 0；回 None 讓 UI 顯示「—」，顯示 0% 會被誤讀成損益兩平
    summary = service.get_summary(1, START, END)

    assert summary["total"]["pnl"] == 0
    assert summary["total"]["ratio"] is None


def test_unrealized_uses_latest_snapshot_when_date_not_given(service, daos):
    _, inventory_dao = daos
    inventory_dao.insert_inventory_data(
        1, [_holding(pnl=-999.0)],
        fetch_timestamp=datetime.datetime(2026, 7, 20, 20, 30)
    )
    inventory_dao.insert_inventory_data(
        1, [_holding(pnl=-300.0)],
        fetch_timestamp=datetime.datetime(2026, 7, 31, 20, 30)
    )

    summary = service.get_summary(1, START, END)

    assert summary["unrealized"]["pnl"] == -300.0


def test_duplicate_snapshot_same_day_counted_once(service, daos):
    # 同一天重跑 20:30 job 會留兩批快照，未實現不可加倍計算
    _, inventory_dao = daos
    inventory_dao.insert_inventory_data(
        1, [_holding()], fetch_timestamp=datetime.datetime(2026, 7, 31, 20, 30)
    )
    inventory_dao.insert_inventory_data(
        1, [_holding()], fetch_timestamp=datetime.datetime(2026, 7, 31, 21, 15)
    )

    summary = service.get_summary(1, START, END)

    assert summary["unrealized"]["count"] == 1
    assert summary["unrealized"]["pnl"] == -300.0


def test_cumulative_series_is_chronological(service, daos):
    pnl_dao, _ = daos
    pnl_dao.insert_profit_loss(1, [
        _realized(trade_date="2026-07-10", dseq="A1", pnl=100.0),
        _realized(trade_date="2026-07-17", dseq="A2", pnl=-40.0),
        _realized(trade_date="2026-07-17", dseq="A3", pnl=10.0),
        _realized(trade_date="2026-07-31", dseq="A4", pnl=50.0),
    ], fetch_timestamp=REALIZED_TS)

    series = service.get_cumulative_realized(1, START, END)

    assert [point["date"] for point in series] == ["2026-07-10", "2026-07-17", "2026-07-31"]
    assert [point["daily_pnl"] for point in series] == [100.0, -30.0, 50.0]
    assert [point["cumulative_pnl"] for point in series] == [100.0, 70.0, 120.0]


def test_realized_outside_range_excluded(service, daos):
    pnl_dao, _ = daos
    pnl_dao.insert_profit_loss(1, [
        _realized(trade_date="2026-06-30", dseq="A1", pnl=999.0),
        _realized(trade_date="2026-07-31", dseq="A2", pnl=100.0),
    ], fetch_timestamp=REALIZED_TS)

    summary = service.get_summary(1, START, END)

    assert summary["realized"]["pnl"] == 100.0
