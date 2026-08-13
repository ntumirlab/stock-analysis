"""logger_manager 兩條 log 抽取 regex 的測試。

extract_order_logs 是「下單 → DB 紀錄 → TG 摘要」整條鏈的單點：regex 對不上
finlab 輸出格式時會靜默抽到 0 筆（單照下但無紀錄無通知）。此處用實際
生產 log 的格式鎖住契約——升級 finlab 後若格式改變，這裡會先紅。
"""

import datetime

import pytest

from utils.logger_manager import LoggerManager


@pytest.fixture
def manager(tmp_path):
    return LoggerManager(
        base_log_directory=str(tmp_path),
        current_datetime=datetime.datetime(2026, 7, 6, 8, 10, 0),
    )


def _write_log(tmp_path, text):
    path = tmp_path / "order.log"
    path.write_text(text, encoding="utf-8")
    return str(path)


# 取自 2026-07-06 實盤 log 的真實格式
ORDER_LOG = """\
2026-07-06 00:10:12,560 - utils.reservation_handler - INFO - 無警示股，跳過圈存流程
2026-07-06 00:10:13,154 - finlab.online.core.executor - INFO - BUY         2059       X 0.001      @ HIGHEST      CASH
2026-07-06 00:10:15,037 - finlab.online.core.executor - INFO - BUY         2330       X 0.002      @ HIGHEST      CASH
2026-07-06 00:10:15,438 - finlab.online.core.executor - INFO - SELL        1101       X 1.5        @ 35.5 with extra bid 2.0% CASH
2026-07-06 00:10:16,621 - __main__ - INFO - Portfolio synced
"""


def test_extract_order_logs_fields(manager, tmp_path):
    orders = manager.extract_order_logs(_write_log(tmp_path, ORDER_LOG))
    assert len(orders) == 3

    buy = orders[0]
    assert buy["action"] == "BUY"
    assert buy["stock_id"] == "2059"
    assert buy["quantity"] == 0.001
    assert buy["limit_price"] is None  # HIGHEST（市價單）存 NULL
    assert buy["extra_bid_pct"] == 0.0
    assert buy["order_condition"] == "CASH"

    sell = orders[2]
    assert sell["action"] == "SELL"
    assert sell["quantity"] == 1.5
    assert sell["limit_price"] == 35.5
    assert sell["extra_bid_pct"] == 0.02  # 2.0% -> 0.02
    assert sell["order_condition"] == "CASH"


def test_extract_order_logs_lowest_is_null_price(manager, tmp_path):
    log = "... - INFO - SELL        2330       X 0.002      @ LOWEST      CASH\n"
    orders = manager.extract_order_logs(_write_log(tmp_path, log))
    assert len(orders) == 1
    assert orders[0]["limit_price"] is None


def test_extract_order_logs_ignores_noise(manager, tmp_path):
    log = """\
2026-07-06 00:10:05,188 - finlab.portfolio.sync_update - INFO - 預計可買入金額: 97525 (70.00%)
2026-07-06 00:10:16,621 - __main__ - INFO - Portfolio synced
"""
    assert manager.extract_order_logs(_write_log(tmp_path, log)) == []


# 取自 2026-08-13 實盤 log：finlab 浮點殘差（~1e-18 張）印出的幽靈委託，
# 換算張/股皆為 0、實際未送單，不應進 order_history 與下單摘要通知
PHANTOM_ORDER_LOG = """\
2026-08-13 08:10:21,422 - finlab.online.core.executor - INFO - SELL        2317       X 0.0        @ LOWEST       CASH
2026-08-13 08:10:21,422 - finlab.online.core.executor - INFO - SELL        2634       X 0.0        @ LOWEST       CASH
2026-08-13 08:10:21,424 - __main__ - INFO - Portfolio synced
"""


def test_extract_order_logs_skips_zero_share_orders(manager, tmp_path):
    assert manager.extract_order_logs(_write_log(tmp_path, PHANTOM_ORDER_LOG)) == []


def test_extract_order_logs_keeps_one_share_order(manager, tmp_path):
    """0.001 張 = 1 股是最小的真實委託，不可被幽靈單的過濾一起吃掉。"""
    log = PHANTOM_ORDER_LOG + (
        "2026-08-13 08:10:21,423 - finlab.online.core.executor - INFO - "
        "BUY         2330       X 0.001      @ HIGHEST      CASH\n"
    )
    orders = manager.extract_order_logs(_write_log(tmp_path, log))
    assert len(orders) == 1
    assert orders[0]["stock_id"] == "2330"
    assert orders[0]["quantity"] == 0.001


# 取自 finlab show_alerting_stocks 的輸出格式（賣出為負值）
ALERTING_LOG = """\
2026-07-06 08:10:00,000 - finlab.online.order_executor - INFO - 買入 8101  0.429 張 - 總價約         2672.67
2026-07-06 08:10:00,001 - finlab.online.order_executor - INFO - 賣出 2492 -0.004 張 - 總價約        -1497.60
"""


def test_extract_alerting_stocks(manager, tmp_path):
    stocks = manager.extract_alerting_stocks(_write_log(tmp_path, ALERTING_LOG))
    assert len(stocks) == 2

    buy = stocks[0]
    assert buy["action"] == "買入"
    assert buy["stock_id"] == "8101"
    assert buy["quantity"] == 0.429
    assert buy["total_amount"] == 2672.67

    sell = stocks[1]
    assert sell["action"] == "賣出"
    assert sell["quantity"] == -0.004
    assert sell["total_amount"] == -1497.60
