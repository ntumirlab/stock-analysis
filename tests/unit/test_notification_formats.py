"""core/notification_formats 的單元測試：下單摘要與清單解析結果的訊息組字。"""

from core.notification_formats import (
    format_no_new_recommendations,
    format_order_summary,
    format_parse_failures,
    format_parse_success,
    shares_from_lots,
)
from dao.recommendation_dao import RecommendationRecord, Stock


def _orders():
    return [
        {"action": "BUY", "stock_id": "2330", "stock_name": "台積電", "quantity": 0.002},
        {"action": "BUY", "stock_id": "2634", "stock_name": "漢翔", "quantity": 0.063},
        {"action": "SELL", "stock_id": "3017", "stock_name": None, "quantity": 1.5},
    ]


def test_shares_from_lots():
    assert shares_from_lots(0.001) == 1
    assert shares_from_lots(0.063) == 63
    assert shares_from_lots("0.002") == 2
    assert shares_from_lots(1.5) == 1500  # 整張也正確換算


def test_order_summary_content():
    body = format_order_summary(_orders(), view_only=False)
    assert "委託 3 筆" in body
    assert "BUY  2330 台積電 2 股" in body
    assert "BUY  2634 漢翔 63 股" in body
    assert "SELL 3017 1500 股" in body  # stock_name 缺時只顯示代號
    assert "模擬" not in body


def test_order_summary_view_only_tag():
    body = format_order_summary(_orders(), view_only=True)
    assert "模擬" in body


def test_parse_success_content():
    records = [
        RecommendationRecord(date="2026-07-05", stocks=[
            Stock(id="3017", sentiment="STRONG_BUY", name="奇鋐"),
            Stock(id="2330", sentiment="BUY", name=None),
        ]),
        RecommendationRecord(date="2026-07-12", stocks=[
            Stock(id="2059", sentiment="BUY", name="川湖"),
        ]),
    ]
    body = format_parse_success("weekly", records)
    assert "weekly 清單解析入庫 2 份" in body
    assert "*2026-07-05*（2 檔）" in body
    assert "3017 奇鋐" in body
    assert "、2330" in body  # name 缺時只顯示代號
    assert "*2026-07-12*（1 檔）" in body


def test_parse_failures_content():
    body = format_parse_failures("weekly", ["2026-07-05", "2026-07-12"])
    assert "2 份解析失敗" in body
    assert "2026-07-05" in body
    assert "2026-07-12" in body


def test_no_new_recommendations_content():
    body = format_no_new_recommendations("monthly")
    assert "monthly" in body
    assert "沒有發現新的推薦清單" in body
