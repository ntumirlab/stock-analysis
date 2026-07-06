"""core/recommendation_publishing 的單元測試：發布 schema 序列化與日期差集。"""

import json

from core.recommendation_publishing import (
    SCHEMA_VERSION,
    build_publish_payload,
    dates_missing_from_drive,
    is_folder_id_configured,
    publish_filename,
)
from dao.recommendation_dao import RecommendationRecord, Stock


def _record():
    return RecommendationRecord(date="2026-07-05", stocks=[
        Stock(id="3017", sentiment="STRONG_BUY", TP=1250.0, SL=980.0, name="奇鋐"),
        Stock(id="2330", sentiment="BUY", name="台積電"),
    ])


def test_payload_schema_and_stock_fields():
    payload = json.loads(build_publish_payload(_record(), "weekly", "2026-07-06T07:30:00+08:00"))
    assert payload["schema"] == SCHEMA_VERSION
    assert payload["frequency"] == "weekly"
    assert payload["date"] == "2026-07-05"
    assert payload["published_at"] == "2026-07-06T07:30:00+08:00"
    assert payload["stocks"][0] == {
        "priority": 1,
        "stock_id": "3017",
        "stock_name": "奇鋐",
        "sentiment": "STRONG_BUY",
        "target_price": 1250.0,
        "stop_loss": 980.0,
    }


def test_payload_keeps_null_fields():
    # 缺 TP/SL 時欄位仍要存在且為 null，消費端 schema 才穩定
    payload = json.loads(build_publish_payload(_record(), "weekly", "x"))
    tsmc = payload["stocks"][1]
    assert tsmc["priority"] == 2
    assert tsmc["target_price"] is None
    assert tsmc["stop_loss"] is None


def test_publish_filename():
    assert publish_filename("weekly", "2026-07-05") == "weekly_2026-07-05.json"


def test_dates_missing_from_drive():
    missing = dates_missing_from_drive(
        ["2026-07-05", "2026-06-28"],
        ["weekly_2026-06-28.json", "unrelated.txt"],
        "weekly",
    )
    assert missing == ["2026-07-05"]


def test_missing_ignores_other_frequency_files():
    # monthly 的檔案不算 weekly 已發布
    assert dates_missing_from_drive(
        ["2026-07-05"], ["monthly_2026-07-05.json"], "weekly"
    ) == ["2026-07-05"]


def test_folder_id_configured():
    assert is_folder_id_configured("1AbCdEfG")
    assert not is_folder_id_configured(None)
    assert not is_folder_id_configured("")
    # .env 缺變數時 ConfigLoader 會殘留字面佔位符，須視為未設定
    assert not is_folder_id_configured("${WEEKLY_PUBLISH_FOLDER_ID}")
