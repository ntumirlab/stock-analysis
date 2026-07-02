"""Gemini 回應 JSON → RecommendationRecord 解析測試。"""

import json
import pytest

from core.recommendation_parsing import parse_recommendation_response


def _stocks_json(stocks):
    return json.dumps({"stocks": stocks})


class TestValidResponse:
    def test_full_fields(self):
        text = _stocks_json([
            {"id": "2330", "sentiment": "STRONG_BUY", "TP": 1200.0, "SL": 950.0, "name": "台積電"},
            {"id": "2317", "sentiment": "BUY", "name": "鴻海"},
        ])
        record = parse_recommendation_response(text, "2026-06-28")
        assert record.date == "2026-06-28"
        assert [s.id for s in record.stocks] == ["2330", "2317"]
        assert record.stocks[0].TP == 1200.0
        assert record.stocks[0].SL == 950.0
        assert record.stocks[1].TP is None

    def test_priority_order_preserved(self):
        # rank 依 stocks 陣列順序，順序不能被打亂
        text = _stocks_json([
            {"id": str(i), "sentiment": "BUY"} for i in [2337, 2382, 2330, 2610]
        ])
        record = parse_recommendation_response(text, "2026-06-28")
        assert [s.id for s in record.stocks] == ["2337", "2382", "2330", "2610"]

    def test_numeric_id_coerced_to_str(self):
        text = _stocks_json([{"id": 2330, "sentiment": "BUY"}])
        record = parse_recommendation_response(text, "2026-06-28")
        assert record.stocks[0].id == "2330"

    def test_empty_stocks_returns_empty_record(self):
        # 目前行為：stocks 為空陣列仍回傳 record（文件化既有行為）
        record = parse_recommendation_response(_stocks_json([]), "2026-06-28")
        assert record is not None
        assert record.stocks == []


class TestInvalidResponse:
    def test_missing_stocks_field_returns_none(self):
        assert parse_recommendation_response('{"foo": 1}', "2026-06-28") is None

    def test_malformed_json_raises(self):
        # 目前行為：壞 JSON 直接拋例外，由呼叫端 retry/錯誤處理接手
        with pytest.raises(json.JSONDecodeError):
            parse_recommendation_response("not json{", "2026-06-28")

    def test_unexpected_stock_field_raises(self):
        # 目前行為：Stock 建構子不接受未知欄位（LLM 多給欄位會炸）
        with pytest.raises(TypeError):
            parse_recommendation_response(
                _stocks_json([{"id": "2330", "sentiment": "BUY", "confidence": 0.9}]),
                "2026-06-28")
