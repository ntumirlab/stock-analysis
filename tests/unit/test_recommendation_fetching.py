"""core/recommendation_fetching 的單元測試：發布 JSON 驗證、入庫差集與拉取流程。"""

import json

import pytest

from core.recommendation_fetching import (
    FileUnavailableError,
    PayloadValidationError,
    date_from_publish_filename,
    dates_missing_from_db,
    fetch_missing_records,
    parse_publish_payload,
)
from core.recommendation_publishing import build_publish_payload, publish_filename
from dao.recommendation_dao import RecommendationDAO, RecommendationRecord, Stock


def _record():
    return RecommendationRecord(date="2026-07-05", stocks=[
        Stock(id="3017", sentiment="STRONG_BUY", TP=1250.0, SL=980.0, name="奇鋐"),
        Stock(id="2330", sentiment="BUY", name="台積電"),
    ])


def _payload_dict(**overrides):
    """以 publisher 的真實輸出為基底，加壞一個欄位做反例。"""
    payload = json.loads(build_publish_payload(_record(), "weekly", "2026-07-06T07:30:00+08:00"))
    payload.update(overrides)
    return payload


# ---- date_from_publish_filename ----

def test_filename_roundtrip():
    assert date_from_publish_filename("weekly_2026-07-05.json", "weekly") == "2026-07-05"


def test_filename_ignores_other_frequency():
    # monthly 檔在 weekly task 下不視為發布檔
    assert date_from_publish_filename("monthly_2026-07-05.json", "weekly") is None


def test_filename_ignores_junk():
    assert date_from_publish_filename("weekly_backup.json", "weekly") is None
    assert date_from_publish_filename("weekly_2026-07-05.json.bak", "weekly") is None
    assert date_from_publish_filename("unrelated.txt", "weekly") is None
    assert date_from_publish_filename("weekly_2026-13-99.json", "weekly") is None


def test_filename_rejects_non_zero_padded_date():
    # strptime 對月/日不要求補零，但 DB 與 payload 都是補零格式；
    # 不補零的檔名視為雜檔，否則會天天觸發交叉比對錯誤
    assert date_from_publish_filename("weekly_2026-7-5.json", "weekly") is None


# ---- dates_missing_from_db ----

def test_dates_missing_from_db():
    missing = dates_missing_from_db(
        ["2026-07-05", "2026-06-28", "2026-06-21"],
        ["2026-06-28"],
    )
    assert missing == ["2026-06-21", "2026-07-05"]


def test_no_dates_missing():
    assert dates_missing_from_db(["2026-07-05"], ["2026-07-05"]) == []
    assert dates_missing_from_db([], []) == []


# ---- parse_publish_payload：正例 ----

def test_parse_roundtrips_publisher_output():
    # publisher 的輸出必須能被 fetcher 原樣還原（兩端 schema 對齊的守門測試）
    payload_text = build_publish_payload(_record(), "weekly", "2026-07-06T07:30:00+08:00")
    record = parse_publish_payload(payload_text, "weekly", "2026-07-05")

    assert record.date == "2026-07-05"
    assert [s.id for s in record.stocks] == ["3017", "2330"]
    first = record.stocks[0]
    assert (first.name, first.sentiment, first.TP, first.SL) == ("奇鋐", "STRONG_BUY", 1250.0, 980.0)


def test_parse_accepts_null_tp_sl_and_name():
    # 缺 TP/SL/名稱以 null 表示是合法輸出，不得拒收
    payload_text = build_publish_payload(
        RecommendationRecord(date="2026-07-05", stocks=[Stock(id="2330", sentiment="BUY")]),
        "weekly", "x",
    )
    record = parse_publish_payload(payload_text, "weekly", "2026-07-05")
    stock = record.stocks[0]
    assert (stock.TP, stock.SL, stock.name) == (None, None, None)


# ---- parse_publish_payload：反例（整份拒收） ----

def test_rejects_invalid_json():
    with pytest.raises(PayloadValidationError, match="invalid JSON"):
        parse_publish_payload("{not json", "weekly", "2026-07-05")


def test_rejects_non_object_payload():
    with pytest.raises(PayloadValidationError, match="JSON object"):
        parse_publish_payload("[1, 2]", "weekly", "2026-07-05")


def test_rejects_unknown_schema_version():
    # 未來版本寧可拒收擋單，也不猜相容
    payload = _payload_dict(schema="goldenai-reclist.v2")
    with pytest.raises(PayloadValidationError, match="unsupported schema"):
        parse_publish_payload(json.dumps(payload), "weekly", "2026-07-05")


def test_rejects_frequency_mismatch():
    with pytest.raises(PayloadValidationError, match="frequency mismatch"):
        parse_publish_payload(json.dumps(_payload_dict()), "monthly", "2026-07-05")


def test_rejects_date_mismatch():
    # 檔名日期與內文日期不一致（檔案被改名或混入）
    with pytest.raises(PayloadValidationError, match="date mismatch"):
        parse_publish_payload(json.dumps(_payload_dict()), "weekly", "2026-06-28")


def test_rejects_empty_stocks():
    payload = _payload_dict(stocks=[])
    with pytest.raises(PayloadValidationError, match="non-empty"):
        parse_publish_payload(json.dumps(payload), "weekly", "2026-07-05")


def test_rejects_non_object_stock_entry():
    payload = _payload_dict()
    payload["stocks"][0] = "2330"
    with pytest.raises(PayloadValidationError, match="not an object"):
        parse_publish_payload(json.dumps(payload), "weekly", "2026-07-05")


def test_rejects_missing_stock_field():
    payload = _payload_dict()
    del payload["stocks"][0]["stop_loss"]
    with pytest.raises(PayloadValidationError, match="missing fields: stop_loss"):
        parse_publish_payload(json.dumps(payload), "weekly", "2026-07-05")


def test_rejects_priority_gap():
    payload = _payload_dict()
    payload["stocks"][1]["priority"] = 3
    with pytest.raises(PayloadValidationError, match="priority"):
        parse_publish_payload(json.dumps(payload), "weekly", "2026-07-05")


def test_rejects_empty_stock_id():
    payload = _payload_dict()
    payload["stocks"][0]["stock_id"] = ""
    with pytest.raises(PayloadValidationError, match="stock_id"):
        parse_publish_payload(json.dumps(payload), "weekly", "2026-07-05")


def test_rejects_non_numeric_target_price():
    payload = _payload_dict()
    payload["stocks"][0]["target_price"] = "1250"
    with pytest.raises(PayloadValidationError, match="target_price"):
        parse_publish_payload(json.dumps(payload), "weekly", "2026-07-05")


def test_rejects_boolean_price_and_priority():
    # bool 是 int 子類別，必須明確排除（true 不是價格、也不是順位）
    payload = _payload_dict()
    payload["stocks"][0]["stop_loss"] = True
    with pytest.raises(PayloadValidationError, match="stop_loss"):
        parse_publish_payload(json.dumps(payload), "weekly", "2026-07-05")

    payload = _payload_dict()
    payload["stocks"][0]["priority"] = True  # True == 1，仍須拒收
    with pytest.raises(PayloadValidationError, match="priority"):
        parse_publish_payload(json.dumps(payload), "weekly", "2026-07-05")


def test_rejects_non_finite_prices():
    # json 的非標準 NaN/Infinity 常數會被 json.loads 解析成 float，
    # publisher 端的 json.dumps（allow_nan 預設 True）也能原樣輸出——必須拒收
    payload = _payload_dict()
    payload["stocks"][0]["target_price"] = float("nan")
    with pytest.raises(PayloadValidationError, match="target_price"):
        parse_publish_payload(json.dumps(payload), "weekly", "2026-07-05")

    payload = _payload_dict()
    payload["stocks"][0]["stop_loss"] = float("inf")
    with pytest.raises(PayloadValidationError, match="stop_loss"):
        parse_publish_payload(json.dumps(payload), "weekly", "2026-07-05")


def test_rejects_non_string_stock_name():
    payload = _payload_dict()
    payload["stocks"][0]["stock_name"] = 123
    with pytest.raises(PayloadValidationError, match="stock_name"):
        parse_publish_payload(json.dumps(payload), "weekly", "2026-07-05")


# ---- fetch_missing_records（拉取流程：假下載函式 + tmp DB） ----

@pytest.fixture
def dao(tmp_path):
    return RecommendationDAO(db_path=str(tmp_path / "test.db"), frequency="weekly")


def _payload_text(date):
    record = RecommendationRecord(date=date, stocks=[
        Stock(id="2330", sentiment="BUY", name="台積電"),
    ])
    return build_publish_payload(record, "weekly", "2026-07-06T07:30:00+08:00")


def _downloader(files_by_id):
    downloaded = []

    def download_text(file_id):
        downloaded.append(file_id)
        return files_by_id[file_id]

    return download_text, downloaded


def test_fetches_only_missing_dates(dao):
    dao.add_record(RecommendationRecord(date="2026-06-28", stocks=[
        Stock(id="2330", sentiment="BUY"),
    ]))
    # file_id 直接用檔名，簡化對照
    remote_files = {
        publish_filename("weekly", "2026-06-28"): publish_filename("weekly", "2026-06-28"),
        publish_filename("weekly", "2026-07-05"): publish_filename("weekly", "2026-07-05"),
        "unrelated.txt": "unrelated.txt",
    }
    download_text, downloaded = _downloader({
        publish_filename("weekly", "2026-07-05"): _payload_text("2026-07-05"),
    })

    fetched, failed = fetch_missing_records(remote_files, download_text, dao, "weekly")

    # 只下載、只入庫缺的那天；已入庫的與雜檔連碰都不碰
    assert [record.date for record in fetched] == ["2026-07-05"]
    assert failed == []
    assert downloaded == [publish_filename("weekly", "2026-07-05")]
    assert dao.get_by_date("2026-07-05") is not None


def test_noop_when_db_up_to_date(dao):
    dao.add_record(RecommendationRecord(date="2026-07-05", stocks=[
        Stock(id="2330", sentiment="BUY"),
    ]))
    remote_files = {publish_filename("weekly", "2026-07-05"): "id-1"}
    download_text, downloaded = _downloader({})

    assert fetch_missing_records(remote_files, download_text, dao, "weekly") == ([], [])
    assert downloaded == []


def test_bad_payload_rejected_without_partial_write(dao):
    payload = json.loads(_payload_text("2026-07-05"))
    payload["stocks"][0]["priority"] = 99
    remote_files = {publish_filename("weekly", "2026-07-05"): "id-1"}
    download_text, _ = _downloader({"id-1": json.dumps(payload)})

    fetched, failed = fetch_missing_records(remote_files, download_text, dao, "weekly")

    # 整份拒收：記入失敗清單，DB 不得留下該日期的任何殘料
    assert fetched == []
    assert failed == ["2026-07-05"]
    assert dao.get_by_date("2026-07-05") is None


def test_old_bad_file_does_not_block_current_week(dao):
    # 歷史壞檔只記失敗，不得擋住其後日期（當週清單）入庫
    remote_files = {
        publish_filename("weekly", "2026-06-28"): "id-1",
        publish_filename("weekly", "2026-07-05"): "id-2",
    }
    download_text, _ = _downloader({
        "id-1": "{broken",
        "id-2": _payload_text("2026-07-05"),
    })

    fetched, failed = fetch_missing_records(remote_files, download_text, dao, "weekly")

    assert [record.date for record in fetched] == ["2026-07-05"]
    assert failed == ["2026-06-28"]
    assert dao.get_by_date("2026-07-05") is not None
    assert dao.get_by_date("2026-06-28") is None


def test_non_utf8_file_isolated_like_bad_payload(dao):
    # 非 UTF-8 檔在 decode 就爆，也必須走「單檔拒收、其他照入庫」的隔離
    def download_text(file_id):
        if file_id == "id-1":
            return bytes([0xFF, 0xFE]).decode("utf-8")  # raises UnicodeDecodeError
        return _payload_text("2026-07-05")

    remote_files = {
        publish_filename("weekly", "2026-06-28"): "id-1",
        publish_filename("weekly", "2026-07-05"): "id-2",
    }

    fetched, failed = fetch_missing_records(remote_files, download_text, dao, "weekly")

    assert [record.date for record in fetched] == ["2026-07-05"]
    assert failed == ["2026-06-28"]


def test_unavailable_file_isolated_like_bad_payload(dao):
    # 單檔下載不可用（fetcher 端把 403/404 轉成 FileUnavailableError）
    # 也必須走「單檔拒收、其他照入庫」的隔離，不得中斷整輪擋住當週清單
    def download_text(file_id):
        if file_id == "id-1":
            raise FileUnavailableError("403 fileNotDownloadable")
        return _payload_text("2026-07-05")

    remote_files = {
        publish_filename("weekly", "2026-06-28"): "id-1",
        publish_filename("weekly", "2026-07-05"): "id-2",
    }

    fetched, failed = fetch_missing_records(remote_files, download_text, dao, "weekly")

    assert [record.date for record in fetched] == ["2026-07-05"]
    assert failed == ["2026-06-28"]
    assert dao.get_by_date("2026-07-05") is not None


def test_earlier_good_dates_survive_later_bad_one(dao):
    # 逐日 atomic：較早日期入庫成功，之後的壞檔不回滾好的
    remote_files = {
        publish_filename("weekly", "2026-06-28"): "id-1",
        publish_filename("weekly", "2026-07-05"): "id-2",
    }
    download_text, _ = _downloader({
        "id-1": _payload_text("2026-06-28"),
        "id-2": "{broken",
    })

    fetched, failed = fetch_missing_records(remote_files, download_text, dao, "weekly")

    assert [record.date for record in fetched] == ["2026-06-28"]
    assert failed == ["2026-07-05"]
    assert dao.get_by_date("2026-06-28") is not None
    assert dao.get_by_date("2026-07-05") is None
