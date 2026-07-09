"""推薦清單拉取的純邏輯（發布 JSON 驗證與入庫差集，無外部依賴）。

供 jobs/recommendations_fetcher.py（Lite 端）使用：publisher 發布的
goldenai-reclist.v1 JSON 在寫入本地 DB 前的 schema 驗證、以及「哪些
日期還沒入庫」的判斷抽在這裡，讓單元測試可以覆蓋。
"""

import json
import logging
import math
from datetime import datetime
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from core.recommendation_publishing import SCHEMA_VERSION
from dao.recommendation_dao import RecommendationRecord, Stock

logger = logging.getLogger(__name__)


class PayloadValidationError(ValueError):
    """發布 JSON 不符 schema（版本、欄位、priority 順序）——整份拒收不入庫。"""


class FileUnavailableError(Exception):
    """單一檔案下載不可用（權限、已移走、非二進位檔）——單檔隔離，不擋其他日期。

    由呼叫端的 download_text 拋出（core 不依賴 Drive SDK 的例外型別）。
    """


def date_from_publish_filename(filename: str, frequency: str) -> Optional[str]:
    """從發布檔名（如 weekly_2026-07-06.json）取回日期。

    非本 frequency、或日期格式不符的檔案回 None（資料夾裡的雜檔直接忽略，
    monthly 檔在 weekly task 下也會因 prefix 不符被略過）。
    """
    prefix = f"{frequency}_"
    suffix = ".json"
    if not (filename.startswith(prefix) and filename.endswith(suffix)):
        return None

    date = filename[len(prefix):-len(suffix)]
    try:
        parsed = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return None
    # strptime 對月/日不要求補零（'2026-7-5' 也放行），但 DB 與 payload 都是
    # 補零格式，回寫比對不一致的一律視為雜檔，免得它天天觸發交叉比對錯誤
    if parsed.strftime("%Y-%m-%d") != date:
        return None
    return date


def dates_missing_from_db(drive_dates: Iterable[str],
                          db_dates: Iterable[str]) -> List[str]:
    """回傳 Drive 已發布、但本地 DB 還沒有的日期（升冪）。"""
    existing = set(db_dates)
    return sorted(d for d in set(drive_dates) if d not in existing)


# 發布 schema 的 stock 欄位固定出現（缺值為 null），少一個 key 都算走樣
_REQUIRED_STOCK_FIELDS = (
    "priority", "stock_id", "stock_name", "sentiment", "target_price", "stop_loss",
)


def _require_optional_number(stock: dict, field: str, position: int) -> None:
    value = stock[field]
    if value is None:
        return
    # bool 是 int 的子類別，須明確排除（true 不是合法的價格）；
    # json.loads 會把非標準的 NaN/Infinity 常數解析成 float，一併拒收
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise PayloadValidationError(
            f"stocks[{position}].{field} must be a finite number or null, got {value!r}"
        )


def parse_publish_payload(payload_text: str, expected_frequency: str,
                          expected_date: str) -> RecommendationRecord:
    """把發布 JSON 驗證後還原成 RecommendationRecord。

    任一驗證不過即丟 PayloadValidationError，整份不入庫：寧可缺當週清單
    被 freshness 檢查擋下單，也不寫入半套或走樣的清單。expected_* 取自
    檔名，與內容交叉比對，防止檔名與內文不一致的檔案混入。
    """
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as e:
        raise PayloadValidationError(f"invalid JSON: {e}")

    if not isinstance(payload, dict):
        raise PayloadValidationError("payload must be a JSON object")

    schema = payload.get("schema")
    if schema != SCHEMA_VERSION:
        raise PayloadValidationError(
            f"unsupported schema {schema!r} (expected {SCHEMA_VERSION!r})"
        )

    if payload.get("frequency") != expected_frequency:
        raise PayloadValidationError(
            f"frequency mismatch: payload has {payload.get('frequency')!r}, "
            f"filename implies {expected_frequency!r}"
        )

    if payload.get("date") != expected_date:
        raise PayloadValidationError(
            f"date mismatch: payload has {payload.get('date')!r}, "
            f"filename implies {expected_date!r}"
        )

    stocks_data = payload.get("stocks")
    if not isinstance(stocks_data, list) or not stocks_data:
        raise PayloadValidationError("stocks must be a non-empty list")

    for position, stock in enumerate(stocks_data):
        if not isinstance(stock, dict):
            raise PayloadValidationError(f"stocks[{position}] is not an object")

        missing = [f for f in _REQUIRED_STOCK_FIELDS if f not in stock]
        if missing:
            raise PayloadValidationError(
                f"stocks[{position}] missing fields: {', '.join(missing)}"
            )

        if isinstance(stock["priority"], bool) or stock["priority"] != position + 1:
            raise PayloadValidationError(
                f"stocks[{position}].priority must be {position + 1} "
                f"(1-based, contiguous), got {stock['priority']!r}"
            )

        for field in ("stock_id", "sentiment"):
            if not stock[field] or not isinstance(stock[field], str):
                raise PayloadValidationError(
                    f"stocks[{position}].{field} must be a non-empty string"
                )

        if stock["stock_name"] is not None and not isinstance(stock["stock_name"], str):
            raise PayloadValidationError(
                f"stocks[{position}].stock_name must be a string or null, "
                f"got {stock['stock_name']!r}"
            )

        _require_optional_number(stock, "target_price", position)
        _require_optional_number(stock, "stop_loss", position)

    return RecommendationRecord(
        date=expected_date,
        stocks=[
            Stock(
                id=stock["stock_id"],
                sentiment=stock["sentiment"],
                TP=stock["target_price"],
                SL=stock["stop_loss"],
                name=stock["stock_name"],
            )
            for stock in stocks_data
        ],
    )


def fetch_missing_records(remote_files: Dict[str, str],
                          download_text: Callable[[str], str],
                          dao, task_name: str
                          ) -> Tuple[List[RecommendationRecord], List[str]]:
    """把 Drive 已發布、本地 DB 還沒有的日期抓齊入庫。

    remote_files 為 {檔名: file_id}，download_text 以 file_id 取回 JSON 內文，
    dao 為該 frequency 的 RecommendationDAO。抽在 core 讓單元測試不必
    import jobs 套件（jobs/__init__ 頂層會拉 finlab，CI 環境沒有）。

    逐日下載→驗證→入庫（每日各自 atomic）；單一檔案的問題（schema 走樣、
    非 UTF-8、單檔下載不可用 FileUnavailableError）記入失敗清單後繼續處理
    其他日期——歷史壞檔不得擋住當週清單入庫。全域性的基礎設施錯誤
    （認證失效、list 失敗等）則直接上拋。
    回傳 (本次新入庫的 records, 拒收的日期)。
    """
    db_dates = {record.date for record in dao.load()}

    remote_dates = {}
    for filename, file_id in remote_files.items():
        date = date_from_publish_filename(filename, task_name)
        if date:
            remote_dates[date] = file_id

    missing = dates_missing_from_db(remote_dates.keys(), db_dates)

    if not missing:
        logger.info(f"{task_name}: local DB already has all {len(remote_dates)} published dates.")
        return [], []

    logger.info(f"{task_name}: fetching {len(missing)} new dates: {missing}")

    fetched = []
    failed_dates = []
    for date in missing:
        try:
            record = parse_publish_payload(download_text(remote_dates[date]), task_name, date)
        except (PayloadValidationError, UnicodeDecodeError, FileUnavailableError) as e:
            logger.error(f"{task_name} {date}: payload rejected — {e}")
            failed_dates.append(date)
            continue
        dao.add_record(record)
        fetched.append(record)

    logger.info(
        f"{task_name}: fetched {len(fetched)} files into local DB"
        + (f", rejected {len(failed_dates)}: {failed_dates}" if failed_dates else ".")
    )
    return fetched, failed_dates
