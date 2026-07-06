"""推薦清單發布的純邏輯（JSON 序列化與發布差集，無外部依賴）。

供 jobs/recommendations_publisher.py 使用：發布 schema 與「哪些日期還沒
發布」的判斷抽在這裡，讓單元測試可以覆蓋。
"""

import json
from typing import Iterable, List, Optional

from dao.recommendation_dao import RecommendationRecord

SCHEMA_VERSION = "goldenai-reclist.v1"


def is_folder_id_configured(folder_id: Optional[str]) -> bool:
    """publish_folder_id 空值、或 .env 缺變數而殘留 ${...} 佔位符，都視為未設定。"""
    return bool(folder_id) and "${" not in folder_id


def publish_filename(frequency: str, date: str) -> str:
    return f"{frequency}_{date}.json"


def dates_missing_from_drive(db_dates: Iterable[str],
                             existing_filenames: Iterable[str],
                             frequency: str) -> List[str]:
    """回傳 DB 有、但 Drive 上還沒有對應 JSON 檔的日期（升冪）。"""
    published = set(existing_filenames)
    return sorted(d for d in db_dates if publish_filename(frequency, d) not in published)


def build_publish_payload(record: RecommendationRecord, frequency: str,
                          published_at: str) -> str:
    """把 RecommendationRecord 序列化成發布用 JSON 字串。

    欄位固定出現（缺值輸出 null）、priority 為 1-based 清單順序，
    消費端（Lite 下單系統）以 schema 版本做相容性判斷。
    """
    payload = {
        "schema": SCHEMA_VERSION,
        "frequency": frequency,
        "date": record.date,
        "published_at": published_at,
        "stocks": [
            {
                "priority": idx + 1,
                "stock_id": stock.id,
                "stock_name": stock.name,
                "sentiment": stock.sentiment,
                "target_price": stock.TP,
                "stop_loss": stock.SL,
            }
            for idx, stock in enumerate(record.stocks)
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
