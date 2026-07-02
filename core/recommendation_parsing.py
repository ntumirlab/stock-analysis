"""推薦清單的檔名解析與 LLM 回應解析（純邏輯，無外部依賴）。

從 jobs/recommendations_parser.py 抽出，讓上游檔名格式與 Gemini 回應格式
這兩個已知的回歸熱點可以被單元測試覆蓋。
"""

import json
import re
import logging
from datetime import datetime
from typing import Optional

from dao.recommendation_dao import RecommendationRecord, Stock

logger = logging.getLogger(__name__)

TASK_EXPECTED_DAYS = {'weekly': '5', 'monthly': '30'}


def extract_recommendation_date(filename: str, task_name: str) -> Optional[str]:
    """從推薦清單檔名解析日期（YYYY-MM-DD），格式或天數不符則回傳 None。

    支援兩種上游命名（新格式優先）：
    - 新（2026-06-28 起）：{YYYYMMDD}_{HHMMSS}_tw_{N}d_recommendation.md
    - 舊：{YYYYMMDD}_{HHMM}_推薦股票_台股{N}日_金策智能.md
    """
    match = re.match(r"^(\d{8})_\d{6}_tw_(\d+)d_recommendation\.md$", filename)
    if not match:
        match = re.match(r"^(\d{8})_\d{4}_推薦股票_台股(\d+)日_金策智能\.md$", filename)
    if not match:
        return None

    date_raw, days = match.group(1), match.group(2)
    expected_days = TASK_EXPECTED_DAYS.get(task_name)
    if expected_days and days != expected_days:
        logger.warning(f"Skipping {filename}: {days}d does not match task '{task_name}' (expected {expected_days}d)")
        return None

    try:
        return datetime.strptime(date_raw, "%Y%m%d").strftime("%Y-%m-%d")
    except ValueError:
        return None


def parse_recommendation_response(response_text: str, date_str: str) -> Optional[RecommendationRecord]:
    """把 LLM 回傳的 JSON 文字解析成 RecommendationRecord。

    - 缺 'stocks' 欄位 → 記 warning、回傳 None
    - JSON 格式錯誤或 stock 欄位不符 Stock 建構子 → 直接拋例外
      （由呼叫端的 retry/錯誤處理決定怎麼辦，維持原本行為）
    """
    parsed_json = json.loads(response_text)

    if 'stocks' not in parsed_json:
        logger.warning(f"JSON missing 'stocks' field in {date_str}")
        return None

    stocks = [Stock(**stock_dict) for stock_dict in parsed_json.get('stocks', [])]
    record = RecommendationRecord(date=date_str, stocks=stocks)

    logger.info(f"[{date_str}] Parsed {len(stocks)} stocks: {[s.id for s in stocks]}")
    return record
