"""已實現損益抓取的純邏輯：日期正規化與查詢區間推算。

抽出到 core/ 的理由同 trading_cycles：jobs/ 會 eager import finlab 與 shioaji，
CI 的 unit tests 碰不到；這裡不 import 任何券商或 finlab 相關套件。
"""

import datetime

# 首次抓取（DB 內無該帳戶資料）時往回抓的天數
DEFAULT_INITIAL_LOOKBACK_DAYS = 90
# 例行抓取時，自已入庫最新成交日再往前重疊的天數。
# 重疊是為了補「某天 20:30 job 掛掉」的漏，重複資料由 DAO 唯一索引擋掉。
DEFAULT_OVERLAP_DAYS = 5


def normalize_trade_date(value):
    """把券商回傳的日期正規化成 YYYY-MM-DD 字串。

    券商欄位型別不保證（可能是 date 物件或字串、可能帶斜線或不帶分隔），
    但 DAO 的唯一索引與區間查詢都靠這個字串比對，格式不一致會讓同一筆
    平倉重複入庫、損益被灌水。

    Args:
        value: 券商回傳的日期（str / date / datetime / None）

    Returns:
        str | None: "YYYY-MM-DD"，無法解析時原樣回傳字串
    """
    if value is None:
        return None
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.strftime("%Y-%m-%d")

    text = str(value).strip().replace("/", "-")
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    if len(text) >= 10:
        return text[:10]
    return text


def resolve_fetch_range(today, latest_trade_date=None, begin_date=None, end_date=None,
                        overlap_days=DEFAULT_OVERLAP_DAYS,
                        initial_lookback_days=DEFAULT_INITIAL_LOOKBACK_DAYS):
    """決定這次要向券商查詢的日期區間。

    三種情境：
    1. 呼叫端明確指定（一次性 backfill）→ 照用
    2. DB 已有資料 → 自最新成交日往前重疊數天，補可能漏掉的那幾天
    3. DB 無資料（首次）→ 往回抓 initial_lookback_days 天

    Args:
        today (datetime.date): 今天（由呼叫端傳入，方便測試與時區控制）
        latest_trade_date (str | datetime.date, optional): DB 內最新成交日
        begin_date (datetime.date, optional): 明確指定的起始日
        end_date (datetime.date, optional): 明確指定的結束日
        overlap_days (int): 情境 2 的回溯重疊天數
        initial_lookback_days (int): 情境 3 的回溯天數

    Returns:
        tuple[datetime.date, datetime.date]: (begin_date, end_date)

    Raises:
        ValueError: 起始日晚於結束日
    """
    if end_date is None:
        end_date = today

    if begin_date is None:
        if latest_trade_date:
            latest = latest_trade_date
            if isinstance(latest, str):
                latest = datetime.datetime.strptime(
                    normalize_trade_date(latest), "%Y-%m-%d"
                ).date()
            begin_date = latest - datetime.timedelta(days=overlap_days)
        else:
            begin_date = today - datetime.timedelta(days=initial_lookback_days)

    if begin_date > end_date:
        raise ValueError(
            f"begin_date {begin_date} is later than end_date {end_date}"
        )

    return begin_date, end_date
