"""Telegram 通知訊息的組字邏輯（純函式，無外部依賴）。

供 jobs/order_executor.py（下單摘要）與 jobs/recommendations_parser.py
（清單解析結果）使用，抽在這裡讓訊息內容可以被單元測試覆蓋。
"""

from typing import Dict, Iterable, List


def shares_from_lots(quantity) -> int:
    """張 → 股（finlab 委託量以張計，0.001 張 = 1 股）。"""
    return int(round(float(quantity) * 1000))


def format_order_summary(order_logs: List[Dict], view_only: bool) -> str:
    """把 order_executor 抽出的委託 log 組成摘要內文。

    order_logs 每筆需含 action / stock_id / quantity（張），stock_name 可缺。
    """
    title = f"📋 *委託 {len(order_logs)} 筆*"
    if view_only:
        title += "　🧪 模擬 (view\\_only)"

    lines = []
    for order in order_logs:
        name = order.get("stock_name")
        shares = shares_from_lots(order["quantity"])
        entry = f"{order['action']:<4} {order['stock_id']}"
        if name:
            entry += f" {name}"
        entry += f" {shares} 股"
        lines.append(entry)

    return (
        title
        + "\n```\n" + "\n".join(lines) + "\n```"
        + "\n_（委託清單，實際成交以券商回報為準）_"
    )


def format_parse_success(task_name: str, records: Iterable) -> str:
    """清單解析成功的內文。records 為 RecommendationRecord（duck typing：.date / .stocks[.id/.name]）。"""
    records = list(records)
    parts = []
    for record in records:
        stocks = "、".join(
            f"{stock.id} {stock.name}" if stock.name else str(stock.id)
            for stock in record.stocks
        )
        parts.append(f"*{record.date}*（{len(record.stocks)} 檔）:\n{stocks}")

    return f"{task_name} 清單解析入庫 {len(records)} 份\n\n" + "\n\n".join(parts)


def format_parse_failures(task_name: str, failed_dates: List[str]) -> str:
    """有新檔但 Gemini 解析失敗（重試耗盡）的內文。此前為靜默失敗，現在明確警告。"""
    return (
        f"{task_name} 清單有 {len(failed_dates)} 份解析失敗（Gemini 重試耗盡）: "
        f"{', '.join(failed_dates)}\n"
        f"清單未入庫，隔日下單會被 freshness 檢查擋下；請查 log 後手動重跑 parser。"
    )


def format_universe_missing(missing_ids: List[str]) -> str:
    """推薦清單股票不在 finlab 股價表（position 對齊時會被靜默排除）的警告內文。"""
    ids = "、".join(str(sid) for sid in missing_ids)
    return (
        f"推薦清單中有 {len(missing_ids)} 檔不在 finlab 股價資料範圍，"
        f"將無法買進（position 對齊時被排除）：{ids}\n"
        f"可能原因：新上市、代號有誤、資料未更新。"
    )


def format_no_new_recommendations(task_name: str) -> str:
    """週日執行卻沒有新清單檔案的警告內文（weekly 與 monthly 上游均每週更新）。"""
    return (
        f"{task_name} 本次執行沒有發現新的推薦清單檔案。\n"
        f"週日晚間應有新清單（weekly 與 monthly 均每週更新）—— "
        f"請確認上游已上傳、drive_fetcher 是否正常。"
    )
