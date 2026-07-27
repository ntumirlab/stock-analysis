from dao.profit_loss_dao import ProfitLossDAO
from dao.inventory_dao import InventoryDAO


def _safe_ratio(pnl, cost):
    """報酬率 = 損益 / 投入成本（%）。

    成本為 0 或負數時回 None 而非 0——UI 顯示「—」比顯示假的 0% 誠實。
    分母刻意用「實際投入成本」而非帳戶總資產：總資產會被出入金污染
    （帳戶資金頁的月度熱力圖就有這個問題），成本則不會。
    """
    if not cost or cost <= 0:
        return None
    return pnl / cost * 100


def _realized_cost_basis(record):
    """推算單筆已實現損益的投入成本 ＝ 賣出價金 − 損益。

    券商的 price 是成交價（平倉時即賣出價），乘上股數得到賣出價金；
    再減掉已實現損益就是這批股票的成本。三個欄位在 shioaji 的
    StockProfitLoss 都是必填，缺漏會在解析階段就失敗，不會是 None。

    刻意不用 pr_ratio 回推：其刻度（百分比 4.32 或小數 0.0432）未經實盤確認，
    賭錯會讓成本差 100 倍。這裡的算式只用加減乘除，沒有刻度問題。
    """
    pnl = record.get('pnl') or 0
    price = record.get('price') or 0
    shares = (record.get('quantity') or 0) * 1000

    cost = price * shares - pnl
    return cost if cost > 0 else 0


def _aggregate(records):
    """把一批損益明細彙總成 {pnl, cost, ratio, count}。

    已實現與未實現的成本來源不同（前者由賣出價金回推、後者用成本均價），
    但兩者都已在明細上算好 cost_basis，因此彙總邏輯只有這一份。
    """
    pnl = sum(record.get('pnl') or 0 for record in records)
    cost = sum(record.get('cost_basis') or 0 for record in records)
    return {
        'pnl': pnl,
        'cost': cost,
        'ratio': _safe_ratio(pnl, cost),
        'count': len(records),
    }


class ProfitLossService:
    def __init__(self, db_path="data_prod.db"):
        self.profit_loss_dao = ProfitLossDAO(db_path)
        self.inventory_dao = InventoryDAO(db_path)

    def get_realized_records(self, account_id, start_date, end_date):
        """取得期間內的已實現損益明細，每筆補上推算的成本與報酬率。

        報酬率一律自行以「損益 ÷ 成本」計算，不直接顯示券商的 pr_ratio——
        後者的刻度未確認，原值仍留在 DB 供日後比對。

        Returns:
            list[dict]: 含 trade_date / stock_id / stock_name / quantity /
                price / pnl / pr_ratio（券商原值）/ cost_basis / ratio（自算）
        """
        records = self.profit_loss_dao.get_profit_loss(account_id, start_date, end_date)
        for record in records:
            record['cost_basis'] = _realized_cost_basis(record)
            record['ratio'] = _safe_ratio(record.get('pnl') or 0, record['cost_basis'])
        return records

    def get_unrealized_records(self, account_id, query_date=None):
        """取得指定日期的未實現損益（現有持股）。

        成本均價只存在 inventory_history 的 raw_data JSON 內，因此這裡直接
        走 DAO 而非 InventoryService（後者會濾掉 raw_data）。

        Args:
            account_id (int): 帳戶ID
            query_date (datetime.date, optional): 取哪天的持股快照，
                未給則取最新一筆；帳戶完全沒有快照時回空清單
                （防呆放在這層，呼叫端不必各自處理 None）

        Returns:
            list[dict]: 含 stock_id / stock_name / quantity / last_price /
                cost_price / pnl / cost_basis / ratio
        """
        if query_date is None:
            query_date = self.inventory_dao.get_latest_inventory_date(account_id)
        if query_date is None:
            return []

        inventories = self.inventory_dao.get_inventories_by_account_and_date(
            account_id, query_date
        )

        # 同一天可能有多批快照（重跑 job），取最新一批：DAO 已依時間新到舊排序，
        # 同一檔股票只留第一次出現的那筆
        seen = set()
        records = []
        for inventory in inventories:
            stock_id = inventory.get('stock_id')
            if stock_id in seen:
                continue
            seen.add(stock_id)

            raw = inventory.get('raw_data') or {}
            cost_price = raw.get('price') or 0
            shares = raw.get('quantity')
            if shares is None:
                shares = (inventory.get('quantity') or 0) * 1000

            pnl = inventory.get('pnl') or 0
            cost_basis = float(cost_price) * float(shares)

            records.append({
                'stock_id': stock_id,
                'stock_name': inventory.get('stock_name'),
                'quantity': inventory.get('quantity'),
                'last_price': inventory.get('last_price'),
                'cost_price': float(cost_price),
                'pnl': pnl,
                'cost_basis': cost_basis,
                'ratio': _safe_ratio(pnl, cost_basis),
            })

        return records

    def get_summary(self, account_id, start_date, end_date, inventory_date=None):
        """彙總已實現 / 未實現 / 合計的金額與報酬率。

        Args:
            account_id (int): 帳戶ID
            start_date (datetime.date): 已實現損益的起始日
            end_date (datetime.date): 已實現損益的結束日
            inventory_date (datetime.date, optional): 未實現損益取哪天的持股快照，
                預設為 DB 內最新的一筆（未實現沒有區間概念，且 20:30 抓取前用今天會撲空）

        Returns:
            dict: {'realized': {...}, 'unrealized': {...}, 'total': {...}}
                  每個區塊含 pnl / cost / ratio / count
        """
        realized = self.get_realized_records(account_id, start_date, end_date)
        unrealized = self.get_unrealized_records(account_id, inventory_date)

        return {
            'realized': _aggregate(realized),
            'unrealized': _aggregate(unrealized),
            'total': _aggregate(realized + unrealized),
        }

    def get_cumulative_realized(self, account_id, start_date, end_date):
        """已實現損益的每日與累積序列，供折線圖使用。

        Returns:
            list[dict]: [{'date': 'YYYY-MM-DD', 'daily_pnl': x, 'cumulative_pnl': y}, ...]
                        依日期由舊到新
        """
        records = self.profit_loss_dao.get_profit_loss(account_id, start_date, end_date)

        daily = {}
        for record in records:
            trade_date = record.get('trade_date')
            if not trade_date:
                continue
            daily[trade_date] = daily.get(trade_date, 0) + (record.get('pnl') or 0)

        series = []
        cumulative = 0
        for trade_date in sorted(daily.keys()):
            cumulative += daily[trade_date]
            series.append({
                'date': trade_date,
                'daily_pnl': daily[trade_date],
                'cumulative_pnl': cumulative,
            })

        return series
