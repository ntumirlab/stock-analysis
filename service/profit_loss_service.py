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
    """推算單筆已實現損益的投入成本。

    優先用券商自己的報酬率回推（pnl / pr_ratio），這樣彙總後的比率與券商
    單筆顯示的口徑一致；pr_ratio 為 0（或券商未提供）時退回 price × 股數，
    此時不含手續費與交易稅，屬近似值。
    """
    pnl = record.get('pnl') or 0
    pr_ratio = record.get('pr_ratio') or 0
    if pr_ratio:
        return abs(pnl / (pr_ratio / 100))

    price = record.get('price') or 0
    shares = (record.get('quantity') or 0) * 1000
    return price * shares


class ProfitLossService:
    def __init__(self, db_path="data_prod.db"):
        self.profit_loss_dao = ProfitLossDAO(db_path)
        self.inventory_dao = InventoryDAO(db_path)

    def get_realized_records(self, account_id, start_date, end_date):
        """取得期間內的已實現損益明細，每筆補上推算的成本與報酬率。

        Returns:
            list[dict]: 含 trade_date / stock_id / stock_name / quantity /
                price / pnl / pr_ratio / cost_basis
        """
        records = self.profit_loss_dao.get_profit_loss(account_id, start_date, end_date)
        for record in records:
            record['cost_basis'] = _realized_cost_basis(record)
        return records

    def get_unrealized_records(self, account_id, query_date):
        """取得指定日期的未實現損益（現有持股）。

        成本均價只存在 inventory_history 的 raw_data JSON 內，因此這裡直接
        走 DAO 而非 InventoryService（後者會濾掉 raw_data）。

        Returns:
            list[dict]: 含 stock_id / stock_name / quantity / last_price /
                cost_price / pnl / cost_basis / pr_ratio
        """
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
                'pr_ratio': _safe_ratio(pnl, cost_basis),
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
        if inventory_date is None:
            inventory_date = self.inventory_dao.get_latest_inventory_date(account_id)

        realized = self.get_realized_records(account_id, start_date, end_date)
        unrealized = (
            self.get_unrealized_records(account_id, inventory_date)
            if inventory_date else []
        )

        realized_pnl = sum(r.get('pnl') or 0 for r in realized)
        realized_cost = sum(r.get('cost_basis') or 0 for r in realized)
        unrealized_pnl = sum(r.get('pnl') or 0 for r in unrealized)
        unrealized_cost = sum(r.get('cost_basis') or 0 for r in unrealized)

        total_pnl = realized_pnl + unrealized_pnl
        total_cost = realized_cost + unrealized_cost

        return {
            'realized': {
                'pnl': realized_pnl,
                'cost': realized_cost,
                'ratio': _safe_ratio(realized_pnl, realized_cost),
                'count': len(realized),
            },
            'unrealized': {
                'pnl': unrealized_pnl,
                'cost': unrealized_cost,
                'ratio': _safe_ratio(unrealized_pnl, unrealized_cost),
                'count': len(unrealized),
            },
            'total': {
                'pnl': total_pnl,
                'cost': total_cost,
                'ratio': _safe_ratio(total_pnl, total_cost),
                'count': len(realized) + len(unrealized),
            },
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
