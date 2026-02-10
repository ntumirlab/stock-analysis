from finlab import data
from finlab.backtest import sim


class RevenuePriceStrategy:
    def __init__(self):
        self.report = None

    def run_strategy(self):
        close = data.get('price:收盤價')
        vol = data.get('price:成交股數')
        rev = data.get('monthly_revenue:當月營收')
        rev_yoy_growth = data.get('monthly_revenue:去年同月增減(%)')

        # 近2月平均營收
        rev_ma = rev.average(2)

        # 近2月平均營收創12個月來新高
        condition1 = rev_ma == rev_ma.rolling(12, min_periods=6).max()
        # 近5日內有2日股價創新高
        condition2 = (close == close.rolling(200).max()).sustain(5, 2)
        # 五日成交均量大於500張
        condition3 = vol.average(5) > 500 * 1000
        conditions = condition1 & condition2 & condition3

        # 符合選股條件的名單中，再選出單月營收年增率前10強，並在營收公告截止日換股。
        position = rev_yoy_growth * conditions
        position = position[position > 0].is_largest(10).reindex(rev.index_str_to_date().index, method='ffill')

        # 設定position_limit避免重壓
        self.report = sim(
            position=position,
            stop_loss=0.2,
            take_profit=0.8,
            position_limit=0.25,
            fee_ratio=1.425 / 1000 * 0.3,
            name="營收股價雙渦輪",
            live_performance_start='2022-10-01',
        )

        return self.report

    def get_report(self):
        return self.report if self.report else "report物件為空，請先運行策略"


# Example usage
strategy = RevenuePriceStrategy()
strategy.run_strategy()
print(strategy.report)
