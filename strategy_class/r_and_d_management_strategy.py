from finlab import data
from finlab.backtest import sim


class RAndDManagementStrategy:
    def __init__(self):
        self.report = None

    def run_strategy(self):
        close = data.get('price:收盤價')
        volume = data.get('price:成交股數')

        rd_ratio = data.get('fundamental_features:研究發展費用率')
        pm_ratio = data.get('fundamental_features:管理費用率')
        eq_ratio = data.get('fundamental_features:淨值除資產').deadline()

        rd_pm = rd_ratio / pm_ratio
        eq_price = eq_ratio / close.reindex(eq_ratio.index, method='ffill')

        rebalance = eq_price.index

        position = eq_price[
            (
                (close > close.average(60))
                & (volume > 200_000)
                & (volume.average(10) > volume.average(60))
                & (rd_pm.deadline().rank(axis=1, pct=True) > 0.5)
            ).reindex(rebalance)
        ].is_largest(20)

        rebalance = eq_ratio.index

        self.report = sim(position.loc['2020':], resample=rebalance, upload=False)
        return self.report

    def get_report(self):
        return self.report if self.report else "report物件為空，請先運行策略"


# Example usage:
if __name__ == "__main__":
    strategy = RAndDManagementStrategy()
    report = strategy.run_strategy()
    print(report)
