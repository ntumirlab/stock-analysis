from strategy_class.roger_tw_strategy_base import RogerTWStrategyBase

class RogerTWStrategyWeekly(RogerTWStrategyBase):
    def __init__(self, config_path="config.yaml"):
        super().__init__(task_name="weekly", config_path=config_path)

    def _apply_trading_window(self, position):
        """
        週循環的交易視窗，確保在該週「買入日」抱到「賣出日」
        """
        dow = position.index.dayofweek
        buy = self.buy_weekday
        sell = self.sell_weekday

        if buy == sell:
            return position
        elif buy < sell:
            mask = (dow >= buy) & (dow < sell)
        else:
            mask = (dow >= buy) | (dow < sell)

        position = position.loc[mask].reindex(position.index, fill_value=False)
        return position

if __name__ == '__main__':
    strategy = RogerTWStrategyWeekly()
    strategy.run_strategy()
    report = strategy.get_report()