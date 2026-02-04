from strategy_class.roger_tw_strategy_base import RogerTWStrategyBase

class RogerTWStrategyMonthly(RogerTWStrategyBase):
    def __init__(self, config_path="config.yaml"):
        super().__init__(task_name="monthly", max_stocks=5, buy_weekday=1, sell_weekday=5, config_path=config_path)

if __name__ == '__main__':
    strategy = RogerTWStrategyMonthly()
    strategy.run_strategy()
    report = strategy.get_report()