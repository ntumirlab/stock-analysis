from strategy_class.roger_tw_strategy_base import RogerTWStrategyBase

class RogerTWStrategyWeekly(RogerTWStrategyBase):
    def __init__(self, config_path="config.yaml", override_params=None):
        super().__init__(task_name="weekly", config_path=config_path, override_params=override_params)

if __name__ == '__main__':
    strategy = RogerTWStrategyWeekly()
    strategy.run_strategy()
    report = strategy.get_report()