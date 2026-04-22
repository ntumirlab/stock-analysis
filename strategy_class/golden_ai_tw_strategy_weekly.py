from strategy_class.golden_ai_tw_strategy_base import GoldenAITWStrategyBase

class GoldenAITWStrategyWeekly(GoldenAITWStrategyBase):
    def __init__(self, config_path="config.yaml", override_params=None):
        super().__init__(task_name="weekly", config_path=config_path, override_params=override_params)

if __name__ == '__main__':
    strategy = GoldenAITWStrategyWeekly()
    strategy.run_strategy()
    report = strategy.get_report()
