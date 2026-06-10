from strategy_class.golden_ai_tw_strategy_monthly import GoldenAITWStrategyMonthly


class GoldenAITWStrategyWeekly4W(GoldenAITWStrategyMonthly):
    """行為與月策略（GoldenAITWStrategyMonthly）完全相同：
    Week1~4 loop、跑所有 ranks 子集組合、multi-worker、每週存一筆 DB record。
    唯一差別是推薦清單來源吃 weekly（recommendation_frequency='weekly'），
    DB / 報告檔名的 strategy 欄位為 'weekly_4w'（由 task_name 帶入）。
    """

    def __init__(self, config_path="config.yaml", override_params=None):
        super(GoldenAITWStrategyMonthly, self).__init__(
            task_name="weekly_4w", config_path=config_path, override_params=override_params
        )
        # 推薦清單共用 weekly（與 GoldenAITWStrategyWeekly 同來源）
        self.recommendation_frequency = 'weekly'


if __name__ == '__main__':
    strategy = GoldenAITWStrategyWeekly4W()
    strategy.run_strategy()
    report = strategy.get_report()
