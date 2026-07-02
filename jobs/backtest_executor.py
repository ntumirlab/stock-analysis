import argparse
import importlib
import logging
import os
import traceback
from utils.config_loader import ConfigLoader
from utils.logger_manager import LoggerManager
from utils.notifier import create_notification_manager
from datetime import datetime
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# 可回測的策略：名稱 → (module, class)。lazy import 避免一次載入所有策略模組。
BACKTEST_STRATEGIES = {
    'TibetanMastiffTWStrategy': ('strategy_class.tibetanmastiff_tw_strategy', 'TibetanMastiffTWStrategy'),
    'PeterWuStrategy': ('strategy_class.peterwu_tw_strategy', 'PeterWuStrategy'),
    'AlanTWStrategyACE': ('strategy_class.alan_tw_strategy_ACE', 'AlanTWStrategyACE'),
    'AlanTWStrategyFG': ('strategy_class.alan_tw_strategy_FG', 'AlanTWStrategyFG'),
    'AlanTWStrategyEFG': ('strategy_class.alan_tw_strategy_EFG', 'AlanTWStrategyEFG'),
    'AlanTWStrategyEFGObserve': ('strategy_class.alan_tw_strategy_EFG_observe', 'AlanTWStrategyEFGObserve'),
    'AlanTWStrategyEFGObserveDI21Bias05': ('strategy_class.alan_tw_strategy_EFG_observe_di21_bias05', 'AlanTWStrategyEFGObserveDI21Bias05'),
    'AlanTWStrategyEFGObserveDI21Bias35MACDBias25': ('strategy_class.alan_tw_strategy_EFG_observe_di21_bias35_macd_bias25', 'AlanTWStrategyEFGObserveDI21Bias35MACDBias25'),
    'AlanTWStrategyNotStart': ('strategy_class.alan_tw_strategy_not_start', 'AlanTWStrategyNotStart'),
    'AlanTWStrategyNotStartA': ('strategy_class.alan_tw_strategy_not_start_A', 'AlanTWStrategyNotStartA'),
    'AlanTWStrategyNotStartB': ('strategy_class.alan_tw_strategy_not_start_B', 'AlanTWStrategyNotStartB'),
    'RAndDManagementStrategy': ('strategy_class.r_and_d_management_strategy', 'RAndDManagementStrategy'),
    'GoldenAITWStrategyWeekly': ('strategy_class.golden_ai_tw_strategy_weekly', 'GoldenAITWStrategyWeekly'),
    'GoldenAITWStrategyMonthly': ('strategy_class.golden_ai_tw_strategy_monthly', 'GoldenAITWStrategyMonthly'),
    'GoldenAITWStrategyWeekly4W': ('strategy_class.golden_ai_tw_strategy_weekly_4w', 'GoldenAITWStrategyWeekly4W'),
    'OscarAndOrStrategy': ('strategy_class.oscar.oscar_strategy_andor', 'OscarAndOrStrategy'),
    'OscarCompositeStrategy': ('strategy_class', 'OscarCompositeStrategy'),
    '2560AndOrTWStrategy': ('strategy_class', '_2560AndOrTWStrategy'),
}

class BacktestExecutor:
    def __init__(self, strategy_class_name, config_path="config.yaml", base_log_directory="logs"):
        self.strategy_class_name = strategy_class_name
        self.backtest_timestamp = datetime.now(ZoneInfo("Asia/Taipei"))
        self.logger_manager = LoggerManager(
            base_log_directory=base_log_directory,
            current_datetime=self.backtest_timestamp,
        )
        self.config_loader = ConfigLoader(config_path)
        self.config_loader.load_global_env_vars()
        self.log_file = self.logger_manager.setup_logging()

    def run_strategy_and_save(self):
        strategy = self.load_strategy()
        is_golden_ai = self.strategy_class_name in ('GoldenAITWStrategyWeekly', 'GoldenAITWStrategyMonthly', 'GoldenAITWStrategyWeekly4W')
        if is_golden_ai:
            strategy.run_strategy()
        else:
            report = strategy.run_strategy()
            self.save_finlab_report(report)


    def save_finlab_report(self, report, base_directory="assets/"):
        subdirectory = self.strategy_class_name
        report_directory = os.path.join(base_directory, subdirectory)
        if not os.path.exists(report_directory):
            os.makedirs(report_directory)
        datetime_str = self.backtest_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        save_report_path = os.path.join(report_directory, f"{datetime_str}.html")
        report.display(save_report_path=save_report_path)
        


    def load_strategy(self):
        entry = BACKTEST_STRATEGIES.get(self.strategy_class_name)
        if entry is None:
            raise ValueError(f"Unknown strategy class: {self.strategy_class_name}")
        module_path, class_name = entry
        strategy_class = getattr(importlib.import_module(module_path), class_name)
        return strategy_class()

if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root_dir)

    parser = argparse.ArgumentParser(description="Run BacktestExecutor")
    parser.add_argument("--strategy_class_name", required=True, help="strategy_class_name (e.g., TibetanMastiffTWStrategy)")

    args = parser.parse_args()
    logger.info(f"args: {args}")

    # 初始化通知管理器
    config_loader = ConfigLoader(os.path.join(root_dir, "config.yaml"))
    notifier = create_notification_manager(config_loader.config.get('notification', {}), logger)

    try:
        backtest_executor = BacktestExecutor(strategy_class_name=args.strategy_class_name)
        backtest_executor.run_strategy_and_save()
    except Exception as e:
        logger.exception(e)

        # 發送錯誤通知
        notifier.send_error(
            task_name="回測執行",
            error_message=str(e),
            error_traceback=traceback.format_exc()
        )

    # python -m jobs.backtest_executor --strategy_class_name AlanTwStrategy1
