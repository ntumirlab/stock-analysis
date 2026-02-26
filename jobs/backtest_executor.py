import argparse
import logging
import os
import traceback
from utils.config_loader import ConfigLoader
from utils.logger_manager import LoggerManager
from utils.notifier import create_notification_manager
from datetime import datetime
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


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
        if self.strategy_class_name == 'TibetanMastiffTWStrategy':
            from strategy_class.tibetanmastiff_tw_strategy import TibetanMastiffTWStrategy as strategy_class
        elif self.strategy_class_name == 'PeterWuStrategy':
            from strategy_class.peterwu_tw_strategy import PeterWuStrategy as strategy_class
        elif self.strategy_class_name == 'AlanTWStrategyACE':
            from strategy_class.alan_tw_strategy_ACE import AlanTWStrategyACE as strategy_class
        elif self.strategy_class_name == 'AlanTWStrategyNotStart':
            from strategy_class.alan_tw_strategy_not_start import AlanTWStrategyNotStart as strategy_class
        elif self.strategy_class_name == 'AlanTWStrategyNotStartA':
            from strategy_class.alan_tw_strategy_not_start_A import AlanTWStrategyNotStartA as strategy_class
        elif self.strategy_class_name == 'AlanTWStrategyNotStartB':
            from strategy_class.alan_tw_strategy_not_start_B import AlanTWStrategyNotStartB as strategy_class
        elif self.strategy_class_name == 'RAndDManagementStrategy':
            from strategy_class.r_and_d_management_strategy import RAndDManagementStrategy as strategy_class
        elif self.strategy_class_name == 'RogerTWStrategyWeekly':
            from strategy_class.roger_tw_strategy_weekly import RogerTWStrategyWeekly as strategy_class
        elif self.strategy_class_name == 'RogerTWStrategyMonthly':
            from strategy_class.roger_tw_strategy_monthly import RogerTWStrategyMonthly as strategy_class
        elif self.strategy_class_name == 'OscarTWStrategy':
            from strategy_class.oscar_tw_strategy import OscarTWStrategy as strategy_class
        else:
            raise ValueError(f"Unknown strategy class: {self.strategy_class_name}")
        return strategy_class()


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root_dir)

    parser = argparse.ArgumentParser(description="Run BacktestExecutor")
    parser.add_argument(
        "--strategy_class_name", required=True, help="strategy_class_name (e.g., TibetanMastiffTWStrategy)"
    )

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
        notifier.send_error(task_name="回測執行", error_message=str(e), error_traceback=traceback.format_exc())

    # python -m jobs.backtest_executor --strategy_class_name AlanTwStrategy1
