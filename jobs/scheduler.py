import datetime
import logging
import traceback
from zoneinfo import ZoneInfo
from jobs.balance_fetcher import BalanceFetcherBase
from jobs.inventory_fetcher import InventoryFetcher
from utils.authentication import Authenticator
from utils.config_loader import ConfigLoader
from utils.logger_manager import LoggerManager
from utils.notifier import create_notification_manager

logger = logging.getLogger(__name__)


class Scheduler:
    def __init__(self, user_name, broker_name, config_path="config.yaml", base_log_directory="logs"):
        self.user_name = user_name
        self.broker_name = broker_name

        self.config_loader = ConfigLoader(config_path)
        self.config_loader.load_global_env_vars()
        self.config_loader.load_user_config(user_name, broker_name)

        self.auth = Authenticator(self.config_loader)
        self.auth.login_finlab()
        self.account = self.auth.login_broker(broker_name)

        self.fetch_timestamp = datetime.datetime.now(ZoneInfo("Asia/Taipei"))
        self.logger_manager = LoggerManager(
            base_log_directory=base_log_directory,
            current_datetime=self.fetch_timestamp,
        )
        self.log_file = self.logger_manager.setup_logging()
        logger.info(f"user_name: {self.user_name}, broker_name: {self.broker_name}")

    def run(self):
        inventory_fetcher = InventoryFetcher.create(
            self.user_name, self.broker_name, self.account, self.fetch_timestamp
        )
        inventory_data = inventory_fetcher.fetch_and_save()
        balance_fetcher = BalanceFetcherBase(self.user_name, self.broker_name, self.account, self.fetch_timestamp)
        balance_data = balance_fetcher.fetch_and_save()


if __name__ == "__main__":
    import argparse
    import os

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root_dir)

    parser = argparse.ArgumentParser(description="Fetch inventory data from broker API")
    parser.add_argument("--user_name", required=True, help="User name (e.g., junting)")
    parser.add_argument("--broker_name", required=True, help="Broker name (e.g., fugle or shioaji)")

    args = parser.parse_args()
    logger.info(f"args: {args}")

    # 初始化通知管理器
    config_loader = ConfigLoader(os.path.join(root_dir, "config.yaml"))
    notifier = create_notification_manager(config_loader.config.get('notification', {}), logger)

    try:
        scheduler = Scheduler(
            user_name=args.user_name,
            broker_name=args.broker_name,
            config_path=os.path.join(root_dir, "config.yaml"),
            base_log_directory=os.path.join(root_dir, "logs"),
        )
        scheduler.run()
    except Exception as e:
        logger.exception(e)

        # 發送錯誤通知
        notifier.send_error(
            task_name="帳務資料抓取",
            error_message=str(e),
            user_name=args.user_name,
            broker_name=args.broker_name,
            error_traceback=traceback.format_exc(),
        )

    # python -m jobs.scheduler --user_name junting --broker_name fugle
    # python -m jobs.scheduler --user_name junting --broker_name shioaji
