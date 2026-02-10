import argparse
import datetime
import logging
import os
import traceback
from zoneinfo import ZoneInfo
from utils.logger_manager import LoggerManager
from utils.authentication import Authenticator
from utils.config_loader import ConfigLoader
from utils.reservation_handler import ReservationHandlerFactory
from utils.finlab_patcher import apply_finlab_patches  # 自動修補 finlab
from utils.notifier import create_notification_manager
from finlab.portfolio import Portfolio, PortfolioSyncManager
from dao import OrderDAO, AccountDAO
from utils.stock_mapper import StockMapper

# 在導入 finlab 後立即應用補丁
apply_finlab_patches()

logger = logging.getLogger(__name__)


class OrderExecutor:
    def __init__(
        self, user_name, market_order, broker_name, view_only, config_path="config.yaml", base_log_directory="logs"
    ):
        self.user_name = user_name
        self.market_order = market_order
        self.broker_name = broker_name
        self.view_only = view_only

        self.config_loader = ConfigLoader(config_path)
        self.config_loader.load_global_env_vars()
        self.config_loader.load_user_config(user_name, broker_name)

        self.auth = Authenticator(self.config_loader)
        self.auth.login_finlab()
        self.account = self.auth.login_broker(broker_name)

        self.order_timestamp = datetime.datetime.now(ZoneInfo("Asia/Taipei"))
        self.logger_manager = LoggerManager(
            base_log_directory=base_log_directory, current_datetime=self.order_timestamp
        )
        self.log_file = self.logger_manager.setup_logging()
        logger.info(f"user_name: {self.user_name}, broker_name: {self.broker_name}")

        self.order_dao = OrderDAO()
        self.account_dao = AccountDAO()
        self.stock_mapper = StockMapper()

    def run_strategy_and_sync(self):
        strategy_class_name = self.config_loader.get_user_constant("strategy_class_name")
        strategy = self.load_strategy(strategy_class_name)
        report = strategy.run_strategy()

        port = Portfolio(
            {
                'strategy': (report, 1.0),
            }
        )
        pm_name = f"{self.user_name}_{self.broker_name}"
        try:
            pm = PortfolioSyncManager.from_local(name=pm_name)
        except FileNotFoundError:
            pm = PortfolioSyncManager()

        total_balance = self.account.get_total_balance()
        logger.info(f"Total balance: {total_balance}")
        if total_balance <= 0:
            raise ValueError(
                f"{self.user_name}'s total balance is not positive. Please check your {self.broker_name} account balance."
            )

        safety_weight = self.config_loader.get_user_constant("rebalance_safety_weight")
        pm.update(
            port,
            total_balance=total_balance,
            rebalance_safety_weight=safety_weight,
            odd_lot=True,
            force_override_difference=True,
            smooth_transition=False,
        )
        pm.to_local(name=pm_name)

        # 創建 order_executor（不實際下單，只用來顯示警示股）
        order_executor = pm.create_order_executor(self.account)
        order_executor.show_alerting_stocks()

        # 處理警示股圈存
        if not self.view_only:
            self._handle_alerting_stocks_reservation()

        # 執行同步下單（會打印完整下單資訊）
        pm.sync(self.account, market_order=self.market_order, view_only=self.view_only)
        logger.info("Portfolio synced")

        order_logs = self.logger_manager.extract_order_logs(self.log_file)

        if not order_logs:
            logger.warning("Today not have any order")
            return
        else:
            for order in order_logs:
                order['stock_name'] = self.stock_mapper.map(order['stock_id'])

            account_name = self.user_name + "_" + self.broker_name
            account_id = self.account_dao.get_account_id(
                account_name, broker_name=self.broker_name, user_name=self.user_name
            )
            self.order_dao.insert_order_logs(order_logs, account_id, self.order_timestamp, view_only=self.view_only)

    def _handle_alerting_stocks_reservation(self):
        """處理警示股圈存（使用策略模式支援多券商）"""
        # 從 log 提取警示股資訊
        alerting_stocks = self.logger_manager.extract_alerting_stocks(self.log_file)

        # 建立對應券商的圈存處理器
        handler = ReservationHandlerFactory.create(self.broker_name, self.account)

        # 執行圈存
        handler.handle_alerting_stocks(alerting_stocks)

    def load_strategy(self, strategy_class_name):
        if strategy_class_name == 'TibetanMastiffTWStrategy':
            from strategy_class.tibetanmastiff_tw_strategy import TibetanMastiffTWStrategy as strategy_class
        elif strategy_class_name == 'PeterWuStrategy':
            from strategy_class.peterwu_tw_strategy import PeterWuStrategy as strategy_class
        elif strategy_class_name == 'AlanTWStrategyACE':
            from strategy_class.alan_tw_strategy_ACE import AlanTWStrategyACE as strategy_class
        elif strategy_class_name == 'RAndDManagementStrategy':
            from strategy_class.r_and_d_management_strategy import RAndDManagementStrategy as strategy_class
        else:
            raise ValueError(f"Unknown strategy class: {strategy_class_name}")
        return strategy_class()


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root_dir)

    parser = argparse.ArgumentParser(description="Run OrderExecutor")
    parser.add_argument("--user_name", required=True, help="User name (e.g., junting)")
    parser.add_argument("--broker_name", required=True, help="Broker name (e.g., fugle)")
    parser.add_argument("--view_only", action='store_true', help="Run in view-only mode")

    args = parser.parse_args()
    logger.info(f"args: {args}")

    # 初始化通知管理器
    config_loader = ConfigLoader(os.path.join(root_dir, "config.yaml"))
    notifier = create_notification_manager(config_loader.config.get('notification', {}), logger)

    try:
        order_executor = OrderExecutor(
            user_name=args.user_name,
            market_order=True,
            broker_name=args.broker_name,
            view_only=args.view_only,
            config_path=os.path.join(root_dir, "config.yaml"),
            base_log_directory=os.path.join(root_dir, "logs"),
        )
        order_executor.run_strategy_and_sync()
    except Exception as e:
        logger.exception(e)

        # 發送錯誤通知
        notifier.send_error(
            task_name="早盤下單",
            error_message=str(e),
            user_name=args.user_name,
            broker_name=args.broker_name,
            error_traceback=traceback.format_exc(),
        )

    # python -m jobs.order_executor --user_name junting --broker_name fugle --view_only
    # python -m jobs.order_executor --user_name junting --broker_name shioaji --view_only
    # python -m jobs.order_executor --user_name alan --broker_name shioaji --view_only
