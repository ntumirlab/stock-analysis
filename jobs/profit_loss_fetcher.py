import datetime
import logging
from zoneinfo import ZoneInfo

from finlab.online.base_account import Account
from finlab.online.sinopac_account import SinopacAccount
import shioaji as sj

from core.profit_loss_range import normalize_trade_date, resolve_fetch_range
from dao.account_dao import AccountDAO
from dao.profit_loss_dao import ProfitLossDAO
from utils.stock_mapper import StockMapper

logger = logging.getLogger(__name__)


class ProfitLossFetcherBase:
    def __init__(self, user_name, broker_name, account_obj: Account, fetch_timestamp=None):
        self.user_name = user_name
        self.broker_name = broker_name
        self.account = account_obj
        self.fetch_timestamp = fetch_timestamp
        self.profit_loss_dao = ProfitLossDAO()
        self.account_dao = AccountDAO()
        self.stock_mapper = StockMapper()

    def fetch_and_save(self, begin_date=None, end_date=None, save=True):
        """抓取並寫入已實現損益。

        Args:
            begin_date (datetime.date, optional): 起始日，未給則自 DB 現況推算
            end_date (datetime.date, optional): 結束日，未給則為今天
            save (bool): False 時只抓取與轉換、不寫入 DB。用於比對不同 unit
                的涵蓋範圍——兩種 unit 抓到的是同一批交易，唯一鍵相同，
                若都寫入則第二次會被索引擋掉、比不出差異還污染正式資料

        Returns:
            list[dict]: 處理後的已實現損益資料
        """
        account_id = self._get_account_id()
        begin_date, end_date = self.resolve_date_range(account_id, begin_date, end_date)

        raw_data = self.fetch_raw_data(begin_date, end_date)
        processed_data = self.process_data(raw_data)

        if save:
            self.profit_loss_dao.insert_profit_loss(
                account_id,
                processed_data,
                fetch_timestamp=self.fetch_timestamp,
            )
        else:
            logger.info("Dry run: %d records fetched, nothing written", len(processed_data))

        return processed_data

    def _get_account_id(self):
        account_name = f"{self.user_name}_{self.broker_name}"
        return self.account_dao.get_account_id(
            account_name, broker_name=self.broker_name, user_name=self.user_name
        )

    def resolve_date_range(self, account_id, begin_date=None, end_date=None):
        """決定這次要抓的區間（數學在 core.profit_loss_range，這裡只餵 DB 現況）。

        Args:
            account_id (int): 帳戶ID
            begin_date (datetime.date, optional): 明確指定的起始日（backfill 用）
            end_date (datetime.date, optional): 明確指定的結束日

        Returns:
            tuple[datetime.date, datetime.date]
        """
        today = self.fetch_timestamp.date() if self.fetch_timestamp else \
            datetime.datetime.now(ZoneInfo("Asia/Taipei")).date()

        begin_date, end_date = resolve_fetch_range(
            today,
            latest_trade_date=self.profit_loss_dao.get_latest_trade_date(account_id),
            begin_date=begin_date,
            end_date=end_date,
        )

        logger.info(f"Profit/loss query range: {begin_date} ~ {end_date}")
        return begin_date, end_date

    def fetch_raw_data(self, begin_date, end_date):
        raise NotImplementedError("Subclasses must implement fetch_raw_data")

    def process_data(self, raw_data):
        raise NotImplementedError("Subclasses must implement process_data")


class ShioajiProfitLossFetcher(ProfitLossFetcherBase):

    def __init__(self, user_name: str, broker_name: str, account_obj: SinopacAccount,
                 fetch_timestamp=None, unit=None):
        super().__init__(user_name, broker_name, account_obj, fetch_timestamp)
        # 實盤走零股，與 inventory_fetcher 的 list_positions 取用同一種 unit；
        # Common/Share 是否會改變回傳的涵蓋範圍需以實際帳戶核對（見 --unit 參數）
        self.unit = unit or sj.constant.Unit.Share
        # 券商回傳的數量單位隨 unit 而異（Share＝股、Common＝張），入庫一律轉成張。
        # 寫死除以 1000 會讓 Common 的數量差一千倍，故除數綁定 unit
        self.quantity_divisor = 1000 if self.unit == sj.constant.Unit.Share else 1

    def fetch_raw_data(self, begin_date, end_date):
        logger.info("Fetching realized profit/loss from Shioaji API")
        return self.account.api.list_profit_loss(
            self.account.api.stock_account,
            begin_date=begin_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            unit=self.unit,
        )

    def process_data(self, raw_data):
        """處理 Shioaji list_profit_loss 回傳的已實現損益。

        格式轉換對應:
        Shioaji --> database
        date --> trade_date
        code --> stock_id
        (無) --> stock_name（使用 StockMapper 查詢）
        quantity (Share 模式為股 / Common 模式為張) --> quantity (一律轉成張，
            與 inventory_history 同慣例；換算除數見 self.quantity_divisor)
        price --> price
        pnl --> pnl
        pr_ratio --> pr_ratio (%)

        Args:
            raw_data (list): Shioaji API 回傳的 StockProfitLoss 物件

        Returns:
            list[dict]: 處理後的已實現損益資料
        """
        processed_items = []

        for record in raw_data:
            record_dict = record.__dict__

            stock_id = record_dict.get('code')
            raw_quantity = float(record_dict.get('quantity', 0) or 0)
            # cond 是 enum，str() 會存成 "StockOrderCond.Cash"；取 .value 才是 "Cash"
            cond = record_dict.get('cond')
            cond_value = getattr(cond, 'value', cond)

            processed_items.append({
                'trade_date': normalize_trade_date(record_dict.get('date')),
                'stock_id': stock_id,
                'stock_name': self.stock_mapper.map(stock_id),
                'quantity': raw_quantity / self.quantity_divisor,
                'price': float(record_dict.get('price', 0) or 0),
                'pnl': float(record_dict.get('pnl', 0) or 0),
                'pr_ratio': float(record_dict.get('pr_ratio', 0) or 0),
                'cond': '' if cond_value is None else str(cond_value),
                'dseq': str(record_dict.get('dseq', '')),
                'seqno': str(record_dict.get('seqno', '')),
                'raw_data': record_dict,
            })

        logger.info(f"Processed {len(processed_items)} realized profit/loss records from Shioaji")
        return processed_items

    def get_broker_name(self):
        return "shioaji"


class ProfitLossFetcher:
    """用於建立合適的抓取器實例的工廠類"""

    @staticmethod
    def create(user_name, broker_name, account, fetch_timestamp=None, unit=None):
        """建立抓取器；不支援的券商回 None（讓排程略過而非中斷其他帳務抓取）。"""
        if broker_name == "shioaji":
            return ShioajiProfitLossFetcher(
                user_name, broker_name, account, fetch_timestamp, unit=unit
            )
        logger.warning(
            f"Realized profit/loss fetching is not implemented for broker '{broker_name}', skipped"
        )
        return None


if __name__ == "__main__":
    import argparse
    import os
    import traceback

    from utils.authentication import Authenticator
    from utils.config_loader import ConfigLoader
    from utils.logger_manager import LoggerManager

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root_dir)

    parser = argparse.ArgumentParser(
        description="Fetch realized profit/loss from broker API (also used for one-off backfill)"
    )
    parser.add_argument("--user_name", required=True, help="User name (e.g., junting)")
    parser.add_argument("--broker_name", required=True, help="Broker name (e.g., shioaji)")
    parser.add_argument("--begin_date", help="Backfill start date, YYYY-MM-DD")
    parser.add_argument("--end_date", help="Backfill end date, YYYY-MM-DD")
    parser.add_argument("--unit", choices=["share", "common"], default="share",
                        help="Shioaji query unit; use to compare odd-lot vs board-lot coverage")
    parser.add_argument("--dry_run", action="store_true",
                        help="Fetch and print without writing to the database; "
                             "use this when comparing units so neither result is persisted")

    args = parser.parse_args()

    def _parse_date(value):
        return datetime.datetime.strptime(value, "%Y-%m-%d").date() if value else None

    fetch_timestamp = datetime.datetime.now(ZoneInfo("Asia/Taipei"))
    logger_manager = LoggerManager(
        base_log_directory=os.path.join(root_dir, "logs"),
        current_datetime=fetch_timestamp,
    )
    logger_manager.setup_logging()
    logger.info(f"args: {args}")

    try:
        config_loader = ConfigLoader(os.path.join(root_dir, "config.yaml"))
        config_loader.load_global_env_vars()
        config_loader.load_user_config(args.user_name, args.broker_name)

        auth = Authenticator(config_loader)
        auth.login_finlab()
        account = auth.login_broker(args.broker_name)

        unit = sj.constant.Unit.Share if args.unit == "share" else sj.constant.Unit.Common
        fetcher = ProfitLossFetcher.create(
            args.user_name, args.broker_name, account, fetch_timestamp, unit=unit
        )
        if fetcher is None:
            raise SystemExit(f"Unsupported broker: {args.broker_name}")

        records = fetcher.fetch_and_save(
            begin_date=_parse_date(args.begin_date),
            end_date=_parse_date(args.end_date),
            save=not args.dry_run,
        )

        # 逐筆列出，供兩種 unit 的結果並排比對：raw 是券商原始數量，
        # qty(張) 是換算後入庫的值
        print(f"\n=== unit={args.unit} | {len(records)} records "
              f"| {'DRY RUN (not saved)' if args.dry_run else 'saved'} ===")
        print(f"{'date':<12}{'code':<8}{'raw':>10}{'qty(張)':>12}"
              f"{'price':>10}{'pnl':>12}{'pr_ratio':>10}")
        for record in sorted(records, key=lambda r: (r['trade_date'] or '', r['stock_id'] or '')):
            raw_quantity = (record.get('raw_data') or {}).get('quantity')
            print(f"{record['trade_date'] or '?':<12}{record['stock_id'] or '?':<8}"
                  f"{str(raw_quantity):>10}{record['quantity']:>12.3f}"
                  f"{record['price']:>10.2f}{record['pnl']:>12.2f}"
                  f"{record['pr_ratio']:>9.2f}%")

        total_pnl = sum(r['pnl'] for r in records)
        codes = sorted({r['stock_id'] for r in records})
        print(f"\ntotal pnl = {total_pnl:.2f} | codes = {codes}")
        logger.info(f"Fetched {len(records)} records, total realized pnl = {total_pnl:.2f}")
    except Exception as e:
        logger.exception(e)
        logger.error(traceback.format_exc())

    # python -m jobs.profit_loss_fetcher --user_name kiri --broker_name shioaji
    # python -m jobs.profit_loss_fetcher --user_name junting --broker_name shioaji \
    #     --begin_date 2025-05-01 --end_date 2026-04-30
