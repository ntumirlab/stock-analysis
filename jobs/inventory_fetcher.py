import datetime
import logging
from utils.authentication import Authenticator
from dao.inventory_dao import InventoryDAO
from dao.account_dao import AccountDAO
from finlab.online.base_account import Account
from finlab.online.fugle_account import FugleAccount
from finlab.online.sinopac_account import SinopacAccount
from utils.stock_mapper import StockMapper
import shioaji as sj

logger = logging.getLogger(__name__)


class InventoryFetcherBase:
    def __init__(self, user_name, broker_name, account_obj: Account, fetch_timestamp=None):
        self.user_name = user_name
        self.broker_name = broker_name
        self.account = account_obj
        self.inventory_dao = InventoryDAO()
        self.account_dao = AccountDAO()
        self.stock_mapper = StockMapper()
        self.fetch_timestamp = fetch_timestamp

    def fetch_and_save(self):
        raw_data = self.fetch_raw_data()
        processed_data = self.process_data(raw_data)
        self.save_to_db(processed_data)
        return processed_data

    def fetch_raw_data(self):
        raise NotImplementedError("Subclasses must implement fetch_raw_data")

    def process_data(self, raw_data):
        raise NotImplementedError("Subclasses must implement process_data")

    def save_to_db(self, processed_data):
        account_name = f"{self.user_name}_{self.broker_name}"
        account_id = self.account_dao.get_account_id(
            account_name, broker_name=self.broker_name, user_name=self.user_name
        )
        self.inventory_dao.insert_inventory_data(
            account_id, inventory_data=processed_data, fetch_timestamp=self.fetch_timestamp
        )
        logger.info(f"Saved inventory data for {self.user_name} using {self.get_broker_name()}")


class FugleInventoryFetcher(InventoryFetcherBase):
    def __init__(self, user_name: str, broker_name: str, account_obj: FugleAccount, fetch_timestamp=None):
        super().__init__(user_name, broker_name, account_obj, fetch_timestamp)

    def fetch_raw_data(self):
        logger.info("Fetching data from Fugle API")
        raw_inventory = self.account.sdk.get_inventories()
        return raw_inventory

    def process_data(self, raw_data):
        """
        處理 Fugle API 回傳的庫存資料

        格式轉換對應:
        fugle --> database
        stk_no --> stock_id
        stk_na --> stock_name
        cost_qty (股) --> quantity (張，除以1000)
        price_mkt --> last_price
        make_a_sum --> pnl

        Args:
            raw_data (list): Fugle API 回傳的原始庫存資料

        Returns:
            list[dict]: 處理後的庫存資料
        """
        processed_items = []

        for item in raw_data:
            shares = float(item.get('cost_qty', 0))
            lots = shares / 1000
            processed_item = {
                'stock_id': item.get('stk_no'),
                'stock_name': item.get('stk_na'),
                'quantity': lots,
                'last_price': float(item.get('price_mkt', 0)),
                'pnl': float(item.get('make_a_sum', 0)),
                'raw_data': item,  # 保留原始資料供參考
            }
            processed_items.append(processed_item)

        logger.info(f"Processed {len(processed_items)} inventory items from Fugle")
        return processed_items

    def get_broker_name(self):
        return "fugle"


class ShioajiInventoryFetcher(InventoryFetcherBase):
    def __init__(self, user_name: str, broker_name: str, account_obj: SinopacAccount, fetch_timestamp=None):
        super().__init__(user_name, broker_name, account_obj, fetch_timestamp)

    def fetch_raw_data(self):
        logger.info("Fetching data from Shioaji API")
        raw_inventory = self.account.api.list_positions(self.account.api.stock_account, unit=sj.constant.Unit.Share)
        return raw_inventory

    def process_data(self, raw_data):
        """
        處理 Shioaji API 回傳的庫存資料

        格式轉換對應:
        Shioaji --> database
        code --> stock_id
        (無) --> stock_name (使用 StockMapper 查詢)
        quantity (股) --> quantity (張，除以1000)
        last_price --> last_price
        pnl --> pnl

        Args:
            raw_data (list): Shioaji API 回傳的原始庫存資料

        Returns:
            list[dict]: 處理後的庫存資料
        """
        processed_items = []

        for position in raw_data:
            # 由於 Shioaji 的 API 回傳是物件，需要先轉成字典
            position_dict = position.__dict__

            # 使用 StockMapper 查詢股票名稱
            stock_id = position_dict.get('code')
            stock_name = self.stock_mapper.map(stock_id)
            shares = float(position_dict.get('quantity', 0))
            lots = shares / 1000

            processed_item = {
                'stock_id': stock_id,
                'stock_name': stock_name,
                'quantity': lots,
                'last_price': float(position_dict.get('last_price', 0)),
                'pnl': float(position_dict.get('pnl', 0)),
                'raw_data': position_dict,  # 保留原始資料供參考
            }
            processed_items.append(processed_item)

        logger.info(f"Processed {len(processed_items)} inventory items from Shioaji")
        return processed_items

    def get_broker_name(self):
        return "shioaji"


class InventoryFetcher:
    """用於建立合適的抓取器實例的工廠類"""

    @staticmethod
    def create(user_name, broker_name, account, fetch_timestamp=None):
        if broker_name == "fugle":
            return FugleInventoryFetcher(user_name, broker_name, account, fetch_timestamp)
        elif broker_name == "shioaji":
            return ShioajiInventoryFetcher(user_name, broker_name, account, fetch_timestamp)
        else:
            raise ValueError(f"Unsupported broker: {broker_name}")
