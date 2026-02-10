import datetime
import logging
from finlab.online.base_account import Account
from dao.balance_dao import BalanceDAO

logger = logging.getLogger(__name__)


class BalanceFetcherBase:
    def __init__(self, user_name, broker_name, account_obj: Account, fetch_timestamp=None):
        self.user_name = user_name
        self.broker_name = broker_name
        self.account = account_obj
        self.fetch_timestamp = fetch_timestamp
        self.balance_dao = BalanceDAO()

    def fetch_and_save(self):
        raw_data = self.fetch_raw_data()
        processed_data = self.process_data(raw_data)
        self.save_to_db(processed_data)
        return processed_data

    def fetch_raw_data(self):
        """
        取得帳戶餘額相關資料

        Returns:
            dict: 包含帳戶餘額相關的原始資料
        """
        bank_balance = self.account.get_cash()
        settlements = self.account.get_settlement()
        total_assets = self.account.get_total_balance()
        adjusted_bank_balance = bank_balance + settlements
        market_value = total_assets - adjusted_bank_balance

        raw_data = {
            'bank_balance': bank_balance,
            'settlements': settlements,
            'total_assets': total_assets,
            'adjusted_bank_balance': adjusted_bank_balance,
            'market_value': market_value,
        }

        logger.debug(f"Fetched raw balance data: {raw_data}")
        return raw_data

    def process_data(self, raw_data):
        processed_data = {
            'bank_balance': raw_data['bank_balance'],
            'settlements': raw_data['settlements'],
            'adjusted_bank_balance': raw_data['adjusted_bank_balance'],
            'market_value': raw_data['market_value'],
            'total_assets': raw_data['total_assets'],
        }

        logger.debug(f"Processed balance data: {processed_data}")
        return processed_data

    def save_to_db(self, processed_data):
        account_name = f"{self.user_name}_{self.broker_name}"
        account_id = self.balance_dao.get_account_id(
            account_name, broker_name=self.broker_name, user_name=self.user_name
        )
        self.balance_dao.insert_balance(account_id, balance_data=processed_data, fetch_timestamp=self.fetch_timestamp)
        logger.info(f"Saved balance data for account {account_name} (ID: {account_id})")
