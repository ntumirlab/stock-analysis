import sys
import os
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.reservation_handler import ShioajiReservationHandler


def _make_handler():
    account = MagicMock()
    account.api.stock_account = MagicMock()
    return ShioajiReservationHandler(account)


def _stock_info_buy():
    return {'stock_id': '2492', 'quantity': 0.003, 'action': '買入', 'total_amount': 1300.20}


def _stock_info_sell():
    return {'stock_id': '2492', 'quantity': -0.004, 'action': '賣出', 'total_amount': -1497.60}


def _make_resp(status: bool, info: str = ''):
    resp = MagicMock()
    resp.status = status
    resp.info = info
    return resp


class TestShioajiReservationHandlerBuy(unittest.TestCase):

    def test_buy_success(self):
        handler = _make_handler()
        handler.account.api.reserve_earmarking.return_value = _make_resp(True, 'ok')
        # Should not raise
        handler._reserve_for_buy(_stock_info_buy())

    def test_buy_failure_raises(self):
        """API 回傳 status=False 時應 raise，不能誤報成功。"""
        handler = _make_handler()
        handler.account.api.reserve_earmarking.return_value = _make_resp(False, '非交易服務時間')
        with self.assertRaises(RuntimeError) as ctx:
            handler._reserve_for_buy(_stock_info_buy())
        self.assertIn('非交易服務時間', str(ctx.exception))
        self.assertIn('2492', str(ctx.exception))

    def test_buy_failure_is_caught_by_handle_alerting_stocks(self):
        """handle_alerting_stocks 的 try/except 應捕捉失敗，繼續處理下一筆。"""
        handler = _make_handler()
        handler.account.api.reserve_earmarking.return_value = _make_resp(False, '非交易服務時間')
        # Should not propagate — handle_alerting_stocks catches per-stock errors
        handler.handle_alerting_stocks([_stock_info_buy()])


class TestShioajiReservationHandlerSell(unittest.TestCase):

    def test_sell_success(self):
        handler = _make_handler()
        handler.account.api.reserve_stock.return_value = _make_resp(True, 'ok')
        handler._reserve_for_sell(_stock_info_sell())

    def test_sell_failure_raises(self):
        """賣出圈存失敗時應 raise，不能誤報成功。"""
        handler = _make_handler()
        handler.account.api.reserve_stock.return_value = _make_resp(False, '餘額不足')
        with self.assertRaises(RuntimeError) as ctx:
            handler._reserve_for_sell(_stock_info_sell())
        self.assertIn('餘額不足', str(ctx.exception))
        self.assertIn('2492', str(ctx.exception))

    def test_sell_uses_abs_quantity_for_shares(self):
        """負數 quantity 經 abs() 後 shares 應為正值且非零。"""
        handler = _make_handler()
        handler.account.api.reserve_stock.return_value = _make_resp(True)
        handler._reserve_for_sell(_stock_info_sell())
        call_args = handler.account.api.reserve_stock.call_args
        shares_arg = call_args[0][2]  # positional arg: (stock_account, contract, shares)
        self.assertGreater(shares_arg, 0)


if __name__ == '__main__':
    unittest.main()
