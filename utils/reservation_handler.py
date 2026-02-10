import logging
from finlab.online.base_account import Account
from finlab.online.fugle_account import FugleAccount
from finlab.online.sinopac_account import SinopacAccount
import shioaji as sj

logger = logging.getLogger(__name__)


class ReservationHandlerBase:
    """警示股圈存處理基類"""

    def __init__(self, account: Account):
        self.account = account

    def handle_alerting_stocks(self, alerting_stocks: list):
        """
        處理警示股圈存

        Args:
            alerting_stocks: 警示股列表，每個元素包含:
                - stock_id: 股票代號
                - quantity: 張數
                - action: '買入' 或 '賣出'
                - total_amount: 預估總金額
        """
        if not alerting_stocks:
            logger.info("無警示股，跳過圈存流程")
            return

        logger.info(f"發現 {len(alerting_stocks)} 筆警示股，開始圈存流程...")

        for stock_info in alerting_stocks:
            stock_id = stock_info['stock_id']
            quantity = stock_info['quantity']
            action = stock_info['action']

            try:
                if action == '買入':
                    self._reserve_for_buy(stock_info)
                elif action == '賣出':
                    self._reserve_for_sell(stock_info)
            except Exception as e:
                logger.error(f"圈存失敗 {stock_id} ({action}): {e}")
                # 繼續處理下一筆，不中斷流程

    def _reserve_for_buy(self, stock_info):
        """買入時的圈存（預收款項）"""
        raise NotImplementedError("Subclasses must implement _reserve_for_buy")

    def _reserve_for_sell(self, stock_info):
        """賣出時的圈存（預收股票）"""
        raise NotImplementedError("Subclasses must implement _reserve_for_sell")


class FugleReservationHandler(ReservationHandlerBase):
    """Fugle 券商圈存處理（目前不支援）"""

    def __init__(self, account: FugleAccount):
        super().__init__(account)

    def _reserve_for_buy(self, stock_info):
        """Fugle 目前不支援買入圈存"""
        logger.warning(f"Fugle 券商目前不支援警示股圈存功能，請手動處理: {stock_info['stock_id']}")

    def _reserve_for_sell(self, stock_info):
        """Fugle 目前不支援賣出圈存"""
        logger.warning(f"Fugle 券商目前不支援警示股圈存功能，請手動處理: {stock_info['stock_id']}")


class ShioajiReservationHandler(ReservationHandlerBase):
    """
    Shioaji (永豐) 券商圈存處理

    根據永豐證券客服確認：
    - reserve_earmarking: 預收款項（買入警示股用）
    - reserve_stock: 預收股票（賣出警示股用）
    - 兩者皆與借券無關
    """

    def __init__(self, account: SinopacAccount):
        super().__init__(account)

    def _reserve_for_buy(self, stock_info):
        """買入警示股：圈存資金（預收款項）"""
        stock_id = stock_info['stock_id']
        quantity = stock_info['quantity']

        # 建立合約（login 時 fetch_contract=False）
        contract = sj.contracts.Contract(security_type='STK', code=stock_id, exchange='TSE')

        # 計算股數與預估價格
        shares = int(abs(quantity) * 1000)
        if shares == 0:
            logger.warning(f"跳過 {stock_id}：股數為 0")
            return

        estimated_price = round(abs(stock_info['total_amount']) / shares, 2)

        logger.info(f"圈存資金(買入): {stock_id}, 數量={quantity}張({shares}股), 預估價格={estimated_price:.2f}")

        resp = self.account.api.reserve_earmarking(self.account.api.stock_account, contract, shares, estimated_price)

        logger.info(f"圈存資金成功: {stock_id}, 回應={resp}")

    def _reserve_for_sell(self, stock_info):
        """賣出警示股：圈存股票（預收股票）"""
        stock_id = stock_info['stock_id']
        quantity = stock_info['quantity']

        # 建立合約（login 時 fetch_contract=False）
        contract = sj.contracts.Contract(security_type='STK', code=stock_id, exchange='TSE')

        # 計算股數
        shares = int(abs(quantity) * 1000)
        if shares == 0:
            logger.warning(f"跳過 {stock_id}：股數為 0")
            return

        logger.info(f"圈存股票(賣出): {stock_id}, 數量={abs(quantity)}張({shares}股)")

        resp = self.account.api.reserve_stock(self.account.api.stock_account, contract, shares)

        logger.info(f"圈存股票成功: {stock_id}, 回應={resp}")


class ReservationHandlerFactory:
    """圈存處理器工廠類"""

    @staticmethod
    def create(broker_name: str, account: Account) -> ReservationHandlerBase:
        """
        根據券商名稱建立對應的圈存處理器

        Args:
            broker_name: 券商名稱 ('fugle' 或 'shioaji')
            account: 券商帳號物件

        Returns:
            對應的圈存處理器實例
        """
        if broker_name == "fugle":
            return FugleReservationHandler(account)
        elif broker_name == "shioaji":
            return ShioajiReservationHandler(account)
        else:
            raise ValueError(f"Unsupported broker: {broker_name}")
