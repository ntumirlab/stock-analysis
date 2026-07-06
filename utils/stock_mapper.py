import logging
from finlab import data
import pandas as pd

logger = logging.getLogger(__name__)


class StockMapper:
    """stock_id → 公司簡稱 的對照（DB 紀錄與通知的顯示用途）。

    純裝飾性功能：載入失敗時退化成「代號當名稱」，不得中止下單流程。
    """

    def __init__(self):
        try:
            self.mapping = self._load_mapping()
        except Exception as e:
            logger.warning(f"載入 company_basic_info 失敗，股票名稱將以代號代替: {e}")
            self.mapping = {}

    def _load_mapping(self):
        df = data.get('company_basic_info')
        mapping = pd.Series(df['公司簡稱'].values, index=df['stock_id'].astype(str)).to_dict()
        return mapping

    def map(self, stock_id):
        return self.mapping.get(str(stock_id), stock_id)


if __name__ == '__main__':
    mapper = StockMapper()
    print("Stock 1101 maps to:", mapper.map("1101"))
