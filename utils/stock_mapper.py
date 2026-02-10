from finlab import data
import pandas as pd


class StockMapper:
    def __init__(self):
        self.mapping = self._load_mapping()

    def _load_mapping(self):
        df = data.get('company_basic_info')
        mapping = pd.Series(df['公司簡稱'].values, index=df['stock_id'].astype(str)).to_dict()
        return mapping

    def map(self, stock_id):
        return self.mapping.get(str(stock_id), stock_id)


if __name__ == '__main__':
    mapper = StockMapper()
    print("Stock 1101 maps to:", mapper.map("1101"))
