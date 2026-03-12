from finlab import data
from finlab.markets.tw import TWMarket
import pandas as pd

class TargetWeekdayTWMarket(TWMarket):
    def __init__(self, buy_weekday=None):
        super().__init__()
        self.buy_weekday = buy_weekday

    def get_trading_price(self, name, adj=True):
        return self.get_price(name, adj=adj)
    
    def get_price(self, trade_at_price, adj=True):
        if isinstance(trade_at_price, pd.Series):
            return trade_at_price.to_frame()

        if isinstance(trade_at_price, pd.DataFrame):
            return trade_at_price

        if isinstance(trade_at_price, str):
            if trade_at_price == 'volume':
                return data.get('price:成交股數')

            # 開盤價 or 收盤價 or 最高價 or 最低價
            if trade_at_price in ['open', 'close', 'high', 'low']:
                if adj:
                    table_name = 'etl:adj_'
                    price_name = trade_at_price
                else:
                    table_name = 'price:'
                    price_name = {'open': '開盤價', 'close': '收盤價', 'high': '最高價', 'low': '最低價'}[trade_at_price]

                price = data.get(f'{table_name}{price_name}')
                return price

            # 收盤價與開盤價的均價
            if trade_at_price == 'close_open_avg':
                if adj:
                    adj_open = data.get('etl:adj_open')
                    adj_close = data.get('etl:adj_close')
                    adj_avg_price = round((adj_open + adj_close)/2,2)
                    return adj_avg_price
                else:
                    open_ = data.get('price:開盤價')
                    close = data.get('price:收盤價')
                    avg_price = round((open_ + close)/2,2)
                    return avg_price

            # 最高價與最低價的均價
            if trade_at_price == 'high_low_avg':
                if adj:
                    adj_high = data.get('etl:adj_high')
                    adj_low = data.get('etl:adj_low')
                    adj_avg_price = round((adj_high + adj_low)/2,2)
                    return adj_avg_price
                else:
                    high = data.get('price:最高價')
                    low = data.get('price:最低價')
                    avg_price = round((high + low)/2,2)
                    return avg_price

            # 成交均價
            if trade_at_price == 'transaction_avg':
                vol = data.get('price:成交股數')
                vol_price = data.get('price:成交金額')
                avg_price = round(vol_price/vol,2)
                if adj:
                    close = data.get('price:收盤價')
                    adj_close = data.get('etl:adj_close')
                    adj_avg_price = adj_close/close*avg_price
                    return adj_avg_price
                else:
                    return avg_price

            # 買入日使用開盤價，其餘使用收盤價
            if trade_at_price == 'open_close_mix':
                if self.buy_weekday is None:
                    raise Exception("使用 'open_close_mix' 時，必須在初始化 MarketInfo 時提供 buy_weekday")
                if adj:
                    adj_open = data.get('etl:adj_open')
                    adj_close = data.get('etl:adj_close')

                    buy_days = adj_open.index.dayofweek == self.buy_weekday

                    adj_open_close_mix = adj_close.copy()
                    adj_open_close_mix.loc[buy_days] = adj_open.loc[buy_days]

                    return adj_open_close_mix
                else:
                    open_ = data.get('price:開盤價')
                    close = data.get('price:收盤價')

                    buy_days = open_.index.dayofweek == self.buy_weekday

                    open_close_mix = close.copy()
                    open_close_mix.loc[buy_days] = open_.loc[buy_days]
                    return open_close_mix


        raise Exception(f'**ERROR: trade_at_price is not allowed (accepted types: pd.DataFrame, pd.Series, str).')
