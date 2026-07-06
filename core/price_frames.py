"""價格 DataFrame 的純運算（無 finlab 依賴，CI 可測）。

供 markets/target_weekday_tw_market.py 使用。
"""

import pandas as pd


def mix_open_close(open_df: pd.DataFrame, close_df: pd.DataFrame,
                   buy_weekday: int) -> pd.DataFrame:
    """買入日（pandas dayofweek 慣例，週一=0）用開盤價、其餘日用收盤價。

    開盤價與收盤價兩個資料集在收盤後是非同步更新的，index / columns
    可能短暫不一致（例如 adj_close 已有今日、adj_open 還停在前一交易日，
    會使 boolean mask 長度對不上而 IndexError）。先取交集對齊——行為
    等同於「資料尚未更新的早晨」，多出來的那一天下次執行自然補上。
    """
    open_df, close_df = open_df.align(close_df, join='inner')
    buy_days = open_df.index.dayofweek == buy_weekday
    mixed = close_df.copy()
    mixed.loc[buy_days] = open_df.loc[buy_days]
    return mixed
