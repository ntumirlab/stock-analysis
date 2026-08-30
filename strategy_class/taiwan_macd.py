"""
台灣標準 MACD 指標計算模組
匹配台灣看盤軟體（XQ、Goodinfo）的 MACD 計算

與 talib.MACD 的差異只在價格基準：台股看盤軟體以「加權收盤價」(H+L+2C)/4
為輸入，talib 以收盤價為輸入。平滑公式兩者相同（EMA，α = 2/(n+1)）。

2330 對帳 XQ 還原日線圖（實價尺度）：
    2026-08-14   本地 DIF 8.591 / DEA -0.495 / OSC 9.086   XQ 8.62 / -0.45 / 9.07
    2026-08-21   本地 DIF 4.459 / DEA  3.462 / OSC 0.997   XQ 4.48 /  3.49 / 0.99
詳見 docs/20260815_EFG95_full_technical_indicator_verification.md
"""


def weighted_close(high_df, low_df, close_df):
    """加權收盤價 (最高 + 最低 + 2×收盤) / 4"""
    return (high_df + low_df + 2 * close_df) / 4


def taiwan_macd(high_df, low_df, close_df,
                fastperiod=12, slowperiod=26, signalperiod=9):
    """
    台灣標準 MACD - 完全向量化

    Args:
        high_df / low_df / close_df: 價格 DataFrame（欄為股票代號）
        fastperiod / slowperiod / signalperiod: 快線 / 慢線 / 訊號線期數

    Returns:
        (dif, dea, osc) 三個 DataFrame
        dif = EMA(fast) - EMA(slow)、dea = EMA(dif)、osc = dif - dea
    """
    wc = weighted_close(high_df, low_df, close_df)

    ema_fast = wc.ewm(span=fastperiod, adjust=False).mean()
    ema_slow = wc.ewm(span=slowperiod, adjust=False).mean()

    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signalperiod, adjust=False).mean()

    return dif, dea, dif - dea
