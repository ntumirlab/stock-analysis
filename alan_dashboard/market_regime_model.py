"""
多空轉折模型：評分邏輯
======================
純計算模組，不含 Dash 與 FinLab 載入，供 ``alan_dashboard/pages/market_regime.py`` 使用，
也讓 ``tests/unit`` 可以在沒有 finlab 的環境測試市值前 150 檔與現貨檔數差的邏輯
（talib 只在 ``compute_components`` 內延遲 import）。

分數組成（總分 −10 ~ +10）：

- 均線分（−2 ~ +2）：收盤與 MA5／MA10／MA20 的相對位置
- DMI 分（−4 ~ +4）：+DI 與 −DI 各依門檻計分
- 動能微調（−3 ~ +3）：小計 < 5 時 DIF／MACD／KD 下彎各 −1；小計 > −5 時上彎各 +1
- 現貨分（−1 ~ +1）：市值前 150 檔中「三條均線同時向上」減「同時向下」的檔數差，
  > +25 為 +1、< −25 為 −1，直接加在最後，不影響動能微調的觸發條件
"""

import pandas as pd

TOP_N = 150             # 台灣50 + 台灣中型100
BREADTH_THRESHOLD = 25  # 檔數差門檻
MA_WINDOWS = (5, 10, 20)


# ── 現貨：市值前 150 檔 ────────────────────────────────────────────────────────

def listed_common_stocks(categories: pd.DataFrame) -> list[str]:
    """上市普通股代號：market 為 sii、4 碼數字（排除 ETF、特別股、TDR），排除創新板（名稱 -創）。"""
    df = categories[categories['market'] == 'sii']
    ids = df['stock_id'].astype(str)
    names = df['name'].astype(str)
    keep = ids.str.fullmatch(r'\d{4}') & ~names.str.endswith('-創')
    return ids[keep].tolist()


def top_n_membership(market_value: pd.DataFrame, universe: list[str], n: int = TOP_N) -> pd.DataFrame:
    """每季末以市值重排一次、整季鎖定的前 n 大成分（bool，日期 × 股票）。

    以每季最後一個交易日的市值排名，自次一交易日起生效，直到下一季末；
    對應台灣50／中型100 每季審核換股的行為（近似，官方另有自由流通與緩衝規則）。
    """
    cols = [c for c in universe if c in market_value.columns]
    mv = market_value[cols]
    quarterly = mv.resample('QE').last().dropna(how='all')
    rank = quarterly.rank(axis=1, ascending=False, method='first')
    member_q = (rank <= n).astype(float)  # float：reindex/shift 產生的 NaN 不會變成 object dtype
    member = member_q.reindex(mv.index, method='ffill').shift(1)  # 季末次一交易日生效
    return member.fillna(0).astype(bool)


def ma_direction_breadth(close: pd.DataFrame, membership: pd.DataFrame) -> pd.Series:
    """成分股中「MA5/10/20 同時向上」的檔數減「同時向下」的檔數（向上 = 當日均線值高於前一日）。"""
    cols = membership.columns.intersection(close.columns)
    c = close[cols]
    # 成分名單以市值資料日為準、整季不變，故對收盤價日期 ffill（個股價格通常比市值早更新）
    member = membership[cols].astype(float).reindex(c.index, method='ffill').fillna(0).astype(bool)
    rising = pd.DataFrame(True, index=c.index, columns=cols)
    falling = rising.copy()
    for w in MA_WINDOWS:
        ma = c.rolling(w).mean()
        rising &= ma > ma.shift(1)
        falling &= ma < ma.shift(1)
    diff = (rising & member).sum(axis=1) - (falling & member).sum(axis=1)
    # 個股價格尚未更新（該日成分股有收盤價的不足半數）時視為無值，避免被算成 0
    valid = (c.notna() & member).sum(axis=1)
    return diff.where(valid * 2 >= member.sum(axis=1)).astype(float)


def breadth_score(diff: pd.Series, threshold: int = BREADTH_THRESHOLD) -> pd.Series:
    """檔數差 > threshold 為 +1、< −threshold 為 −1，其餘 0。"""
    return (diff > threshold).astype(int) - (diff < -threshold).astype(int)


# ── 指數本身：均線 + DMI + 動能 ─────────────────────────────────────────────────

def _direction(series: pd.Series) -> pd.Series:
    """較前一日上升 +1、下降 −1、持平或無值 0。"""
    return (series > series.shift(1)).astype(int) - (series < series.shift(1)).astype(int)


def compute_components(ohlc: pd.DataFrame, dmi_hi: int, dmi_mid: int, dmi_lo: int,
                       breadth_diff: pd.Series | None = None) -> pd.DataFrame:
    """回傳每日各條件的數值與分數（DataFrame），``total`` 欄為總分。

    欄位：close, above_ma5/10/20, ma_score, plus_di, minus_di, di_score, subtotal,
    dif_dir, dif_score, macd_dir, macd_score, kd_dir, kd_score,
    breadth_diff, breadth_score, total
    """
    import talib
    from talib import abstract

    close, high, low = ohlc['close'], ohlc['high'], ohlc['low']

    plus_di  = abstract.Function('PLUS_DI')(ohlc, timeperiod=14)
    minus_di = abstract.Function('MINUS_DI')(ohlc, timeperiod=14)
    di_score = (
        (plus_di > dmi_hi).astype(int)                               *  2
      + ((plus_di >= dmi_mid) & (plus_di <= dmi_hi)).astype(int)    *  1
      + ((plus_di >= dmi_lo)  & (plus_di <  dmi_mid)).astype(int)   * -1
      + (plus_di < dmi_lo).astype(int)                              * -2
      + (minus_di > dmi_hi).astype(int)                             * -2
      + ((minus_di >= dmi_mid) & (minus_di <= dmi_hi)).astype(int)  * -1
      + ((minus_di >= dmi_lo)  & (minus_di <  dmi_mid)).astype(int) *  1
      + (minus_di < dmi_lo).astype(int)                             *  2
    )

    dif, macd, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    dif_dir, macd_dir = _direction(dif), _direction(macd)

    low9  = low.rolling(9).min()
    high9 = high.rolling(9).max()
    rsv   = ((close - low9) / (high9 - low9).replace(0, float('nan')) * 100).fillna(50)
    K = rsv.ewm(com=2, adjust=False).mean()
    D = K.ewm(com=2, adjust=False).mean()
    k_dir, d_dir = _direction(K), _direction(D)
    kd_dir = ((k_dir == 1) & (d_dir == 1)).astype(int) - ((k_dir == -1) & (d_dir == -1)).astype(int)

    above = {w: close > close.rolling(w).mean() for w in MA_WINDOWS}
    a5, a10, a20 = above[5], above[10], above[20]
    ma_score = (
        ( a5 &  a10 &  a20).astype(int) *  2
      + ( a5 &  a10 & ~a20).astype(int) *  1
      + (~a5 &  a10 &  a20).astype(int) *  1
      + (~a5 & ~a10 &  a20).astype(int) * -1
      + (~a5 & ~a10 & ~a20).astype(int) * -2
    )

    subtotal = ma_score + di_score
    lt5, gtn5 = subtotal < 5, subtotal > -5

    def _momentum(direction: pd.Series) -> pd.Series:
        return (gtn5 & (direction == 1)).astype(int) - (lt5 & (direction == -1)).astype(int)

    dif_score, macd_score, kd_score = _momentum(dif_dir), _momentum(macd_dir), _momentum(kd_dir)

    # 無成分股資料的日期（例如指數已更新、個股尚未更新）保留 NaN：現貨分為 0，明細表顯示「—」
    if breadth_diff is None:
        b_diff = pd.Series(float('nan'), index=ohlc.index)
    else:
        b_diff = breadth_diff.reindex(ohlc.index).astype(float)
    b_score = breadth_score(b_diff)

    total = subtotal + dif_score + macd_score + kd_score + b_score

    return pd.DataFrame({
        'close': close,
        'above_ma5': a5, 'above_ma10': a10, 'above_ma20': a20, 'ma_score': ma_score,
        'plus_di': plus_di, 'minus_di': minus_di, 'di_score': di_score,
        'subtotal': subtotal,
        'dif_dir': dif_dir, 'dif_score': dif_score,
        'macd_dir': macd_dir, 'macd_score': macd_score,
        'kd_dir': kd_dir, 'kd_score': kd_score,
        'breadth_diff': b_diff, 'breadth_score': b_score,
        'total': total,
    })
