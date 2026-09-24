"""
多空轉折模型：評分邏輯
======================
純計算模組，不含 Dash 與 FinLab 載入，供 ``alan_dashboard/pages/market_regime.py`` 使用，
也讓 ``tests/unit`` 可以在沒有 finlab 的環境測試市值前 150 檔與現貨檔數差的邏輯
（talib 只在 ``compute_components`` 內延遲 import）。

分數組成（總分 −12 ~ +12）：

- 均線分（−2 ~ +2）：收盤與 MA5／MA10／MA20 的相對位置
- DMI 分（−4 ~ +4）：+DI 與 −DI 各依門檻計分
- 動能微調（−3 ~ +3）：小計 < 5 時 DIF／MACD／KD 下彎各 −1；小計 > −5 時上彎各 +1
- 現貨分（−1 ~ +1）：台灣50 + 中型100（以市值前 150 檔模擬）中「三條均線同時向上」減
  「同時向下」的檔數差，> +25 為 +1、< −25 為 −1，直接加在最後，不影響動能微調的觸發條件
- 選擇權分（−2 ~ +2）：台指選擇權未平倉量 Put/Call 比（賣權 ÷ 買權），分成當月（最近未到期月契約）
  與遠月（其餘月契約合計，週契約不計）。當月 > 120% 為 +1、當月 < 90% 為 −1、
  當月 > 140% 且 > 遠月再 +1、遠月 > 100% 且 > 當月再 −1；各條件可累加，同現貨分直接加在最後

MACD 與 KD 採 strategy_class 的台股自訂算法（taiwan_macd：加權收盤價 (H+L+2C)/4；
taiwan_kd_fast：alpha = 1/3），與各策略一致；DMI 用 talib。
"""

import pandas as pd

TOP_N = 150             # 台灣50 + 台灣中型100
BREADTH_THRESHOLD = 25  # 檔數差門檻
BREADTH_MIN_COVERAGE = 0.9  # 當日有收盤價的成分股比例低於此值視為資料未更新（容許少數停牌）
MA_WINDOWS = (5, 10, 20)
# 選擇權分門檻（未平倉 P/C，%）
PCR_NEAR_LONG = 120    # 當月 > 此值 +1
PCR_NEAR_SHORT = 90    # 當月 < 此值 −1
PCR_NEAR_STRONG = 140  # 當月 > 此值且 > 遠月，再 +1
PCR_FAR_SHORT = 100    # 遠月 > 此值且 > 當月，再 −1
DIRECTION_TOL = 1e-9  # 上升／下降判斷的相對容忍值：浮點運算（rolling mean 滑動累加、EMA 遞迴）殘留 1e-13 等級誤差，
                      # 變動未超過自身十億分之一者視為持平；均線與 DIF/DEA/K/D 一體適用，與策略 direction_tol 相同

# 台灣50／中型100 季度審核（FTSE TWSE Taiwan Index Series Ground Rules 6.1、6.3）：
# 每年 3、6、9、12 月審核，變動於該月第三個星期五收盤後生效（即下一個交易日，通常是星期一），
# 審核資料為生效日前四週星期一的收盤資料。
# 緩衝規則：非成分股市值排名升至 ENTRY_RANK 以上才納入，成分股排名跌至 EXIT_RANK 以下才剔除，
# 名單數目維持固定（不足由排名最高的非成分股遞補、超出則剔除排名最低者）。
# 這裡模擬的是 150 檔合併名單，只需要 150 名邊界的門檻（中型100 為 130／171；台灣50 本身是 40／61）。
REVIEW_MONTHS = (3, 6, 9, 12)
ENTRY_RANK = 130
EXIT_RANK = 171


# ── 現貨：市值前 150 檔 ────────────────────────────────────────────────────────

def listed_common_stocks(categories: pd.DataFrame) -> list[str]:
    """上市普通股代號：market 為 sii、4 碼數字（排除 ETF、特別股），排除存託憑證與創新板（名稱 -創）。

    TDR（台灣存託憑證）有些是 4 碼代號（如 9103 美德醫療-DR），要用類別「存託憑證」明確排除；
    實際上 FinLab 的市值資料（etl:market_value）沒有任何 TDR，排不進排名，這裡排除是為了與文件一致。
    """
    df = categories[(categories['market'] == 'sii') & (categories['category'] != '存託憑證')]
    ids = df['stock_id'].astype(str)
    names = df['name'].astype(str)
    keep = ids.str.fullmatch(r'\d{4}') & ~names.str.endswith('-創')
    return ids[keep].tolist()


def _third_friday(year: int, month: int) -> pd.Timestamp:
    first = pd.Timestamp(year=year, month=month, day=1)
    return first + pd.Timedelta(days=(4 - first.weekday()) % 7 + 14)


def review_schedule(trading_days: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """回傳每次季度審核的 (資料截止日, 生效日)。

    生效日 = 3/6/9/12 月第三個星期五之後的第一個交易日；
    截止日 = 名義生效日（第三個星期五的下一個星期一）往前四週當天或之前最近的交易日。
    生效日尚未有交易資料的審核不列入（例如 9 月的審核在第三個星期五後才生效，之前仍沿用 6 月名單）。
    """
    schedule = []
    for year in range(trading_days[0].year, trading_days[-1].year + 1):
        for month in REVIEW_MONTHS:
            third_friday = _third_friday(year, month)
            after = trading_days[trading_days > third_friday]
            on_or_before_cutoff = trading_days[trading_days <= third_friday + pd.Timedelta(days=3 - 28)]
            if len(after) == 0 or len(on_or_before_cutoff) == 0:
                continue
            schedule.append((on_or_before_cutoff[-1], after[0]))
    return schedule


def top_n_membership(market_value: pd.DataFrame, universe: list[str], n: int = TOP_N,
                     entry_rank: int = ENTRY_RANK, exit_rank: int = EXIT_RANK) -> pd.DataFrame:
    """依季度審核時程與緩衝規則模擬的前 n 大成分（bool，日期 × 股票）。

    第一次審核直接取市值前 n 名；之後每次審核：排名跌到 exit_rank（含）以下的成分股剔除、
    排名升到 entry_rank（含）以上的非成分股納入，再以排名補足／削減到 n 檔。
    """
    cols = [c for c in universe if c in market_value.columns]
    mv = market_value[cols]
    member = pd.DataFrame(False, index=mv.index, columns=cols)
    current: set[str] = set()
    for cutoff, effective in review_schedule(mv.index):
        rank = mv.loc[cutoff].dropna().rank(ascending=False, method='first')
        if not current:
            current = set(rank[rank <= n].index)
        else:
            keep = {s for s in current if rank.get(s, float('inf')) < exit_rank}
            new = keep | set(rank[rank <= entry_rank].index)
            if len(new) < n:  # 遞補：排名最高的非成分股
                new |= set(rank.drop(list(new)).nsmallest(n - len(new)).index)
            elif len(new) > n:  # 削減：排名最低的成分股
                new -= set(rank.reindex(list(new)).nlargest(len(new) - n).index)
            current = new
        member.loc[effective:] = False
        member.loc[effective:, list(current)] = True
    return member


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
        delta, tol = ma - ma.shift(1), ma.abs() * DIRECTION_TOL
        rising &= delta > tol
        falling &= delta < -tol
    diff = (rising & member).sum(axis=1) - (falling & member).sum(axis=1)
    # 尚無成分名單、或個股價格尚未更新／只更新一部分時視為無值：缺價的股票既不算向上也不算向下，
    # 部分更新會讓檔數差失真，所以要求當日有收盤價的成分股達 BREADTH_MIN_COVERAGE（容許少數停牌）
    n_member = member.sum(axis=1)
    valid = (c.notna() & member).sum(axis=1)
    return diff.where((n_member > 0) & (valid >= n_member * BREADTH_MIN_COVERAGE)).astype(float)


def breadth_score(diff: pd.Series, threshold: int = BREADTH_THRESHOLD) -> pd.Series:
    """檔數差 > threshold 為 +1、< −threshold 為 −1，其餘 0。"""
    return (diff > threshold).astype(int) - (diff < -threshold).astype(int)


# ── 指數本身：均線 + DMI + 動能 ─────────────────────────────────────────────────

def _direction(series: pd.Series) -> pd.Series:
    """較前一日上升 +1、下降 −1、持平或無值 0（變動未超過 DIRECTION_TOL 倍視為持平）。"""
    delta, tol = series - series.shift(1), series.abs() * DIRECTION_TOL
    return (delta > tol).astype(int) - (delta < -tol).astype(int)


def compute_components(ohlc: pd.DataFrame, dmi_hi: int, dmi_mid: int, dmi_lo: int,
                       breadth_diff: pd.Series | None = None,
                       pcr: pd.DataFrame | None = None) -> pd.DataFrame:
    """回傳每日各條件的數值與分數（DataFrame），``total`` 欄為總分。

    ``pcr`` 為 ``txo_put_call_ratio`` 的輸出（near_pcr、far_pcr），無值的日期選擇權分為 0。

    欄位：close, above_ma5/10/20, ma_score, plus_di, minus_di, di_score, subtotal,
    dif_dir, dif_score, macd_dir, macd_score, k_dir, d_dir, kd_dir, kd_score,
    breadth_diff, breadth_score, pcr_near, pcr_far, options_score, total
    """
    from talib import abstract
    from strategy_class.taiwan_kd import taiwan_kd_fast
    from strategy_class.taiwan_macd import taiwan_macd

    close, high, low = ohlc['close'], ohlc['high'], ohlc['low']
    # 自訂指標吃「欄為股票」的 DataFrame，指數以單欄 DataFrame 傳入
    frames = tuple(x.to_frame('idx') for x in (high, low, close))

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

    dif, dea, _ = taiwan_macd(*frames, fastperiod=12, slowperiod=26, signalperiod=9)
    dif_dir, macd_dir = _direction(dif['idx']), _direction(dea['idx'])

    K, D = taiwan_kd_fast(*frames, fastk_period=9, alpha=1 / 3, verbose=False)
    k_dir, d_dir = _direction(K['idx']), _direction(D['idx'])
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

    # 選擇權分同現貨分：資料起始（2025-09-25）前或尚未更新的日期為 NaN，計 0、明細表顯示「—」
    if pcr is None:
        near = far = pd.Series(float('nan'), index=ohlc.index)
    else:
        near, far = pcr['near_pcr'].reindex(ohlc.index), pcr['far_pcr'].reindex(ohlc.index)
    o_score = options_score(near, far)

    total = subtotal + dif_score + macd_score + kd_score + b_score + o_score

    return pd.DataFrame({
        'close': close,
        'above_ma5': a5, 'above_ma10': a10, 'above_ma20': a20, 'ma_score': ma_score,
        'plus_di': plus_di, 'minus_di': minus_di, 'di_score': di_score,
        'subtotal': subtotal,
        'dif_dir': dif_dir, 'dif_score': dif_score,
        'macd_dir': macd_dir, 'macd_score': macd_score,
        'k_dir': k_dir, 'd_dir': d_dir, 'kd_dir': kd_dir, 'kd_score': kd_score,
        'breadth_diff': b_diff, 'breadth_score': b_score,
        'pcr_near': near, 'pcr_far': far, 'options_score': o_score,
        'total': total,
    })


# ── 選擇權：台指選擇權未平倉 Put/Call 比 ─────────────────────────────────────

# 到期月份代號：月契約為 YYYYMM（含季月）；週契約為 YYYYMMWn（週三到期）、YYYYMMFn（週二到期），不計
_MONTHLY_EXPIRY = r'\d{6}'


def txo_put_call_ratio(liquidity: pd.DataFrame) -> pd.DataFrame:
    """台指選擇權（TXO）未平倉量 Put/Call 比（%），依到期月份分成當月與遠月。

    ``liquidity`` 為 FinLab ``tw_taifex_option_liquidity``（逐契約未沖銷部位），
    需要欄位 symbol、date、買賣權、到期月份(週別)、未沖銷部位。

    - 當月 = 當日最近一個尚未到期的月契約；遠月 = 其餘所有月契約（含季月）合計
    - 週契約不計；月契約到期日當天的資料已不含該契約，當月自動換成下一個月
    - 比率 = 賣權未沖銷部位 ÷ 買權未沖銷部位 × 100；買權為 0 時無值

    回傳 DataFrame（index = date）：``near_month``（當月代號 YYYYMM）、``near_pcr``、``far_pcr``。
    """
    df = liquidity[liquidity['symbol'] == 'TXO']
    expiry = df['到期月份(週別)'].astype(str)
    df = df[expiry.str.fullmatch(_MONTHLY_EXPIRY)].assign(expiry=expiry)
    near_month = df.groupby('date')['expiry'].min()  # YYYYMM 字串排序即時間排序
    is_near = df['expiry'] == df['date'].map(near_month)
    oi = (df.groupby([df['date'], is_near.rename('is_near'), df['買賣權']], observed=True)['未沖銷部位'].sum()
            .unstack('買賣權').reindex(columns=['賣權', '買權'], fill_value=0.0))
    pcr = (oi['賣權'] / oi['買權'] * 100).where(oi['買權'] > 0).unstack('is_near').reindex(columns=[True, False])
    return pd.DataFrame({'near_month': near_month, 'near_pcr': pcr[True], 'far_pcr': pcr[False]})


def options_score(near_pcr: pd.Series, far_pcr: pd.Series) -> pd.Series:
    """選擇權分（−2 ~ +2）：四個條件各自成立即計分、可累加；P/C 無值的條件視為不成立。

    - 當月 > 120%：+1
    - 當月 < 90%：−1
    - 當月 > 140% 且當月 > 遠月：+1
    - 遠月 > 100% 且遠月 > 當月：−1
    """
    return (
        (near_pcr > PCR_NEAR_LONG).astype(int)
        - (near_pcr < PCR_NEAR_SHORT).astype(int)
        + ((near_pcr > PCR_NEAR_STRONG) & (near_pcr > far_pcr)).astype(int)
        - ((far_pcr > PCR_FAR_SHORT) & (far_pcr > near_pcr)).astype(int)
    )
