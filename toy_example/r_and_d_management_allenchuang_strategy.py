from finlab import backtest
from finlab import data

with data.universe(market='TSE_OTC'):
    close = data.get('price:收盤價')
    adj_close = data.get('etl:adj_close')
    adj_open = data.get('etl:adj_open')
    volume = data.get('price:成交股數')


def build_technical_buy_condition():

    # 計算均線
    ma3 = adj_close.rolling(3).mean()
    ma5 = adj_close.rolling(5).mean()
    ma10 = adj_close.rolling(10).mean()
    ma20 = adj_close.rolling(20).mean()
    ma60 = adj_close.rolling(60).mean()
    ma120 = adj_close.rolling(120).mean()
    ma240 = adj_close.rolling(240).mean()

    # 均線上升
    ma_up_buy_condition = (
        (ma5 > ma5.shift(1)) & (ma10 > ma10.shift(1)) & (ma20 > ma20.shift(1)) & (ma60 > ma60.shift(1))
    )

    # 5 日線大於 60/240 日線
    ma5_above_others_condition = (ma5 > ma60) & (ma5 > ma240)

    # 價格在均線之上
    price_above_ma_buy_condition = (adj_close > ma5) & (adj_close > ma10) & (adj_close > ma20) & (adj_close > ma60)

    # 計算乖離率
    bias_5 = (adj_close - ma5) / ma5
    bias_10 = (adj_close - ma10) / ma10
    bias_20 = (adj_close - ma20) / ma20
    bias_60 = (adj_close - ma60) / ma60
    bias_120 = (adj_close - ma120) / ma120
    bias_240 = (adj_close - ma240) / ma240

    # 設定進場乖離率
    bias_buy_condition = (
        (bias_5 <= 0.12)
        & (bias_5 >= 0.02)
        & (bias_10 <= 0.15)
        & (bias_10 >= 0.05)
        & (bias_20 <= 0.20)
        & (bias_20 >= 0.05)
        & (bias_60 <= 0.20)
        & (bias_60 >= 0.05)
        & (bias_120 <= 0.25)
        & (bias_120 >= 0.10)
        & (bias_240 <= 0.25)
        & (bias_240 >= 0.10)
    )

    # 今收盤 > 今開盤，且今收盤 > 昨收盤
    positive_close_condition = (adj_close > adj_open) & (adj_close > adj_close.shift(1))

    price_above_12_condition = close > 12

    # 成交量大於昨日的2倍
    volume_condition = volume > volume.shift(1)

    # 今日成交張數 > 500 張
    volume_above_500_condition = volume > 500 * 1000

    # 成交金額大於 3000 萬
    amount_condition = (close * volume) > 30000000

    with data.universe(market='TSE_OTC'):
        # 計算DMI指標
        plus_di = data.indicator('PLUS_DI', timeperiod=14, adjust_price=True)
        minus_di = data.indicator('MINUS_DI', timeperiod=14, adjust_price=True)

    # DMI條件
    dmi_buy_condition = (plus_di > 24) & (minus_di < 21)

    with data.universe(market='TSE_OTC'):
        # 計算 KD 指標
        k, d = data.indicator('STOCH', fastk_period=9, slowk_period=3, slowd_period=3, adjust_price=True)

    # KD 指標條件：%K 和 %D 都向上
    kd_buy_condition = (k > k.shift(1)) & (d > d.shift(1))

    with data.universe(market='TSE_OTC'):
        # 計算 MACD 指標
        dif, macd, _ = data.indicator('MACD', fastperiod=12, slowperiod=26, signalperiod=9, adjust_price=True)

    # MACD DIF 向上
    macd_dif_buy_condition = dif > dif.shift(1)

    # 創新高
    high_60 = adj_close.rolling(window=60).max()
    new_high_60_condition = adj_close >= high_60
    new_high_condition = new_high_60_condition

    # 技術面
    technical_buy_condition = (
        ma_up_buy_condition
        &
        # ma5_above_others_condition &
        price_above_ma_buy_condition
        & bias_buy_condition
        & volume_condition
        &
        # positive_close_condition &
        volume_above_500_condition
        & price_above_12_condition
        & amount_condition
        & dmi_buy_condition
        & kd_buy_condition
        & macd_dif_buy_condition
        & new_high_condition
    )
    return technical_buy_condition


with data.universe(market='TSE_OTC'):
    rd_ratio = data.get('fundamental_features:研究發展費用率')
    pm_ratio = data.get('fundamental_features:管理費用率')
    eq_ratio = data.get('fundamental_features:淨值除資產').deadline()

rd_pm = rd_ratio / pm_ratio
eq_price = eq_ratio / close.reindex(eq_ratio.index, method='ffill')

rebalance = eq_price.index

buy_signal = eq_price[
    (build_technical_buy_condition() & (rd_pm.deadline().rank(axis=1, pct=True) > 0.5)).reindex(rebalance)
].is_largest(50)


def build_sell_condition():
    ma3 = adj_close.rolling(3).mean()
    ma5 = adj_close.rolling(5).mean()
    ma20 = adj_close.rolling(20).mean()
    dif, macd, _ = data.indicator('MACD', fastperiod=12, slowperiod=26, signalperiod=9, adjust_price=True)

    # 法一: 短線出場
    # sell_condition = (ma3 < ma3.shift(1)) & (dif < dif.shift(1))

    # 法二: 中線出場
    sell_condition = (
        (ma5 < ma5.shift(1)) & (dif < dif.shift(1)) & (macd < macd.shift(1)) & (adj_close < ma20).sustain(2)
    )

    return sell_condition


sell_condition = build_sell_condition()
position = buy_signal.hold_until(sell_condition)


rebalance = eq_ratio.index
report = backtest.sim(position.loc['2020':])
# report = backtest.sim(buy_signal.loc['2020':], resample=rebalance)
