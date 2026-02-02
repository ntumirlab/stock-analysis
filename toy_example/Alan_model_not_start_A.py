from finlab import data
from finlab.markets.tw import TWMarket
import pandas as pd
import numpy as np
from taiwan_kd import taiwan_kd_fast

class AdjustTWMarketInfo(TWMarket):
    def get_trading_price(self, name, adj=True):
        return self.get_price(name, adj=adj).shift(1)

with data.universe(market='TSE_OTC'):
    # 獲取三大法人的買賣超股數數據
    foreign_net_buy_shares = data.get('institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)')
    investment_trust_net_buy_shares = data.get('institutional_investors_trading_summary:投信買賣超股數')
    dealer_self_net_buy_shares = data.get('institutional_investors_trading_summary:自營商買賣超股數(自行買賣)')
    # 發行股數作為總股數
    shares_outstanding = data.get('internal_equity_changes:發行股數')

def build_chip_buy_condition(top_n):
    # 計算外資、投信、自營商的買賣超佔發行量比例 (股數)
    foreign_net_buy_ratio = foreign_net_buy_shares / shares_outstanding
    investment_trust_net_buy_ratio = investment_trust_net_buy_shares / shares_outstanding
    dealer_self_net_buy_ratio = dealer_self_net_buy_shares / shares_outstanding

    # 計算外資、投信、自營商的2天、3天累積買超比例
    foreign_net_buy_ratio_2d_sum = foreign_net_buy_ratio.rolling(2).sum()
    foreign_net_buy_ratio_3d_sum = foreign_net_buy_ratio.rolling(3).sum()

    investment_trust_net_buy_ratio_2d_sum = investment_trust_net_buy_ratio.rolling(2).sum()
    investment_trust_net_buy_ratio_3d_sum = investment_trust_net_buy_ratio.rolling(3).sum()

    dealer_self_net_buy_ratio_2d_sum = dealer_self_net_buy_ratio.rolling(2).sum()
    dealer_self_net_buy_ratio_3d_sum = dealer_self_net_buy_ratio.rolling(3).sum()


    # 外資：取當天、前2天、前3天累積買超比例前幾
    foreign_top_1d_ratio = foreign_net_buy_ratio.rank(axis=1, ascending=False) <= top_n
    foreign_top_2d_ratio = foreign_net_buy_ratio_2d_sum.rank(axis=1, ascending=False) <= top_n
    foreign_top_3d_ratio = foreign_net_buy_ratio_3d_sum.rank(axis=1, ascending=False) <= top_n
    foreign_buy_condition = foreign_top_1d_ratio | foreign_top_2d_ratio | foreign_top_3d_ratio

    # 投信：取當天、前2天、前3天累積買超比例前幾
    investment_trust_top_1d_ratio = investment_trust_net_buy_ratio.rank(axis=1, ascending=False) <= top_n
    investment_trust_top_2d_ratio = investment_trust_net_buy_ratio_2d_sum.rank(axis=1, ascending=False) <= top_n
    investment_trust_top_3d_ratio = investment_trust_net_buy_ratio_3d_sum.rank(axis=1, ascending=False) <= top_n
    investment_trust_buy_condition = investment_trust_top_1d_ratio | investment_trust_top_2d_ratio | investment_trust_top_3d_ratio

    # 自營商：取當天、前2天、前3天累積買超比例前幾
    dealer_self_top_1d_ratio = dealer_self_net_buy_ratio.rank(axis=1, ascending=False) <= top_n
    dealer_self_top_2d_ratio = dealer_self_net_buy_ratio_2d_sum.rank(axis=1, ascending=False) <= top_n
    dealer_self_top_3d_ratio = dealer_self_net_buy_ratio_3d_sum.rank(axis=1, ascending=False) <= top_n
    dealer_self_buy_condition = dealer_self_top_1d_ratio | dealer_self_top_2d_ratio | dealer_self_top_3d_ratio

    # institutional_investors_top_buy_condition = foreign_buy_condition | investment_trust_buy_condition | dealer_self_buy_condition

    with data.universe(market='TSE_OTC'):
        # 獲取主力籌碼數據 (買超和賣超)
        top15_buy_shares = data.get('etl:broker_transactions:top15_buy')
        top15_sell_shares = data.get('etl:broker_transactions:top15_sell')

    # 計算買賣超差額股數
    net_buy_shares = (top15_buy_shares - top15_sell_shares) * 1000

    # 買賣超差額股數佔發行股數的比例
    net_buy_ratio = net_buy_shares / shares_outstanding

    # 計算2天、3天買賣超差額股數佔發行股數的比
    net_buy_ratio_2d_sum = net_buy_ratio.rolling(2).sum()
    net_buy_ratio_3d_sum = net_buy_ratio.rolling(3).sum()

    # 主力籌碼條件
    main_force_top_1d_buy = net_buy_ratio.rank(axis=1, ascending=False) <= top_n
    main_force_top_2d_buy = net_buy_ratio_2d_sum.rank(axis=1, ascending=False) <= top_n
    main_force_top_3d_buy = net_buy_ratio_3d_sum.rank(axis=1, ascending=False) <= top_n
    main_force_condition_1d = net_buy_ratio > 0.0008
    main_force_condition_2d = net_buy_ratio_2d_sum > 0.0015
    main_force_condition_3d = net_buy_ratio_3d_sum > 0.0025

    main_force_buy_condition = ( main_force_top_1d_buy & main_force_condition_1d ) | ( main_force_top_2d_buy & main_force_condition_2d ) | ( main_force_top_3d_buy & main_force_condition_3d )

    chip_buy_condition = foreign_buy_condition | dealer_self_buy_condition | main_force_buy_condition

    return {
        'chip_buy_condition': chip_buy_condition,
        'foreign_buy_condition': foreign_buy_condition,
        'investment_trust_buy_condition': investment_trust_buy_condition,
        'dealer_self_buy_condition': dealer_self_buy_condition,
        'main_force_buy_condition': main_force_buy_condition
    }

with data.universe(market='TSE_OTC'):
    close = data.get("price:收盤價")
    adj_close = data.get('etl:adj_close')
    adj_open = data.get('etl:adj_open')
    adj_high = data.get('etl:adj_high')
    adj_low = data.get('etl:adj_low')
    volume = data.get('price:成交股數')

def build_technical_buy_condition(bias_5_range=(-0.03, 0.10), bias_10_range=(-0.03, 0.10),
                                  bias_20_range=(-0.03, 0.14), bias_60_range=(-0.03, 0.14),
                                  bias_120_range=(-0.03, 0.25), bias_240_range=(-0.03, 0.25),
                                  new_high_days=120):

    # 計算均線
    ma3 = adj_close.rolling(3).mean()
    ma5 = adj_close.rolling(5).mean()
    ma10 = adj_close.rolling(10).mean()
    ma20 = adj_close.rolling(20).mean()
    ma60 = adj_close.rolling(60).mean()
    ma120 = adj_close.rolling(120).mean()
    ma240 = adj_close.rolling(240).mean()

    # 均線上升
    ma_up_buy_condition = (ma5 > ma5.shift(1)) & (ma10 > ma10.shift(1)) & (ma20 > ma20.shift(1)) & (ma60 > ma60.shift(1))

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

    bias_5_condition = (bias_5 >= bias_5_range[0]) & (bias_5 <= bias_5_range[1])
    bias_10_condition = (bias_10 >= bias_10_range[0]) & (bias_10 <= bias_10_range[1])
    bias_20_condition = (bias_20 >= bias_20_range[0]) & (bias_20 <= bias_20_range[1])
    bias_60_condition = (bias_60 >= bias_60_range[0]) & (bias_60 <= bias_60_range[1])
    bias_120_condition = (bias_120 >= bias_120_range[0]) & (bias_120 <= bias_120_range[1])
    bias_240_condition = (bias_240 >= bias_240_range[0]) & (bias_240 <= bias_240_range[1])


    # 設定進場乖離率
    bias_buy_condition = (
                        bias_5_condition &
                        bias_10_condition &
                        bias_20_condition &
                        bias_60_condition &
                        bias_120_condition &
                        bias_240_condition
                        )

    # 今收盤 > 今開盤，且今收盤 > 昨收盤
    positive_close_condition = (adj_close > adj_open) & (adj_close > adj_close.shift(1))

    price_above_12_condition = close > 12

    # 成交量大於昨日的2倍
    # volume_doubled_condition = volume > (volume.shift(1) * 2)

    # 今日成交張數 > 300 張
    volume_above_300_condition = volume > 300 * 1000

    # 成交金額大於 1500 萬
    amount_above_15m_condition = (close * volume) > 15000000

    with data.universe(market='TSE_OTC'):
        # 計算DMI指標
        plus_di = data.indicator('PLUS_DI', timeperiod=14, adjust_price=True)
        minus_di = data.indicator('MINUS_DI', timeperiod=14, adjust_price=True)

    # DMI條件
    dmi_buy_condition = (plus_di > 24) & (minus_di < 21)

    # 計算 KD 指標
    # with data.universe(market='TSE_OTC'):
    #     k, d = data.indicator('STOCH',
    #                             fastk_period=9,
    #                             slowk_period=3,
    #                             slowk_matype=0,
    #                             slowd_period=3,
    #                             slowd_matype=0,
    #                             adjust_price=True
    #                             )
    k, d = taiwan_kd_fast(
        high_df=adj_high,
        low_df=adj_low,
        close_df=adj_close,
        fastk_period=9,
        alpha=1/3
    )


    # KD 指標條件：%K 和 %D 都向上
    k_up_condition = k > k.shift(1)
    d_up_condition = d > d.shift(1)
    kd_buy_condition = k_up_condition & d_up_condition

    with data.universe(market='TSE_OTC'):
        # 計算 MACD 指標
        dif, macd , _  = data.indicator('MACD', fastperiod=12, slowperiod=26, signalperiod=9, adjust_price=True)

    # MACD DIF 向上
    macd_dif_buy_condition = dif > dif.shift(1)


    # 120天新高 = 120天內的最高價(盤中High)
    high_120 = adj_high.rolling(window=new_high_days).max()
    # 120天收盤新高 = 120天內的最高收盤價(Close)
    close_high_120 = adj_close.rolling(window=new_high_days).max()

    # 買入條件:價格在「120天收盤新高」的93%以上
    price_above_93pct_close_high_condition = adj_close >= (close_high_120 * 0.93)

    # 技術面
    technical_buy_condition = (
        # ma_up_buy_condition &
        # ma5_above_others_condition &
        # price_above_ma_buy_condition &
        bias_buy_condition &  # 乖離率限制,篩選盤整很久的股票
        # volume_doubled_condition &
        # positive_close_condition &
        volume_above_300_condition &  # 成交 > 300張
        price_above_12_condition &  # 價格 > 12元
        amount_above_15m_condition &  # 成交金額 > 1500萬

        # dmi_buy_condition &
        # kd_buy_condition &
        # macd_dif_buy_condition &
        price_above_93pct_close_high_condition  # 價格在120天收盤新高的93%以上
    )

    return {
        'technical_buy_condition': technical_buy_condition,
        # 'ma_up_buy_condition': ma_up_buy_condition,
        # 'price_above_ma_buy_condition': price_above_ma_buy_condition,
        'bias_buy_condition': bias_buy_condition,
        # 'volume_doubled_condition': volume_doubled_condition,
        'volume_above_300_condition': volume_above_300_condition,
        'price_above_12_condition': price_above_12_condition,
        'amount_above_15m_condition': amount_above_15m_condition,
        # 'dmi_buy_condition': dmi_buy_condition,
        # 'kd_buy_condition': kd_buy_condition,
        # 'macd_dif_buy_condition': macd_dif_buy_condition,
        'price_above_93pct_close_high_condition': price_above_93pct_close_high_condition,

        'bias_values': {
            'bias_5': bias_5,
            'bias_10': bias_10,
            'bias_20': bias_20,
            'bias_60': bias_60,
            'bias_120': bias_120,
            'bias_240': bias_240
        },
        'bias_conditions': {
            'bias_5_condition': bias_5_condition,
            'bias_10_condition': bias_10_condition,
            'bias_20_condition': bias_20_condition,
            'bias_60_condition': bias_60_condition,
            'bias_120_condition': bias_120_condition,
            'bias_240_condition': bias_240_condition
        },

        'kd_values': {
            'k_value': k,
            'd_value': d
        },
        'kd_conditions': {
            'k_up_condition': k_up_condition,
            'd_up_condition': d_up_condition,
            'kd_buy_condition': kd_buy_condition
        },

        'price_data': {
            'adj_close': adj_close,
            'high_120': high_120,
            'close_high_120': close_high_120,
        },

    }

with data.universe(market='TSE_OTC'):
    operating_margin = data.get('fundamental_features:營業利益率')

def build_fundamental_buy_condition(op_growth_threshold):
    operating_margin_increase = (operating_margin > (operating_margin.shift(1) * op_growth_threshold))

    fundamental_buy_condition = (
        operating_margin_increase
    )

    return {
        'fundamental_buy_condition': fundamental_buy_condition,
        'operating_margin_increase': operating_margin_increase,
    }


# A 尚未發動股追蹤模型
# 營益率增12.5%
# 買超前25檔
# 乖離率: -3~13%, -3~16%, -3~19%, -3~20%, -3~27%, 0~31%
# 成交量擴增不限, 價格>12元, 成交>300張, 成交金額>1500萬
# 價格在120天收盤新高的93%以上
buy_signal = (
    build_chip_buy_condition(top_n=25)['chip_buy_condition'] &
    build_technical_buy_condition(
        bias_5_range=(-0.03, 0.13),
        bias_10_range=(-0.03, 0.16),
        bias_20_range=(-0.03, 0.19),
        bias_60_range=(-0.03, 0.20),
        bias_120_range=(-0.03, 0.27),  # A版: -3~27%
        bias_240_range=(0.00, 0.31)    # A版: 0~31%
    )['technical_buy_condition'] &
    build_fundamental_buy_condition(1.125)['fundamental_buy_condition']  # 營益率增12.5%
)

# 設定起始買入日期
start_buy_date = '2017-12-31'
buy_signal = buy_signal.loc[start_buy_date:]

def build_sell_condition():
    # 計算均線
    ma5 = adj_close.rolling(5).mean()
    ma10 = adj_close.rolling(10).mean()
    ma60 = adj_close.rolling(60).mean()
    ma240 = adj_close.rolling(240).mean()

    # 計算乖離率
    bias_5 = (adj_close - ma5) / ma5
    bias_10 = (adj_close - ma10) / ma10
    bias_60 = (adj_close - ma60) / ma60
    bias_240 = (adj_close - ma240) / ma240

    # 計算120天收盤新高(Close)
    close_high_120 = adj_close.rolling(window=120).max()

    # A版賣出條件:
    # 1. 5日線乖離小於-4% 或
    # 2. 10日線乖離小於-4% 或
    # 3. 價格小於「120天收盤新高」的91% 或
    # 4. 60日線乖離大於21% 或
    # 5. 240日線乖離大於32%
    sell_condition = (
        (bias_5 < -0.04) |
        (bias_10 < -0.04) |
        (adj_close < (close_high_120 * 0.91)) |
        (bias_60 > 0.21) |
        (bias_240 > 0.32)  # A版: 32%
    )

    return sell_condition

sell_condition = build_sell_condition()
position = buy_signal.hold_until(sell_condition)


# 執行回測
from finlab.backtest import sim
report = sim(position, resample=None, upload=False, market=AdjustTWMarketInfo())
# report = sim(position, resample=None, upload=False, trade_at_price='open')
# report = sim(position, resample=None, upload=False, trade_at_price='open', position_limit=0.25)
