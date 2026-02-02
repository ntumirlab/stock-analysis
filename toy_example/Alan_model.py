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

def build_technical_buy_condition(bias_5_range=(0.03, 0.13), bias_10_range=(0.05, 0.16),
                                  bias_20_range=(0.05, 0.21), bias_60_range=(0.08, 0.20),
                                  bias_120_range=(0.05, 0.26), bias_240_range=(0.08, 0.26),
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
    volume_doubled_condition = volume > (volume.shift(1) * 2)

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

    # 創新高 - 根據指定天數計算
    high_n = adj_close.rolling(window=new_high_days).max()
    new_high_condition = adj_close >= high_n


    # 技術面
    technical_buy_condition = (
        ma_up_buy_condition & 
        # ma5_above_others_condition &
        price_above_ma_buy_condition & 
        bias_buy_condition & 
        volume_doubled_condition & 
        # positive_close_condition &
        volume_above_500_condition &
        price_above_12_condition &
        amount_condition &

        dmi_buy_condition & 
        kd_buy_condition & 
        macd_dif_buy_condition &
        new_high_condition
    )
    
    return {
        'technical_buy_condition': technical_buy_condition,
        'ma_up_buy_condition': ma_up_buy_condition,
        'price_above_ma_buy_condition': price_above_ma_buy_condition,
        'bias_buy_condition': bias_buy_condition,
        'volume_doubled_condition': volume_doubled_condition,
        'volume_above_500_condition': volume_above_500_condition,
        'price_above_12_condition': price_above_12_condition,
        'amount_condition': amount_condition,
        'dmi_buy_condition': dmi_buy_condition,
        'kd_buy_condition': kd_buy_condition,
        'macd_dif_buy_condition': macd_dif_buy_condition,
        'new_high_condition': new_high_condition,

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
            'high_n': high_n,
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


# 最終的買入訊號 - A|B|C 組合
buy_signal = (
    # A: top_n=20, 營益率 0.1%, BIAS: 3~13, 5~16, 8~19, 8~20, 5~26, 8~26
    (build_chip_buy_condition(top_n=20)['chip_buy_condition'] &
     build_technical_buy_condition(
         bias_5_range=(0.03, 0.13),
         bias_10_range=(0.05, 0.16),
         bias_20_range=(0.08, 0.19),
         bias_60_range=(0.08, 0.20),
         bias_120_range=(0.05, 0.26),
         bias_240_range=(0.08, 0.26)
     )['technical_buy_condition'] &
     build_fundamental_buy_condition(1.001)['fundamental_buy_condition']) 
     
    |

    # # B: top_n=40, 營益率 25%, BIAS: 3~13, 5~16, 8~21, 8~20, 8~26, 8~26
    # (build_chip_buy_condition(top_n=40)['chip_buy_condition'] &
    #  build_technical_buy_condition(
    #      bias_5_range=(0.03, 0.13),
    #      bias_10_range=(0.05, 0.16),
    #      bias_20_range=(0.08, 0.21),
    #      bias_60_range=(0.08, 0.20),
    #      bias_120_range=(0.08, 0.26),
    #      bias_240_range=(0.08, 0.26)
    #  )['technical_buy_condition'] &
    #  build_fundamental_buy_condition(1.25)['fundamental_buy_condition']) 
     
    # |

    # C: top_n=25, 營益率 12.5%, BIAS: 3~13, 5~16, 8~19, 8~20, 5~27, 8~31
    (build_chip_buy_condition(top_n=25)['chip_buy_condition'] &
     build_technical_buy_condition(
         bias_5_range=(0.03, 0.13),
         bias_10_range=(0.05, 0.16),
         bias_20_range=(0.08, 0.19),
         bias_60_range=(0.08, 0.20),
         bias_120_range=(0.05, 0.27),
         bias_240_range=(0.08, 0.31)
     )['technical_buy_condition'] &
     build_fundamental_buy_condition(1.125)['fundamental_buy_condition'])

    |

    # E: top_n=40, 營益率 12.5%, BIAS: 3~13, 5~16, 8~19, 8~20, 5~35, 8~35, 創480天新高
    (build_chip_buy_condition(top_n=40)['chip_buy_condition'] &
     build_technical_buy_condition(
         bias_5_range=(0.03, 0.13),
         bias_10_range=(0.05, 0.16),
         bias_20_range=(0.08, 0.19),
         bias_60_range=(0.08, 0.20),
         bias_120_range=(0.05, 0.35),
         bias_240_range=(0.08, 0.35),
         new_high_days=480
     )['technical_buy_condition'] &
     build_fundamental_buy_condition(1.125)['fundamental_buy_condition'])
)

# 設定起始買入日期
start_buy_date = '2017-12-31'
buy_signal = buy_signal.loc[start_buy_date:]

def build_sell_condition():
    ma3 = adj_close.rolling(3).mean()
    dif, macd , _  = data.indicator('MACD', fastperiod=12, slowperiod=26, signalperiod=9, adjust_price=True)

    # 法一: 短線出場
    sell_condition = (ma3 < ma3.shift(1)) & (dif < dif.shift(1))

    return sell_condition

sell_condition = build_sell_condition()
position = buy_signal.hold_until(sell_condition)


# 執行回測
from finlab.backtest import sim
report = sim(position, resample=None, upload=False, market=AdjustTWMarketInfo())
# report = sim(position, resample=None, upload=False, trade_at_price='open')
# report = sim(position, resample=None, upload=False, trade_at_price='open', position_limit=0.25)

# fee_ratio = 0.001425
# tax_ratio = 0.003
# slippage = 0.01
# report = sim(
#     position, 
#     resample=None, 
#     upload=False, 
#     market=AdjustTWMarketInfo(), 
#     fee_ratio=slippage + fee_ratio,  # 買賣各扣: 滑價 2.5% + 手續費 0.1425%
#     tax_ratio=tax_ratio,  # 賣出時扣: 交易稅 0.3% 
#     position_limit=0.25
#     )

# 查看回測結果

# 取得所有績效指標
metrics = report.get_metrics()
# 年化報酬
annual_return = metrics['profitability']['annualReturn']
# 最大回檔
max_drawdown = metrics['risk']['maxDrawdown']
# 總交易次數
total_trades = report.get_trades().shape[0]
# 打印結果
print("=" * 50)
print("策略績效指標")
print("=" * 50)
print(f"年化報酬率: {annual_return:.2%}")
print(f"最大回檔: {max_drawdown:.2%}")
print(f"總交易次數: {total_trades} 筆")
print("=" * 50)

# -- 

from strategy_diagnostics import diagnose_strategy
from bias_analysis import create_bias_analyzer


def run_diagnosis(target_stocks, analysis_days, start_date, fundamental_quarter=None):
    """運行策略診斷的包裝函數"""
    
    print("🚀 開始診斷...")
    print("📊 計算籌碼面條件...")
    chip_conditions = build_chip_buy_condition(20)
    
    print("📊 計算技術面條件...")
    tech_conditions = build_technical_buy_condition()
    
    print("📊 計算基本面條件...")
    fund_conditions = build_fundamental_buy_condition(1.001)
    
    # 調用獨立的診斷函數
    diagnose_strategy(
        target_stocks=target_stocks,
        analysis_days=analysis_days,
        chip_conditions=chip_conditions,
        tech_conditions=tech_conditions,
        fund_conditions=fund_conditions,
        start_date=start_date,
        fundamental_quarter=fundamental_quarter
    )


def run_bias_analysis(report):
    """
    執行BIAS分析

    Parameters:
    -----------
    report : backtest report
        回測報告

    """
    print("\n🔍 開始進行 BIAS 分析...")

    # 獲取交易數據
    trades = report.get_trades()

    print("📊 算技術面條件...")
    tech_conditions = build_technical_buy_condition()

    # 提取bias數據
    bias_dict = tech_conditions['bias_values']

    # 執行分析
    analyzer = create_bias_analyzer()
    results = analyzer.analyze_all_bias(bias_dict, trades)

    return {
        'analysis_results': results,
        'trades_data': trades,
        'bias_data': bias_dict
    }


# if __name__ == "__main__":
#     # 基本的策略診斷
#     run_diagnosis(['8081'], analysis_days=10, start_date='2025-08-10', fundamental_quarter='2025-Q2')
#     run_diagnosis(['2402'], analysis_days=10, start_date='2025-08-07', fundamental_quarter='2025-Q2')

#     # 執行BIAS分析
#     bias_results = run_bias_analysis(report)