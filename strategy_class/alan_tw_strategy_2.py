from finlab import data
from finlab.markets.tw import TWMarket
from finlab.backtest import sim
from .taiwan_kd import taiwan_kd_fast


class AdjustTWMarketInfo(TWMarket):
    def get_trading_price(self, name, adj=True):
        return self.get_price(name, adj=adj).shift(1)


class AlanTWStrategy2:
    def __init__(self):
        self.report = None
        # 在初始化時載入所有需要的數據
        with data.universe(market='TSE_OTC'):
            # 基本價格數據
            self.close = data.get("price:收盤價")
            self.adj_close = data.get('etl:adj_close')
            self.adj_open = data.get('etl:adj_open')
            self.adj_high = data.get('etl:adj_high')
            self.adj_low = data.get('etl:adj_low')
            self.volume = data.get('price:成交股數')

            # 籌碼面數據
            self.foreign_net_buy_shares = data.get(
                'institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)'
            )
            self.investment_trust_net_buy_shares = data.get('institutional_investors_trading_summary:投信買賣超股數')
            self.dealer_self_net_buy_shares = data.get(
                'institutional_investors_trading_summary:自營商買賣超股數(自行買賣)'
            )
            self.shares_outstanding = data.get('internal_equity_changes:發行股數')
            self.top15_buy_shares = data.get('etl:broker_transactions:top15_buy')
            self.top15_sell_shares = data.get('etl:broker_transactions:top15_sell')

            # 基本面數據
            self.operating_margin = data.get('fundamental_features:營業利益率')

    def build_chip_buy_condition(self, top_n):
        """籌碼面買入條件 - 完全採用model2.py的邏輯"""

        # 計算外資、投信、自營商的買賣超佔發行量比例 (股數)
        foreign_net_buy_ratio = self.foreign_net_buy_shares / self.shares_outstanding
        investment_trust_net_buy_ratio = self.investment_trust_net_buy_shares / self.shares_outstanding
        dealer_self_net_buy_ratio = self.dealer_self_net_buy_shares / self.shares_outstanding

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
        investment_trust_buy_condition = (
            investment_trust_top_1d_ratio | investment_trust_top_2d_ratio | investment_trust_top_3d_ratio
        )

        # 自營商：取當天、前2天、前3天累積買超比例前幾
        dealer_self_top_1d_ratio = dealer_self_net_buy_ratio.rank(axis=1, ascending=False) <= top_n
        dealer_self_top_2d_ratio = dealer_self_net_buy_ratio_2d_sum.rank(axis=1, ascending=False) <= top_n
        dealer_self_top_3d_ratio = dealer_self_net_buy_ratio_3d_sum.rank(axis=1, ascending=False) <= top_n
        dealer_self_buy_condition = dealer_self_top_1d_ratio | dealer_self_top_2d_ratio | dealer_self_top_3d_ratio

        # 計算買賣超差額股數
        net_buy_shares = (self.top15_buy_shares - self.top15_sell_shares) * 1000

        # 買賣超差額股數佔發行股數的比例
        net_buy_ratio = net_buy_shares / self.shares_outstanding

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

        main_force_buy_condition = (
            (main_force_top_1d_buy & main_force_condition_1d)
            | (main_force_top_2d_buy & main_force_condition_2d)
            | (main_force_top_3d_buy & main_force_condition_3d)
        )

        chip_buy_condition = foreign_buy_condition | dealer_self_buy_condition | main_force_buy_condition

        return {
            'chip_buy_condition': chip_buy_condition,
            'foreign_buy_condition': foreign_buy_condition,
            'investment_trust_buy_condition': investment_trust_buy_condition,
            'dealer_self_buy_condition': dealer_self_buy_condition,
            'main_force_buy_condition': main_force_buy_condition,
        }

    def build_technical_buy_condition(self):
        """技術面買入條件 - 完全採用model2.py的邏輯，包含taiwan_kd"""

        # 計算均線
        ma3 = self.adj_close.rolling(3).mean()
        ma5 = self.adj_close.rolling(5).mean()
        ma10 = self.adj_close.rolling(10).mean()
        ma20 = self.adj_close.rolling(20).mean()
        ma60 = self.adj_close.rolling(60).mean()
        ma120 = self.adj_close.rolling(120).mean()
        ma240 = self.adj_close.rolling(240).mean()

        # 均線上升
        ma_up_buy_condition = (
            (ma5 > ma5.shift(1)) & (ma10 > ma10.shift(1)) & (ma20 > ma20.shift(1)) & (ma60 > ma60.shift(1))
        )

        # 5 日線大於 60/240 日線
        ma5_above_others_condition = (ma5 > ma60) & (ma5 > ma240)

        # 價格在均線之上
        price_above_ma_buy_condition = (
            (self.adj_close > ma5) & (self.adj_close > ma10) & (self.adj_close > ma20) & (self.adj_close > ma60)
        )

        # 計算乖離率 - 使用model2.py的參數
        bias_5 = (self.adj_close - ma5) / ma5
        bias_10 = (self.adj_close - ma10) / ma10
        bias_20 = (self.adj_close - ma20) / ma20
        bias_60 = (self.adj_close - ma60) / ma60
        bias_120 = (self.adj_close - ma120) / ma120
        bias_240 = (self.adj_close - ma240) / ma240

        bias_5_condition = (bias_5 <= 0.12) & (bias_5 >= 0.02)
        bias_10_condition = (bias_10 <= 0.15) & (bias_10 >= 0.05)
        bias_20_condition = (bias_20 <= 0.20) & (bias_20 >= 0.05)
        bias_60_condition = (bias_60 <= 0.20) & (bias_60 >= 0.05)
        bias_120_condition = (bias_120 <= 0.25) & (bias_120 >= 0.10)
        bias_240_condition = (bias_240 <= 0.25) & (bias_240 >= 0.10)

        # 設定進場乖離率
        bias_buy_condition = (
            bias_5_condition
            & bias_10_condition
            & bias_20_condition
            & bias_60_condition
            & bias_120_condition
            & bias_240_condition
        )

        # 今收盤 > 今開盤，且今收盤 > 昨收盤
        positive_close_condition = (self.adj_close > self.adj_open) & (self.adj_close > self.adj_close.shift(1))

        price_above_12_condition = self.close > 12

        # 成交量大於昨日的2倍
        volume_doubled_condition = self.volume > (self.volume.shift(1) * 2)

        # 今日成交張數 > 500 張
        volume_above_500_condition = self.volume > 500 * 1000

        # 成交金額大於 3000 萬
        amount_condition = (self.close * self.volume) > 30000000

        with data.universe(market='TSE_OTC'):
            # 計算DMI指標
            plus_di = data.indicator('PLUS_DI', timeperiod=14, adjust_price=True)
            minus_di = data.indicator('MINUS_DI', timeperiod=14, adjust_price=True)

        # DMI條件
        dmi_buy_condition = (plus_di > 24) & (minus_di < 21)

        # 使用台灣標準KD指標 (taiwan_kd_fast) - 完全採用model2.py邏輯
        k, d = taiwan_kd_fast(
            high_df=self.adj_high, low_df=self.adj_low, close_df=self.adj_close, fastk_period=9, alpha=1 / 3
        )

        # KD 指標條件：%K 和 %D 都向上
        k_up_condition = k > k.shift(1)
        d_up_condition = d > d.shift(1)
        kd_buy_condition = k_up_condition & d_up_condition

        with data.universe(market='TSE_OTC'):
            # 計算 MACD 指標
            dif, macd, _ = data.indicator('MACD', fastperiod=12, slowperiod=26, signalperiod=9, adjust_price=True)

        # MACD DIF 向上
        macd_dif_buy_condition = dif > dif.shift(1)

        # 創新高
        high_120 = self.adj_close.rolling(window=120).max()
        new_high_120_condition = self.adj_close >= high_120
        new_high_condition = new_high_120_condition

        # 技術面總條件 - 完全採用model2.py的邏輯
        technical_buy_condition = (
            ma_up_buy_condition
            &
            # ma5_above_others_condition &  # model2.py中被註解掉
            price_above_ma_buy_condition
            & bias_buy_condition
            & volume_doubled_condition
            &
            # positive_close_condition &    # model2.py中被註解掉
            volume_above_500_condition
            & price_above_12_condition
            & amount_condition
            & dmi_buy_condition
            & kd_buy_condition
            & macd_dif_buy_condition
            & new_high_condition
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
                'bias_240': bias_240,
            },
            'bias_conditions': {
                'bias_5_condition': bias_5_condition,
                'bias_10_condition': bias_10_condition,
                'bias_20_condition': bias_20_condition,
                'bias_60_condition': bias_60_condition,
                'bias_120_condition': bias_120_condition,
                'bias_240_condition': bias_240_condition,
            },
            'kd_values': {'k_value': k, 'd_value': d},
            'kd_conditions': {
                'k_up_condition': k_up_condition,
                'd_up_condition': d_up_condition,
                'kd_buy_condition': kd_buy_condition,
            },
            'price_data': {'adj_close': self.adj_close, 'high_120': high_120},
        }

    def build_fundamental_buy_condition(self, op_growth_threshold):
        """基本面買入條件 - 採用model2.py的邏輯"""

        operating_margin_increase = self.operating_margin > (self.operating_margin.shift(1) * op_growth_threshold)

        fundamental_buy_condition = operating_margin_increase

        return {
            'fundamental_buy_condition': fundamental_buy_condition,
            'operating_margin_increase': operating_margin_increase,
        }

    def build_sell_condition(self):
        """賣出條件 - 採用model2.py的邏輯"""
        ma3 = self.adj_close.rolling(3).mean()

        with data.universe(market='TSE_OTC'):
            dif, macd, _ = data.indicator('MACD', fastperiod=12, slowperiod=26, signalperiod=9, adjust_price=True)

        # 短線出場
        sell_condition = (ma3 < ma3.shift(1)) & (dif < dif.shift(1))

        return sell_condition

    def run_strategy(self, start_buy_date='2017-12-31'):
        """執行策略 - 採用model2.py的多層級策略組合"""

        print("🚀 開始執行 AlanTwStrategy2 策略...")
        print("📊 使用多層級策略組合 (model2.py邏輯)")

        # 計算各個條件
        chip_20 = self.build_chip_buy_condition(top_n=20)
        chip_60 = self.build_chip_buy_condition(top_n=60)
        chip_80 = self.build_chip_buy_condition(top_n=80)
        chip_100 = self.build_chip_buy_condition(top_n=100)

        tech_conditions = self.build_technical_buy_condition()

        fund_1001 = self.build_fundamental_buy_condition(1.001)
        fund_110 = self.build_fundamental_buy_condition(1.10)
        fund_120 = self.build_fundamental_buy_condition(1.20)
        fund_130 = self.build_fundamental_buy_condition(1.30)

        # 多層級策略組合 - 完全採用model2.py的邏輯
        buy_signal = (
            (
                chip_20['chip_buy_condition']
                & tech_conditions['technical_buy_condition']
                & fund_1001['fundamental_buy_condition']
            )
            | (
                chip_60['chip_buy_condition']
                & tech_conditions['technical_buy_condition']
                & fund_110['fundamental_buy_condition']
            )
            | (
                chip_80['chip_buy_condition']
                & tech_conditions['technical_buy_condition']
                & fund_120['fundamental_buy_condition']
            )
            | (
                chip_100['chip_buy_condition']
                & tech_conditions['technical_buy_condition']
                & fund_130['fundamental_buy_condition']
            )
        )

        # 設定起始買入日期
        buy_signal = buy_signal.loc[start_buy_date:]

        # 賣出條件
        sell_condition = self.build_sell_condition()
        position = buy_signal.hold_until(sell_condition)

        # 執行回測
        self.report = sim(position, resample=None, upload=False, market=AdjustTWMarketInfo())

        print("✅ 策略執行完成！")
        return self.report

    def get_report(self):
        """取得回測報告"""
        return self.report if self.report else "report物件為空，請先運行策略"


# 使用範例
if __name__ == "__main__":
    strategy = AlanTWStrategy2()
    report = strategy.run_strategy()
