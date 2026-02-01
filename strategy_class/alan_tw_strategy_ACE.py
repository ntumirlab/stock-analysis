"""
Alan TW Strategy ACE (Combined)
策略 ACE: 組合 A、C、E 三個策略條件

此策略為三個子策略的組合 (A|C|E)：
- 策略 A: top_n=20, 營益率 0.1%, BIAS: 3~13, 5~16, 8~19, 8~20, 5~26, 8~26, 創120天新高
- 策略 C: top_n=25, 營益率 12.5%, BIAS: 3~13, 5~16, 8~19, 8~20, 5~27, 8~31, 創120天新高
- 策略 E: top_n=40, 營益率 12.5%, BIAS: 3~13, 5~16, 8~19, 8~20, 5~35, 8~35, 創480天新高

每個策略都結合三大面向：
- 籌碼面：三大法人與主力買賣超
- 技術面：均線、乖離率、DMI、KD、MACD等指標
- 基本面：營業利益率成長
"""

from finlab import data
from finlab.markets.tw import TWMarket
from finlab.backtest import sim
import pandas as pd
import numpy as np
from .taiwan_kd import taiwan_kd_fast


class AdjustTWMarketInfo(TWMarket):
    """自訂市場資訊類別，用於調整交易價格"""
    def get_trading_price(self, name, adj=True):
        return self.get_price(name, adj=adj).shift(1)


class AlanTWStrategyACE:
    """
    Alan TW Strategy ACE - 策略 A|C|E 組合

    Attributes:
        report: 回測報告物件
        position: 持倉訊號
        buy_signal: 買入訊號
        sell_signal: 賣出訊號
    """

    def __init__(self):
        """
        初始化策略參數

        注意：所有參數都在此處硬編碼，若需調整請直接修改此處數值
        """
        # 回測參數
        self.start_date = '2017-12-31'
        self.slippage = 0.0
        self.position_limit = 0.25

        self.report = None
        self.position = None
        self.buy_signal = None
        self.sell_signal = None

        # 載入數據
        self._load_data()

    def _load_data(self):
        """載入所需數據"""
        with data.universe(market='TSE_OTC'):
            # 籌碼面數據
            self.foreign_net_buy_shares = data.get('institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)')
            self.investment_trust_net_buy_shares = data.get('institutional_investors_trading_summary:投信買賣超股數')
            self.dealer_self_net_buy_shares = data.get('institutional_investors_trading_summary:自營商買賣超股數(自行買賣)')
            self.shares_outstanding = data.get('internal_equity_changes:發行股數')

            # 價格與技術指標數據
            self.close = data.get("price:收盤價")
            self.adj_close = data.get('etl:adj_close')
            self.adj_open = data.get('etl:adj_open')
            self.adj_high = data.get('etl:adj_high')
            self.adj_low = data.get('etl:adj_low')
            self.volume = data.get('price:成交股數')

            # 基本面數據
            self.operating_margin = data.get('fundamental_features:營業利益率')

    def _build_chip_buy_condition(self, top_n):
        """建立籌碼面條件"""
        # 計算外資、投信、自營商的買賣超佔發行量比例
        foreign_net_buy_ratio = self.foreign_net_buy_shares / self.shares_outstanding
        investment_trust_net_buy_ratio = self.investment_trust_net_buy_shares / self.shares_outstanding
        dealer_self_net_buy_ratio = self.dealer_self_net_buy_shares / self.shares_outstanding

        # 計算累積買超比例
        foreign_net_buy_ratio_2d_sum = foreign_net_buy_ratio.rolling(2).sum()
        foreign_net_buy_ratio_3d_sum = foreign_net_buy_ratio.rolling(3).sum()
        investment_trust_net_buy_ratio_2d_sum = investment_trust_net_buy_ratio.rolling(2).sum()
        investment_trust_net_buy_ratio_3d_sum = investment_trust_net_buy_ratio.rolling(3).sum()
        dealer_self_net_buy_ratio_2d_sum = dealer_self_net_buy_ratio.rolling(2).sum()
        dealer_self_net_buy_ratio_3d_sum = dealer_self_net_buy_ratio.rolling(3).sum()

        # 外資條件
        foreign_top_1d_ratio = foreign_net_buy_ratio.rank(axis=1, ascending=False) <= top_n
        foreign_top_2d_ratio = foreign_net_buy_ratio_2d_sum.rank(axis=1, ascending=False) <= top_n
        foreign_top_3d_ratio = foreign_net_buy_ratio_3d_sum.rank(axis=1, ascending=False) <= top_n
        foreign_buy_condition = foreign_top_1d_ratio | foreign_top_2d_ratio | foreign_top_3d_ratio

        # 投信條件
        investment_trust_top_1d_ratio = investment_trust_net_buy_ratio.rank(axis=1, ascending=False) <= top_n
        investment_trust_top_2d_ratio = investment_trust_net_buy_ratio_2d_sum.rank(axis=1, ascending=False) <= top_n
        investment_trust_top_3d_ratio = investment_trust_net_buy_ratio_3d_sum.rank(axis=1, ascending=False) <= top_n
        investment_trust_buy_condition = investment_trust_top_1d_ratio | investment_trust_top_2d_ratio | investment_trust_top_3d_ratio

        # 自營商條件
        dealer_self_top_1d_ratio = dealer_self_net_buy_ratio.rank(axis=1, ascending=False) <= top_n
        dealer_self_top_2d_ratio = dealer_self_net_buy_ratio_2d_sum.rank(axis=1, ascending=False) <= top_n
        dealer_self_top_3d_ratio = dealer_self_net_buy_ratio_3d_sum.rank(axis=1, ascending=False) <= top_n
        dealer_self_buy_condition = dealer_self_top_1d_ratio | dealer_self_top_2d_ratio | dealer_self_top_3d_ratio

        # 主力籌碼數據
        with data.universe(market='TSE_OTC'):
            top15_buy_shares = data.get('etl:broker_transactions:top15_buy')
            top15_sell_shares = data.get('etl:broker_transactions:top15_sell')

        net_buy_shares = (top15_buy_shares - top15_sell_shares) * 1000
        net_buy_ratio = net_buy_shares / self.shares_outstanding
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
            (main_force_top_1d_buy & main_force_condition_1d) |
            (main_force_top_2d_buy & main_force_condition_2d) |
            (main_force_top_3d_buy & main_force_condition_3d)
        )

        chip_buy_condition = foreign_buy_condition | dealer_self_buy_condition | main_force_buy_condition

        return chip_buy_condition

    def _build_technical_buy_condition(self, bias_5_range, bias_10_range, bias_20_range,
                                       bias_60_range, bias_120_range, bias_240_range,
                                       new_high_days=120):
        """建立技術面條件"""
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
            (ma5 > ma5.shift(1)) & (ma10 > ma10.shift(1)) &
            (ma20 > ma20.shift(1)) & (ma60 > ma60.shift(1))
        )

        # 價格在均線之上
        price_above_ma_buy_condition = (
            (self.adj_close > ma5) & (self.adj_close > ma10) &
            (self.adj_close > ma20) & (self.adj_close > ma60)
        )

        # 計算乖離率
        bias_5 = (self.adj_close - ma5) / ma5
        bias_10 = (self.adj_close - ma10) / ma10
        bias_20 = (self.adj_close - ma20) / ma20
        bias_60 = (self.adj_close - ma60) / ma60
        bias_120 = (self.adj_close - ma120) / ma120
        bias_240 = (self.adj_close - ma240) / ma240

        bias_5_condition = (bias_5 >= bias_5_range[0]) & (bias_5 <= bias_5_range[1])
        bias_10_condition = (bias_10 >= bias_10_range[0]) & (bias_10 <= bias_10_range[1])
        bias_20_condition = (bias_20 >= bias_20_range[0]) & (bias_20 <= bias_20_range[1])
        bias_60_condition = (bias_60 >= bias_60_range[0]) & (bias_60 <= bias_60_range[1])
        bias_120_condition = (bias_120 >= bias_120_range[0]) & (bias_120 <= bias_120_range[1])
        bias_240_condition = (bias_240 >= bias_240_range[0]) & (bias_240 <= bias_240_range[1])

        bias_buy_condition = (
            bias_5_condition & bias_10_condition & bias_20_condition &
            bias_60_condition & bias_120_condition & bias_240_condition
        )

        # 價格與成交量條件
        price_above_12_condition = self.close > 12
        volume_doubled_condition = self.volume > (self.volume.shift(1) * 2)
        volume_above_500_condition = self.volume > 500 * 1000
        amount_condition = (self.close * self.volume) > 30000000

        # DMI指標
        with data.universe(market='TSE_OTC'):
            plus_di = data.indicator('PLUS_DI', timeperiod=14, adjust_price=True)
            minus_di = data.indicator('MINUS_DI', timeperiod=14, adjust_price=True)

        dmi_buy_condition = (plus_di > 24) & (minus_di < 21)

        # KD指標
        k, d = taiwan_kd_fast(
            high_df=self.adj_high,
            low_df=self.adj_low,
            close_df=self.adj_close,
            fastk_period=9,
            alpha=1/3
        )

        k_up_condition = k > k.shift(1)
        d_up_condition = d > d.shift(1)
        kd_buy_condition = k_up_condition & d_up_condition

        # MACD指標
        with data.universe(market='TSE_OTC'):
            dif, macd, _ = data.indicator('MACD', fastperiod=12, slowperiod=26, signalperiod=9, adjust_price=True)

        macd_dif_buy_condition = dif > dif.shift(1)

        # 創新高
        high_n = self.adj_close.rolling(window=new_high_days).max()
        new_high_condition = self.adj_close >= high_n

        # 技術面綜合條件
        technical_buy_condition = (
            ma_up_buy_condition &
            price_above_ma_buy_condition &
            bias_buy_condition &
            volume_doubled_condition &
            volume_above_500_condition &
            price_above_12_condition &
            amount_condition &
            dmi_buy_condition &
            kd_buy_condition &
            macd_dif_buy_condition &
            new_high_condition
        )

        return technical_buy_condition

    def _build_fundamental_buy_condition(self, op_growth_threshold):
        """建立基本面條件"""
        operating_margin_increase = (
            self.operating_margin > (self.operating_margin.shift(1) * op_growth_threshold)
        )

        return operating_margin_increase

    def _build_sell_condition(self):
        """建立賣出條件"""
        ma3 = self.adj_close.rolling(3).mean()

        with data.universe(market='TSE_OTC'):
            dif, macd, _ = data.indicator('MACD', fastperiod=12, slowperiod=26, signalperiod=9, adjust_price=True)

        # 短線出場
        sell_condition = (ma3 < ma3.shift(1)) & (dif < dif.shift(1))

        return sell_condition

    def run_strategy(self):
        """
        執行策略回測

        Returns:
            report: 回測報告物件
        """
        print("🚀 開始運行策略 ACE (A|C|E 組合)...")

        # 策略 A: top_n=20, 營益率 0.1%, BIAS: 3~13, 5~16, 8~19, 8~20, 5~26, 8~26
        print("📊 計算策略 A 條件...")
        chip_buy_condition_A = self._build_chip_buy_condition(top_n=20)
        technical_buy_condition_A = self._build_technical_buy_condition(
            bias_5_range=(0.03, 0.13),
            bias_10_range=(0.05, 0.16),
            bias_20_range=(0.08, 0.19),
            bias_60_range=(0.08, 0.20),
            bias_120_range=(0.05, 0.26),
            bias_240_range=(0.08, 0.26),
            new_high_days=120
        )
        fundamental_buy_condition_A = self._build_fundamental_buy_condition(1.001)

        buy_signal_A = (
            chip_buy_condition_A &
            technical_buy_condition_A &
            fundamental_buy_condition_A
        )

        # 策略 C: top_n=25, 營益率 12.5%, BIAS: 3~13, 5~16, 8~19, 8~20, 5~27, 8~31
        print("📊 計算策略 C 條件...")
        chip_buy_condition_C = self._build_chip_buy_condition(top_n=25)
        technical_buy_condition_C = self._build_technical_buy_condition(
            bias_5_range=(0.03, 0.13),
            bias_10_range=(0.05, 0.16),
            bias_20_range=(0.08, 0.19),
            bias_60_range=(0.08, 0.20),
            bias_120_range=(0.05, 0.27),
            bias_240_range=(0.08, 0.31),
            new_high_days=120
        )
        fundamental_buy_condition_C = self._build_fundamental_buy_condition(1.125)

        buy_signal_C = (
            chip_buy_condition_C &
            technical_buy_condition_C &
            fundamental_buy_condition_C
        )

        # 策略 E: top_n=40, 營益率 12.5%, BIAS: 3~13, 5~16, 8~19, 8~20, 5~35, 8~35, 創480天新高
        print("📊 計算策略 E 條件...")
        chip_buy_condition_E = self._build_chip_buy_condition(top_n=40)
        technical_buy_condition_E = self._build_technical_buy_condition(
            bias_5_range=(0.03, 0.13),
            bias_10_range=(0.05, 0.16),
            bias_20_range=(0.08, 0.19),
            bias_60_range=(0.08, 0.20),
            bias_120_range=(0.05, 0.35),
            bias_240_range=(0.08, 0.35),
            new_high_days=480
        )
        fundamental_buy_condition_E = self._build_fundamental_buy_condition(1.125)

        buy_signal_E = (
            chip_buy_condition_E &
            technical_buy_condition_E &
            fundamental_buy_condition_E
        )

        # 組合買入訊號: A | C | E
        print("📊 組合 A|C|E 策略...")
        self.buy_signal = buy_signal_A | buy_signal_C | buy_signal_E

        # 設定起始日期
        self.buy_signal = self.buy_signal.loc[self.start_date:]

        # 建立賣出條件
        print("📊 計算賣出條件...")
        self.sell_signal = self._build_sell_condition()

        # 建立持倉訊號
        self.position = self.buy_signal.hold_until(self.sell_signal)

        # 執行回測
        print("🔄 執行回測...")
        fee_ratio = 0.001425
        tax_ratio = 0.003

        sim_params = {
            'resample': None,
            'upload': False,
            'market': AdjustTWMarketInfo(),
            'fee_ratio': self.slippage + fee_ratio,
            'tax_ratio': tax_ratio
        }

        if self.position_limit is not None:
            sim_params['position_limit'] = self.position_limit

        self.report = sim(self.position, **sim_params)

        # 打印結果
        self._print_metrics()

        return self.report

    def _print_metrics(self):
        """打印策略績效指標"""
        if self.report is None:
            print("報告物件為空，請先運行策略")
            return

        metrics = self.report.get_metrics()
        annual_return = metrics['profitability']['annualReturn']
        max_drawdown = metrics['risk']['maxDrawdown']
        total_trades = self.report.get_trades().shape[0]

        print("=" * 50)
        print("策略績效指標 (Strategy ACE)")
        print("=" * 50)
        print(f"年化報酬率: {annual_return:.2%}")
        print(f"最大回檔: {max_drawdown:.2%}")
        print(f"總交易次數: {total_trades} 筆")
        print(f"滑價成本: {self.slippage:.2%}")
        if self.position_limit:
            print(f"單檔持股上限: {self.position_limit:.1%}")
        print("=" * 50)
        print("策略組合: A | C | E")
        print("  - 策略 A: top_n=20, 營益率 0.1%, 創120天新高")
        print("  - 策略 C: top_n=25, 營益率 12.5%, 創120天新高")
        print("  - 策略 E: top_n=40, 營益率 12.5%, 創480天新高")
        print("=" * 50)

    def get_report(self):
        """
        取得回測報告

        Returns:
            report: 回測報告物件，若未運行策略則返回提示訊息
        """
        return self.report if self.report else "report物件為空，請先運行策略"


# Example usage:
if __name__ == "__main__":
    strategy = AlanTWStrategyACE()
    report = strategy.run_strategy()
