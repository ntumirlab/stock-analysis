"""
Alan TW Strategy Not Start A
策略: 尋找尚未啟動的潛力股 (A版 - 調整條件定調)

此策略專注於尋找盤整已久、尚未大幅上漲的股票：
- 籌碼面：三大法人與主力買賣超前25名
- 技術面：乖離率限制（篩選盤整股）、價格接近120天收盤新高93%
- 基本面：營業利益率成長12.5%

A版參數：
- 買超前25檔
- 乖離率: -3~13%, -3~16%, -3~19%, -3~20%, -3~27%, 0~31%
- 120天收盤新高

賣出條件：
- 5日線乖離 < -4% 或
- 10日線乖離 < -4% 或
- 價格 < 120天收盤新高的91% 或
- 60日線乖離 > 21% 或
- 240日線乖離 > 32%
"""

from finlab import data
from finlab.markets.tw import TWMarket
from finlab.backtest import sim
import pandas as pd
import numpy as np


class AdjustTWMarketInfo(TWMarket):
    """自訂市場資訊類別，用於調整交易價格"""
    def get_trading_price(self, name, adj=True):
        return self.get_price(name, adj=adj).shift(1)


class AlanTWStrategyNotStartA:
    """
    Alan TW Strategy Not Start A - 尋找尚未啟動的潛力股 (A版)

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
        self.position_limit = None

        # A版策略參數
        self.top_n = 25
        self.op_growth_threshold = 1.125  # 營益率增12.5%
        self.new_high_days = 120  # A版: 120天

        # A版乖離率參數
        self.bias_5_range = (-0.03, 0.13)
        self.bias_10_range = (-0.03, 0.16)
        self.bias_20_range = (-0.03, 0.19)
        self.bias_60_range = (-0.03, 0.20)
        self.bias_120_range = (-0.03, 0.27)  # A版: -3~27%
        self.bias_240_range = (0.00, 0.31)   # A版: 0~31%

        # A版賣出乖離率參數
        self.sell_bias_240_threshold = 0.32  # A版: 32%

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
            # finlab 2.x 將此資料集載為 nullable Int64（pd.NA）：rank() 會 TypeError，
            # 且 NA 的比較語意與 NaN 不同（NA>x 回 NA 而非 False），統一轉回 float64
            top15_buy_shares = data.get('etl:broker_transactions:top15_buy').astype('float64')
            top15_sell_shares = data.get('etl:broker_transactions:top15_sell').astype('float64')

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

    def _build_technical_buy_condition(self):
        """建立技術面條件"""
        # 計算均線
        ma5 = self.adj_close.rolling(5).mean()
        ma10 = self.adj_close.rolling(10).mean()
        ma20 = self.adj_close.rolling(20).mean()
        ma60 = self.adj_close.rolling(60).mean()
        ma120 = self.adj_close.rolling(120).mean()
        ma240 = self.adj_close.rolling(240).mean()

        # 計算乖離率
        bias_5 = (self.adj_close - ma5) / ma5
        bias_10 = (self.adj_close - ma10) / ma10
        bias_20 = (self.adj_close - ma20) / ma20
        bias_60 = (self.adj_close - ma60) / ma60
        bias_120 = (self.adj_close - ma120) / ma120
        bias_240 = (self.adj_close - ma240) / ma240

        bias_5_condition = (bias_5 >= self.bias_5_range[0]) & (bias_5 <= self.bias_5_range[1])
        bias_10_condition = (bias_10 >= self.bias_10_range[0]) & (bias_10 <= self.bias_10_range[1])
        bias_20_condition = (bias_20 >= self.bias_20_range[0]) & (bias_20 <= self.bias_20_range[1])
        bias_60_condition = (bias_60 >= self.bias_60_range[0]) & (bias_60 <= self.bias_60_range[1])
        bias_120_condition = (bias_120 >= self.bias_120_range[0]) & (bias_120 <= self.bias_120_range[1])
        bias_240_condition = (bias_240 >= self.bias_240_range[0]) & (bias_240 <= self.bias_240_range[1])

        bias_buy_condition = (
            bias_5_condition & bias_10_condition & bias_20_condition &
            bias_60_condition & bias_120_condition & bias_240_condition
        )

        # 價格與成交量條件
        price_above_12_condition = self.close > 12
        volume_above_300_condition = self.volume > 300 * 1000
        amount_above_15m_condition = (self.close * self.volume) > 15000000

        # 計算收盤新高
        close_high = self.adj_close.rolling(window=self.new_high_days).max()

        # 價格在收盤新高的93%以上
        price_above_93pct_close_high_condition = self.adj_close >= (close_high * 0.93)

        # 技術面綜合條件
        technical_buy_condition = (
            bias_buy_condition &
            volume_above_300_condition &
            price_above_12_condition &
            amount_above_15m_condition &
            price_above_93pct_close_high_condition
        )

        return technical_buy_condition

    def _build_fundamental_buy_condition(self):
        """建立基本面條件"""
        operating_margin_increase = (
            self.operating_margin > (self.operating_margin.shift(1) * self.op_growth_threshold)
        )

        return operating_margin_increase

    def _build_sell_condition(self):
        """建立賣出條件"""
        # 計算均線
        ma5 = self.adj_close.rolling(5).mean()
        ma10 = self.adj_close.rolling(10).mean()
        ma60 = self.adj_close.rolling(60).mean()
        ma240 = self.adj_close.rolling(240).mean()

        # 計算乖離率
        bias_5 = (self.adj_close - ma5) / ma5
        bias_10 = (self.adj_close - ma10) / ma10
        bias_60 = (self.adj_close - ma60) / ma60
        bias_240 = (self.adj_close - ma240) / ma240

        # 計算收盤新高
        close_high = self.adj_close.rolling(window=self.new_high_days).max()

        # A版賣出條件:
        # 1. 5日線乖離小於-4% 或
        # 2. 10日線乖離小於-4% 或
        # 3. 價格小於120天收盤新高的91% 或
        # 4. 60日線乖離大於21% 或
        # 5. 240日線乖離大於32%
        sell_condition = (
            (bias_5 < -0.04) |
            (bias_10 < -0.04) |
            (self.adj_close < (close_high * 0.91)) |
            (bias_60 > 0.21) |
            (bias_240 > self.sell_bias_240_threshold)
        )

        return sell_condition

    def run_strategy(self):
        """
        執行策略回測

        Returns:
            report: 回測報告物件
        """
        print("🚀 開始運行策略 Not Start A (尋找尚未啟動的潛力股 - A版)...")

        # 建立買入條件
        print("📊 計算籌碼面條件...")
        chip_buy_condition = self._build_chip_buy_condition(top_n=self.top_n)

        print("📊 計算技術面條件...")
        technical_buy_condition = self._build_technical_buy_condition()

        print("📊 計算基本面條件...")
        fundamental_buy_condition = self._build_fundamental_buy_condition()

        # 組合買入訊號
        print("📊 組合買入條件...")
        self.buy_signal = (
            chip_buy_condition &
            technical_buy_condition &
            fundamental_buy_condition
        )

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
        print("策略績效指標 (Strategy Not Start A)")
        print("=" * 50)
        print(f"年化報酬率: {annual_return:.2%}")
        print(f"最大回檔: {max_drawdown:.2%}")
        print(f"總交易次數: {total_trades} 筆")
        print(f"滑價成本: {self.slippage:.2%}")
        if self.position_limit:
            print(f"單檔持股上限: {self.position_limit:.1%}")
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
    strategy = AlanTWStrategyNotStartA()
    report = strategy.run_strategy()
