"""
Alan TW Strategy Base Class
所有 Alan 策略的基底類別，包含共用的籌碼面、技術面、基本面條件邏輯

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


class AlanTWStrategyBase:
    """
    Alan TW Strategy 基底類別
        - get_strategy_configs(): 返回策略參數配置列表
        - get_strategy_name(): 返回策略名稱

    Attributes:
        report: 回測報告物件
        position: 持倉訊號
        buy_signal: 買入訊號
        sell_signal: 賣出訊號
    """

    def __init__(self):
        """初始化策略參數"""
        # 回測參數
        self.start_date = '2017-12-31'
        self.slippage = 0.0
        self.position_limit = 0.25
        self.min_amount = 30000000  # 最小成交金額

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
                                       new_high_days=120, new_high_pct=1.0, min_amount=None):
        """建立技術面條件"""
        # Validate new_high_pct to avoid degenerate or unsafe signal behavior
        if not (0 < new_high_pct <= 1.0):
            raise ValueError(
                f"new_high_pct must be in the range (0, 1.0], got {new_high_pct!r}"
            )
        if min_amount is None:
            min_amount = self.min_amount

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
        amount_condition = (self.close * self.volume) > min_amount

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

        # 創新高 (支援百分比，如 0.95 代表 95% 新高)
        high_n = self.adj_close.rolling(window=new_high_days).max()
        new_high_condition = self.adj_close >= (high_n * new_high_pct)

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

    def _build_single_strategy_signal(self, config):
        """
        根據單一策略配置建立買入訊號

        Args:
            config: 策略配置字典，包含:
                - name: 策略名稱
                - top_n: 籌碼面排名
                - op_growth: 營益率成長門檻
                - bias_ranges: 乖離率範圍 dict
                - new_high_days: 創新高天數
                - new_high_pct: 創新高百分比 (預設 1.0，即 100%)

        Returns:
            buy_signal: 買入訊號 DataFrame
        """
        chip_condition = self._build_chip_buy_condition(top_n=config['top_n'])
        technical_condition = self._build_technical_buy_condition(
            bias_5_range=config['bias_ranges']['bias_5'],
            bias_10_range=config['bias_ranges']['bias_10'],
            bias_20_range=config['bias_ranges']['bias_20'],
            bias_60_range=config['bias_ranges']['bias_60'],
            bias_120_range=config['bias_ranges']['bias_120'],
            bias_240_range=config['bias_ranges']['bias_240'],
            new_high_days=config['new_high_days'],
            new_high_pct=config.get('new_high_pct', 1.0)
        )
        fundamental_condition = self._build_fundamental_buy_condition(config['op_growth'])

        return chip_condition & technical_condition & fundamental_condition

    def get_strategy_configs(self):
        """
        返回策略參數配置列表 (子類別需實作)

        Returns:
            list: 策略配置列表，每個元素為 dict
        """
        raise NotImplementedError("子類別需實作 get_strategy_configs()")

    def get_strategy_name(self):
        """
        返回策略名稱 (子類別需實作)

        Returns:
            str: 策略名稱
        """
        raise NotImplementedError("子類別需實作 get_strategy_name()")

    def run_strategy(self):
        """
        執行策略回測

        Returns:
            report: 回測報告物件
        """
        strategy_name = self.get_strategy_name()
        configs = self.get_strategy_configs()

        print(f"🚀 開始運行策略 {strategy_name}...")

        # 建立各子策略買入訊號並組合
        combined_signal = None
        for config in configs:
            print(f"📊 計算策略 {config['name']} 條件...")
            signal = self._build_single_strategy_signal(config)

            if combined_signal is None:
                combined_signal = signal
            else:
                combined_signal = combined_signal | signal

        self.buy_signal = combined_signal

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

        strategy_name = self.get_strategy_name()
        configs = self.get_strategy_configs()

        metrics = self.report.get_metrics()
        annual_return = metrics['profitability']['annualReturn']
        max_drawdown = metrics['risk']['maxDrawdown']
        total_trades = self.report.get_trades().shape[0]

        print("=" * 50)
        print(f"策略績效指標 (Strategy {strategy_name})")
        print("=" * 50)
        print(f"年化報酬率: {annual_return:.2%}")
        print(f"最大回檔: {max_drawdown:.2%}")
        print(f"總交易次數: {total_trades} 筆")
        print(f"滑價成本: {self.slippage:.2%}")
        if self.position_limit:
            print(f"單檔持股上限: {self.position_limit:.1%}")
        print("=" * 50)
        print(f"策略組合: {' | '.join([c['name'] for c in configs])}")
        for config in configs:
            new_high_pct = config.get('new_high_pct', 1.0)
            new_high_str = f"創{config['new_high_days']}天新高"
            if new_high_pct < 1.0:
                new_high_str += f"*{new_high_pct:.0%}"
            print(f"  - 策略 {config['name']}: top_n={config['top_n']}, "
                  f"營益率 {(config['op_growth']-1)*100:.1f}%, "
                  f"{new_high_str}")
        print("=" * 50)

    def get_report(self):
        """
        取得回測報告

        Returns:
            report: 回測報告物件，若未運行策略則返回提示訊息
        """
        return self.report if self.report else "report物件為空，請先運行策略"
