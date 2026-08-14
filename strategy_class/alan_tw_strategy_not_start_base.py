"""
未發動股追蹤模型的基底類別。

與發動型 base 的差異：技術面不含量能翻倍、均線上升、價格在均線上、KD、MACD，
乖離下限可為負（容許盤整），並以「接近收盤新高」取代「創新高」。
各子策略的新高天數與乖離上限不同，故進出場皆各自獨立計算、持倉取聯集。

可調參數（以類別屬性覆寫）：
    entry_plus_di_min / entry_minus_di_max: 買進 DMI 門檻，None 表示不使用 DMI
    min_volume_shares / min_amount: 成交量與成交金額門檻
"""

from finlab import data
from finlab.backtest import sim

from .alan_tw_strategy_base import AlanTWStrategyBase, AdjustTWMarketInfo


class AlanTWStrategyNotStartBase(AlanTWStrategyBase):

    # 買進 DMI 門檻；None 表示不加此條件
    entry_plus_di_min = None
    entry_minus_di_max = None

    # 價量門檻（base.__init__ 會以實例屬性設定 min_amount，故於 __init__ 中覆寫）
    min_price = 12
    min_volume_shares = 300 * 1000
    min_amount = 15_000_000

    def __init__(self):
        super().__init__()
        self.position_limit = None
        self.min_amount = type(self).min_amount
        self.sub_buy_signals = {}
        self.sub_sell_signals = {}

    def _build_technical_buy_condition(self, config):
        """乖離區間 + 價量金額門檻 + 接近收盤新高（+ 選用的 DMI）"""
        ma = {n: self.adj_close.rolling(n).mean() for n in (5, 10, 20, 60, 120, 240)}
        bias = {f'bias_{n}': (self.adj_close - m) / m for n, m in ma.items()}

        condition = None
        for key, (lo, hi) in config['bias_ranges'].items():
            cond = (bias[key] >= lo) & (bias[key] <= hi)
            condition = cond if condition is None else (condition & cond)

        close_high = self.adj_close.rolling(window=config['new_high_days']).max()

        condition = (
            condition &
            (self.close > self.min_price) &
            (self.volume > self.min_volume_shares) &
            ((self.close * self.volume) > self.min_amount) &
            (self.adj_close >= close_high * config['entry_high_pct'])
        )

        if self.entry_plus_di_min is not None or self.entry_minus_di_max is not None:
            with data.universe(market='TSE_OTC'):
                plus_di = data.indicator('PLUS_DI', timeperiod=14, adjust_price=True)
                minus_di = data.indicator('MINUS_DI', timeperiod=14, adjust_price=True)
            if self.entry_plus_di_min is not None:
                condition = condition & (plus_di > self.entry_plus_di_min)
            if self.entry_minus_di_max is not None:
                condition = condition & (minus_di < self.entry_minus_di_max)

        return condition

    def _build_sell_condition_for(self, config):
        """5日或10日乖離 < -4%，或跌破新高門檻，或 20/60日乖離超過該子策略進場上限"""
        ma5 = self.adj_close.rolling(5).mean()
        ma10 = self.adj_close.rolling(10).mean()
        ma20 = self.adj_close.rolling(20).mean()
        ma60 = self.adj_close.rolling(60).mean()

        bias_5 = (self.adj_close - ma5) / ma5
        bias_10 = (self.adj_close - ma10) / ma10
        bias_20 = (self.adj_close - ma20) / ma20
        bias_60 = (self.adj_close - ma60) / ma60

        close_high = self.adj_close.rolling(window=config['new_high_days']).max()

        return (
            (bias_5 < -0.04) |
            (bias_10 < -0.04) |
            (self.adj_close < close_high * config['exit_high_pct']) |
            (bias_20 > config['bias_ranges']['bias_20'][1]) |
            (bias_60 > config['bias_ranges']['bias_60'][1])
        )

    def run_strategy(self):
        configs = self.get_strategy_configs()
        print(f"🚀 開始運行策略 {self.get_strategy_name()}...")

        assert len({c['top_n'] for c in configs}) == 1
        assert len({c['op_growth'] for c in configs}) == 1

        chip = self._build_chip_buy_condition(top_n=configs[0]['top_n'])
        fundamental = self._build_fundamental_buy_condition(configs[0]['op_growth'])

        combined_buy = None
        combined_position = None
        for config in configs:
            print(f"📊 計算子策略 {config['name']} ...")
            buy = (chip & self._build_technical_buy_condition(config) & fundamental)
            buy = buy.loc[self.start_date:]
            sell = self._build_sell_condition_for(config)

            self.sub_buy_signals[config['name']] = buy
            self.sub_sell_signals[config['name']] = sell

            position = buy.hold_until(sell)
            combined_buy = buy if combined_buy is None else (combined_buy | buy)
            combined_position = (position if combined_position is None
                                 else (combined_position | position))

        self.buy_signal = combined_buy
        self.position = combined_position

        print("🔄 執行回測...")
        sim_params = {
            'resample': None,
            'upload': False,
            'market': AdjustTWMarketInfo(),
            'fee_ratio': self.slippage + 0.001425,
            'tax_ratio': 0.003,
        }
        if self.position_limit is not None:
            sim_params['position_limit'] = self.position_limit

        self.report = sim(self.position, **sim_params)
        self._print_metrics()
        return self.report

    def _print_metrics(self):
        if self.report is None:
            print("報告物件為空，請先運行策略")
            return

        metrics = self.report.get_metrics()
        print("=" * 60)
        print(f"策略績效指標 ({self.get_strategy_name()})")
        print("=" * 60)
        print(f"年化報酬率: {metrics['profitability']['annualReturn']:.2%}")
        print(f"最大回檔: {metrics['risk']['maxDrawdown']:.2%}")
        print(f"總交易次數: {self.report.get_trades().shape[0]} 筆")
        print(f"單檔持股上限: "
              f"{f'{self.position_limit:.1%}' if self.position_limit else '無'}")
        print("=" * 60)
