"""EFG95% + ACE 組合 — 詳見 docs/alan_tw_strategy_efg95_ace.md"""

from finlab.backtest import sim

from .alan_tw_strategy_base import AlanTWStrategyBase, AdjustTWMarketInfo
from .alan_tw_strategy_efg95_full import AlanTWStrategyEFG95Full
from .alan_tw_strategy_ace_simple import AlanTWStrategyACESimple


class _ACE_A90C90Full(AlanTWStrategyACESimple):
    """組合用的 ACE 分量：A 90%、C 90%，出場與 EFG95% 分量統一為完整出場"""
    extra_high_pct_a = 0.90
    extra_high_pct_c = 0.90
    sell_type = 'full'


class AlanTWStrategyEFG95ACE(AlanTWStrategyBase):
    """
    兩個分量各自計算進場訊號，出場統一為完整出場，持倉取聯集：
        EFG95%、ACE(A90% C90% E)

    出場統一的原因：同一檔標的可能同時被兩個分量選中，
    若兩邊出場條件不同，聯集持倉的實際出場時點會取決於較晚的一方，語意不清。
    """

    COMPONENTS = (AlanTWStrategyEFG95Full, _ACE_A90C90Full)

    def get_strategy_name(self):
        return "EFG95%_加_ACE_A90C90E_完整出場"

    def get_strategy_configs(self):
        configs = []
        for cls in self.COMPONENTS:
            for cfg in cls.get_strategy_configs(cls):
                configs.append({**cfg, 'component': cls.__name__})
        return configs

    def run_strategy(self):
        print(f"🚀 開始運行策略 {self.get_strategy_name()}...")

        self.components = {}
        combined_buy = None
        combined_position = None

        for cls in self.COMPONENTS:
            strategy = cls()
            print(f"📊 計算分量 {strategy.get_strategy_name()} ...")

            buy = None
            for config in strategy.get_strategy_configs():
                signal = strategy._build_single_strategy_signal(config)
                buy = signal if buy is None else (buy | signal)
            buy = buy.loc[self.start_date:]
            sell = strategy._build_sell_condition()

            position = buy.hold_until(sell)
            self.components[cls.__name__] = {'buy': buy, 'sell': sell, 'position': position}

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
        print(f"單檔持股上限: {self.position_limit:.1%}")
        print("=" * 60)
