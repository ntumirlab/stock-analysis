"""
Alan TW Strategy EFG Observe DI21 Bias35 MACD Bias25 ADX31 Bias05

基底同 AlanTWStrategyEFGObserveDI21Bias35MACDBias25，新增 OR 分支：
    ADX > 31 時，用小乖離 -0.5% 快速鎖利

完整出場條件：
    3日線↓ AND DIF↓ AND (
        -DI > 21
        OR (3日乖離 < -3.5% AND 5日乖離 < -3.5%)
        OR (MACD(DEA)↓ AND 3日乖離 < -2.5% AND 5日乖離 < -2.5%)
        OR (ADX > 31 AND 3日乖離 < -0.5% AND 5日乖離 < -0.5%)
    )
"""

from finlab import data
from .alan_tw_strategy_EFG_observe import AlanTWStrategyEFGObserve


class AlanTWStrategyEFGObserveDI21Bias35MACDBias25ADX31(AlanTWStrategyEFGObserve):

    def get_strategy_name(self):
        return "EFG_Observe_新出場_乖離3.5%_DI21_MACD乖離2.5%_ADX31乖離0.5%"

    def _build_sell_condition(self):
        ma3 = self.adj_close.rolling(3).mean()
        ma5 = self.adj_close.rolling(5).mean()

        with data.universe(market='TSE_OTC'):
            dif, macd, _ = data.indicator(
                'MACD', fastperiod=12, slowperiod=26, signalperiod=9, adjust_price=True
            )
            minus_di = data.indicator('MINUS_DI', timeperiod=14, adjust_price=True)
            adx = data.indicator('ADX', timeperiod=14, adjust_price=True)

        bias_3 = (self.adj_close - ma3) / ma3
        bias_5 = (self.adj_close - ma5) / ma5

        return (
            (ma3 < ma3.shift(1)) &
            (dif < dif.shift(1)) &
            (
                (minus_di > 21) |
                ((bias_3 < -0.035) & (bias_5 < -0.035)) |
                ((macd < macd.shift(1)) & (bias_3 < -0.025) & (bias_5 < -0.025)) |
                ((adx > 31) & (bias_3 < -0.005) & (bias_5 < -0.005))
            )
        )
