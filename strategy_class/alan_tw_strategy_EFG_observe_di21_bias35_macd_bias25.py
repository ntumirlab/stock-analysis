"""
Alan TW Strategy EFG Observe DI21 Bias35 MACD Bias25
基底同 AlanTWStrategyEFGObserve，出場條件改為：

    3日線↓ AND DIF↓ AND (
        -DI > 21
        OR (3日乖離 < -3.5% AND 5日乖離 < -3.5%)
        OR (MACD(DEA)↓ AND 3日乖離 < -2.5% AND 5日乖離 < -2.5%)
    )
"""

from finlab import data
from .alan_tw_strategy_EFG_observe import AlanTWStrategyEFGObserve


class AlanTWStrategyEFGObserveDI21Bias35MACDBias25(AlanTWStrategyEFGObserve):
    """
    EFG Observe 策略，搭配三分支 OR 出場條件：
    3日線↓ AND DIF↓ AND (
        -DI > 21
        OR (3日乖離 < -3.5% AND 5日乖離 < -3.5%)
        OR (MACD(DEA)↓ AND 3日乖離 < -2.5% AND 5日乖離 < -2.5%)
    )
    """

    def get_strategy_name(self):
        return "EFG_Observe_新出場_乖離3.5%_DI21_MACD乖離2.5%"

    def _build_sell_condition(self):
        ma3 = self.adj_close.rolling(3).mean()
        ma5 = self.adj_close.rolling(5).mean()

        with data.universe(market='TSE_OTC'):
            dif, macd, _ = data.indicator(
                'MACD', fastperiod=12, slowperiod=26, signalperiod=9, adjust_price=True
            )
            minus_di = data.indicator('MINUS_DI', timeperiod=14, adjust_price=True)

        bias_3 = (self.adj_close - ma3) / ma3
        bias_5 = (self.adj_close - ma5) / ma5

        return (
            (ma3 < ma3.shift(1)) &
            (dif < dif.shift(1)) &
            (
                (minus_di > 21) |
                ((bias_3 < -0.035) & (bias_5 < -0.035)) |
                ((macd < macd.shift(1)) & (bias_3 < -0.025) & (bias_5 < -0.025))
            )
        )


# Example usage:
if __name__ == "__main__":
    from utils.authentication import AuthenticationManager
    auth = AuthenticationManager()
    auth.login_finlab()

    strategy = AlanTWStrategyEFGObserveDI21Bias35MACDBias25()
    report = strategy.run_strategy()
