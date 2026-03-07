"""
Alan TW Strategy FG (Combined)
策略 FG: 組合 F、G 兩個策略條件

此策略為兩個子策略的組合 (F|G)：
- 策略 F: top_n=40, 營益率 12%, BIAS: 3~13, 5~16, 8~24, 8~24, 5~100, 8~150, 創650天新高
- 策略 G: top_n=40, 營益率 12%, BIAS: 3~13, 5~16, 8~28, 8~28, 5~34, 8~34, 創600天新高

每個策略都結合三大面向：
- 籌碼面：三大法人與主力買賣超
- 技術面：均線、乖離率、DMI、KD、MACD等指標
- 基本面：營業利益率成長
"""

from .alan_tw_strategy_base import AlanTWStrategyBase


class AlanTWStrategyFG(AlanTWStrategyBase):
    """
    Alan TW Strategy FG - 策略 F|G 組合
    """

    def get_strategy_name(self):
        """返回策略名稱"""
        return "FG"

    def get_strategy_configs(self):
        """
        返回 F|G 策略參數配置

        Returns:
            list: 策略配置列表
        """
        return [
            {
                'name': 'F',
                'top_n': 40,
                'op_growth': 1.12,  # 營益率+12%
                'new_high_days': 650,
                'bias_ranges': {
                    'bias_5': (0.03, 0.13),
                    'bias_10': (0.05, 0.16),
                    'bias_20': (0.08, 0.24),
                    'bias_60': (0.08, 0.24),
                    'bias_120': (0.05, 1.00),
                    'bias_240': (0.08, 1.50),
                }
            },
            {
                'name': 'G',
                'top_n': 40,
                'op_growth': 1.12,  # 營益率+12%
                'new_high_days': 600,
                'bias_ranges': {
                    'bias_5': (0.03, 0.13),
                    'bias_10': (0.05, 0.16),
                    'bias_20': (0.08, 0.28),
                    'bias_60': (0.08, 0.28),
                    'bias_120': (0.05, 0.34),
                    'bias_240': (0.08, 0.34),
                }
            },
        ]


# Example usage:
if __name__ == "__main__":
    from utils.authentication import AuthenticationManager
    auth = AuthenticationManager()
    auth.login_finlab()

    strategy = AlanTWStrategyFG()
    report = strategy.run_strategy()
