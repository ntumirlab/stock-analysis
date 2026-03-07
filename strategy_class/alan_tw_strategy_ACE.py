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

from .alan_tw_strategy_base import AlanTWStrategyBase


class AlanTWStrategyACE(AlanTWStrategyBase):
    """
    Alan TW Strategy ACE - 策略 A|C|E 組合
    """

    def get_strategy_name(self):
        """返回策略名稱"""
        return "ACE"

    def get_strategy_configs(self):
        """
        返回 A|C|E 策略參數配置

        Returns:
            list: 策略配置列表
        """
        return [
            {
                'name': 'A',
                'top_n': 20,
                'op_growth': 1.001,  # 營益率+0.1%
                'new_high_days': 120,
                'bias_ranges': {
                    'bias_5': (0.03, 0.13),
                    'bias_10': (0.05, 0.16),
                    'bias_20': (0.08, 0.19),
                    'bias_60': (0.08, 0.20),
                    'bias_120': (0.05, 0.26),
                    'bias_240': (0.08, 0.26),
                }
            },
            {
                'name': 'C',
                'top_n': 25,
                'op_growth': 1.125,  # 營益率+12.5%
                'new_high_days': 120,
                'bias_ranges': {
                    'bias_5': (0.03, 0.13),
                    'bias_10': (0.05, 0.16),
                    'bias_20': (0.08, 0.19),
                    'bias_60': (0.08, 0.20),
                    'bias_120': (0.05, 0.27),
                    'bias_240': (0.08, 0.31),
                }
            },
            {
                'name': 'E',
                'top_n': 40,
                'op_growth': 1.125,  # 營益率+12.5%
                'new_high_days': 480,
                'bias_ranges': {
                    'bias_5': (0.03, 0.13),
                    'bias_10': (0.05, 0.16),
                    'bias_20': (0.08, 0.19),
                    'bias_60': (0.08, 0.20),
                    'bias_120': (0.05, 0.35),
                    'bias_240': (0.08, 0.35),
                }
            },
        ]


# Example usage:
if __name__ == "__main__":
    from utils.authentication import AuthenticationManager
    auth = AuthenticationManager()
    auth.login_finlab()

    strategy = AlanTWStrategyACE()
    report = strategy.run_strategy()
