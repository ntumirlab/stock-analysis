"""
Alan TW Strategy EFG Observe
組合 E、F、G 三個策略條件，調整乖離下限

此策略為三個子策略的組合 (E|F|G)：
- 策略 E: top_n=40, 營益率 12%, BIAS: 3~13, 5~16, 5~19, 5~20, 5~34, 5~34, 價格>=480天新高*95%
- 策略 F: top_n=40, 營益率 12%, BIAS: 3~13, 5~16, 5~24, 5~24, 5~100, 5~150, 價格>=650天新高*95%
- 策略 G: top_n=40, 營益率 12%, BIAS: 3~13, 5~16, 5~28, 5~28, 5~34, 5~34, 價格>=600天新高*95%

與原版 EFG 差異：
- 創新高條件：價格 >= 收盤新高 * 95%
- 乖離率下限：bias_5 = 3%, bias_10/20/60/120/240 = 5%（原為 -3%/0%）

每個策略都結合三大面向：
- 籌碼面：三大法人與主力買賣超
- 技術面：均線、乖離率、DMI、KD、MACD等指標
- 基本面：營業利益率成長
"""

from .alan_tw_strategy_base import AlanTWStrategyBase


class AlanTWStrategyEFGObserve(AlanTWStrategyBase):
    """
    Alan TW Strategy EFG Observe - 策略 E|F|G 組合
    調整乖離下限 + 95% 創新高
    """

    def get_strategy_name(self):
        """返回策略名稱"""
        return "EFG_Observe"

    def get_strategy_configs(self):
        """
        返回 E|F|G 策略參數配置

        Returns:
            list: 策略配置列表
        """
        return [
            {
                'name': 'E',
                'top_n': 40,
                'op_growth': 1.12,  # 營益率+12%
                'new_high_days': 480,
                'new_high_pct': 0.95,  # 95% 創新高
                'bias_ranges': {
                    'bias_5': (0.03, 0.13),
                    'bias_10': (0.05, 0.16),
                    'bias_20': (0.05, 0.19),
                    'bias_60': (0.05, 0.20),
                    'bias_120': (0.05, 0.34),
                    'bias_240': (0.05, 0.34),
                }
            },
            {
                'name': 'F',
                'top_n': 40,
                'op_growth': 1.12,  # 營益率+12%
                'new_high_days': 650,
                'new_high_pct': 0.95,  # 95% 創新高
                'bias_ranges': {
                    'bias_5': (0.03, 0.13),
                    'bias_10': (0.05, 0.16),
                    'bias_20': (0.05, 0.24),
                    'bias_60': (0.05, 0.24),
                    'bias_120': (0.05, 1.00),
                    'bias_240': (0.05, 1.50),
                }
            },
            {
                'name': 'G',
                'top_n': 40,
                'op_growth': 1.12,  # 營益率+12%
                'new_high_days': 600,
                'new_high_pct': 0.95,  # 95% 創新高
                'bias_ranges': {
                    'bias_5': (0.03, 0.13),
                    'bias_10': (0.05, 0.16),
                    'bias_20': (0.05, 0.28),
                    'bias_60': (0.05, 0.28),
                    'bias_120': (0.05, 0.34),
                    'bias_240': (0.05, 0.34),
                }
            },
        ]


# Example usage:
if __name__ == "__main__":
    from utils.authentication import AuthenticationManager
    auth = AuthenticationManager()
    auth.login_finlab()

    strategy = AlanTWStrategyEFGObserve()
    report = strategy.run_strategy()
