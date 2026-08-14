"""新 ACE（A85% C80% E）簡單出場 — 詳見 docs/alan_tw_strategy_ace_simple.md"""

from .alan_tw_strategy_base import AlanTWStrategyBase


class AlanTWStrategyACESimple(AlanTWStrategyBase):

    sell_type = 'simple'

    # A、C 額外要求「收盤 >= 480天收盤新高 × pct」
    extra_high_days = 480
    extra_high_pct_a = 0.85
    extra_high_pct_c = 0.80

    def get_strategy_name(self):
        return (f"ACE_A{self.extra_high_pct_a:.0%}C{self.extra_high_pct_c:.0%}E_簡單出場")

    def get_strategy_configs(self):
        return [
            {
                'name': 'A',
                'top_n': 20,
                'op_growth': 1.001,
                'new_high_days': 120,
                'new_high_pct': 1.0,
                'extra_new_high': (self.extra_high_days, self.extra_high_pct_a),
                'bias_ranges': {
                    'bias_5': (0.03, 0.13),
                    'bias_10': (0.05, 0.16),
                    'bias_20': (0.08, 0.19),
                    'bias_60': (0.08, 0.20),
                    'bias_120': (0.05, 0.26),
                    'bias_240': (0.08, 0.26),
                },
            },
            {
                'name': 'C',
                'top_n': 25,
                'op_growth': 1.125,
                'new_high_days': 120,
                'new_high_pct': 1.0,
                'extra_new_high': (self.extra_high_days, self.extra_high_pct_c),
                'bias_ranges': {
                    'bias_5': (0.03, 0.13),
                    'bias_10': (0.05, 0.16),
                    'bias_20': (0.08, 0.19),
                    'bias_60': (0.08, 0.20),
                    'bias_120': (0.05, 0.27),
                    'bias_240': (0.08, 0.31),
                },
            },
            {
                'name': 'E',
                'top_n': 40,
                'op_growth': 1.125,
                'new_high_days': 480,
                'new_high_pct': 1.0,
                'bias_ranges': {
                    'bias_5': (0.03, 0.13),
                    'bias_10': (0.05, 0.16),
                    'bias_20': (0.08, 0.19),
                    'bias_60': (0.08, 0.20),
                    'bias_120': (0.05, 0.35),
                    'bias_240': (0.08, 0.35),
                },
            },
        ]
