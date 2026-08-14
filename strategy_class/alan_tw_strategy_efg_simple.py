"""EFG 簡單出場 — 詳見 docs/alan_tw_strategy_efg_simple.md"""

from .alan_tw_strategy_base import AlanTWStrategyBase


class AlanTWStrategyEFGSimple(AlanTWStrategyBase):

    sell_type = 'simple'

    def get_strategy_name(self):
        return "EFG_簡單出場"

    def get_strategy_configs(self):
        return [
            {
                'name': 'E',
                'top_n': 40,
                'op_growth': 1.12,
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
            {
                'name': 'F',
                'top_n': 40,
                'op_growth': 1.12,
                'new_high_days': 650,
                'new_high_pct': 1.0,
                'bias_ranges': {
                    'bias_5': (0.03, 0.13),
                    'bias_10': (0.05, 0.16),
                    'bias_20': (0.08, 0.24),
                    'bias_60': (0.08, 0.24),
                    'bias_120': (0.05, 0.45),   # F 收斂：原 100%
                    'bias_240': (0.08, 0.90),   # F 收斂：原 150%
                },
            },
            {
                'name': 'G',
                'top_n': 40,
                'op_growth': 1.12,
                'new_high_days': 600,
                'new_high_pct': 1.0,
                'bias_ranges': {
                    'bias_5': (0.03, 0.13),
                    'bias_10': (0.05, 0.16),
                    'bias_20': (0.08, 0.28),
                    'bias_60': (0.08, 0.28),
                    'bias_120': (0.05, 0.34),
                    'bias_240': (0.08, 0.34),
                },
            },
        ]
