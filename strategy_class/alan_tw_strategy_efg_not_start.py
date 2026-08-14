"""EFG 未發動模型 — 詳見 docs/alan_tw_strategy_efg_not_start.md"""

from .alan_tw_strategy_not_start_base import AlanTWStrategyNotStartBase


class AlanTWStrategyEFGNotStart(AlanTWStrategyNotStartBase):

    entry_plus_di_min = 22
    entry_minus_di_max = 24

    def get_strategy_name(self):
        return "EFG_未發動模型"

    def get_strategy_configs(self):
        return [
            {
                'name': 'E',
                'top_n': 40,
                'op_growth': 1.12,
                'new_high_days': 480,
                'entry_high_pct': 0.93,
                'exit_high_pct': 0.91,
                'bias_ranges': {
                    'bias_5': (-0.03, 0.13),
                    'bias_10': (-0.03, 0.16),
                    'bias_20': (-0.03, 0.19),
                    'bias_60': (-0.03, 0.20),
                    'bias_120': (-0.03, 0.34),
                    'bias_240': (0.00, 0.34),
                },
            },
            {
                'name': 'F',
                'top_n': 40,
                'op_growth': 1.12,
                'new_high_days': 650,
                'entry_high_pct': 0.93,
                'exit_high_pct': 0.91,
                'bias_ranges': {
                    'bias_5': (-0.03, 0.13),
                    'bias_10': (-0.03, 0.16),
                    'bias_20': (-0.03, 0.24),
                    'bias_60': (-0.03, 0.24),
                    'bias_120': (-0.03, 0.45),   # F 收斂：原 100%
                    'bias_240': (0.00, 0.90),    # F 收斂：原 150%
                },
            },
            {
                'name': 'G',
                'top_n': 40,
                'op_growth': 1.12,
                'new_high_days': 600,
                'entry_high_pct': 0.93,
                'exit_high_pct': 0.91,
                'bias_ranges': {
                    'bias_5': (-0.03, 0.13),
                    'bias_10': (-0.03, 0.16),
                    'bias_20': (-0.03, 0.28),
                    'bias_60': (-0.03, 0.28),
                    'bias_120': (-0.03, 0.34),
                    'bias_240': (0.00, 0.34),
                },
            },
        ]
