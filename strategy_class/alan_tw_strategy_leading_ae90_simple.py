"""領先潛力族群 AE 90% 簡單出場 — 詳見 docs/alan_tw_strategy_leading_ae90_simple.md

沿用領先潛力族群儀表板的籌碼＋基本面配對（買超前 20＋營益率 +0.1%、買超前 40＋營益率 +12%），
技術面採 A、E 模型的完整條件（均線上升、站上均線、六組乖離區間），
創新高改為「收盤 ≥ 480 日盤中最高價 × 90%」（取 K 線最高點而非收盤新高，與七模型不同）。
"""

from .alan_tw_strategy_base import AlanTWStrategyBase


class AlanTWStrategyLeadingAE90Simple(AlanTWStrategyBase):

    sell_type = 'simple'

    # 創新高：480 日盤中最高價 × 90%（兩個子策略相同）
    new_high_days = 480
    new_high_pct = 0.90

    def get_strategy_name(self):
        return f"Leading_AE{self.new_high_pct:.0%}_簡單出場"

    def get_strategy_configs(self):
        return [
            {
                'name': "A'",
                'top_n': 20,
                'op_growth': 1.001,
                'new_high_days': self.new_high_days,
                'new_high_pct': self.new_high_pct,
                'new_high_source': 'high',
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
                'name': "E'",
                'top_n': 40,
                'op_growth': 1.12,
                'new_high_days': self.new_high_days,
                'new_high_pct': self.new_high_pct,
                'new_high_source': 'high',
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
