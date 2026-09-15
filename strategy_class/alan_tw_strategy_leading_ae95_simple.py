"""領先潛力族群 AE 95% 簡單出場 — 詳見 docs/alan_tw_strategy_leading_ae95_simple.md

與 AE 90% 完全相同，僅創新高門檻提高為「收盤 ≥ 480 日盤中最高價 × 95%」。
"""

from .alan_tw_strategy_leading_ae90_simple import AlanTWStrategyLeadingAE90Simple


class AlanTWStrategyLeadingAE95Simple(AlanTWStrategyLeadingAE90Simple):

    new_high_pct = 0.95
