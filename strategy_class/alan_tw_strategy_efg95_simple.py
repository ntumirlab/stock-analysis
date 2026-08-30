"""EFG95% 簡單出場 — 詳見 docs/alan_tw_strategy_efg95_simple.md

進場條件與 EFG95% 完整出場（AlanTWStrategyEFG95Full）完全相同，僅出場改為簡單出場。
"""

from .alan_tw_strategy_efg95_full import AlanTWStrategyEFG95Full


class AlanTWStrategyEFG95Simple(AlanTWStrategyEFG95Full):

    sell_type = 'simple'

    def get_strategy_name(self):
        return "EFG95%_簡單出場"
