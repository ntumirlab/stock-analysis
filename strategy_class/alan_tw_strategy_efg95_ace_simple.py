"""EFG95% + ACE 組合（簡單出場）— 詳見 docs/alan_tw_strategy_efg95_ace_simple.md"""

from .alan_tw_strategy_ace_simple import AlanTWStrategyACESimple
from .alan_tw_strategy_efg95_ace_full import AlanTWStrategyEFG95ACEFull
from .alan_tw_strategy_efg95_simple import AlanTWStrategyEFG95Simple


class _ACE_A90C90Simple(AlanTWStrategyACESimple):
    """組合用的 ACE 分量：A 90%、C 90%，出場與 EFG95% 分量統一為簡單出場"""
    extra_high_pct_a = 0.90
    extra_high_pct_c = 0.90
    sell_type = 'simple'


class AlanTWStrategyEFG95ACESimple(AlanTWStrategyEFG95ACEFull):
    """同 AlanTWStrategyEFG95ACEFull，但兩個分量的出場統一為簡單出場"""

    COMPONENTS = (AlanTWStrategyEFG95Simple, _ACE_A90C90Simple)

    def get_strategy_name(self):
        return "EFG95%_加_ACE_A90C90E_簡單出場"
