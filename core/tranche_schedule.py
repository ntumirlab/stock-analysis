"""4 週策略的進場相位（tranche）排程——回測與實盤共用同一套定義（純運算，CI 可測）。

實盤（`core/trading_cycles.compute_cycles` + `build_tranche_specs`）從錨點純日期往後鋪：
每 7 天開一份 tranche、每份持有 4 週，所以每個買入 weekday 都恰好有一份 tranche 進場。

回測原本改用「當月第 n 個週日」挑進場週（`GoldenAITWStrategyMonthly._get_nth_sundays`），
一年只有 12x4=48 個位子、實際卻有約 52.2 週，**結構上每年必然漏掉約 4 份清單**——
錨在哪個星期幾都一樣，這是算術不是 bug。實測 2025-01~2026-12 的 104 個週一：月索引有
8 週沒有任何 tranche 進場，錨點連續輪動是 0 週。清單覆蓋率 42/45 對 45/45。

這裡把回測改成跟實盤同一套：錨點 + 7k 天連續輪動。相位因此不再有日曆語意
（不是「當月第幾週」，而是「錢落在四個錯開槽位的哪一個」），所以叫 tranche 不叫 Week。
"""

import pandas as pd

# 一輪有幾份 tranche。**這個數字同時是「開幾份」與「每份持有幾週」**——週頻進場、
# 持有 N 週，就要恰好 N 份錯開 7 天才無縫接上（出場後隔 3 天、也就是週五賣下週一買，
# 就輪到自己再進場），所以兩者必須相等，只留一個常數。
#
# `core.node_backtest.HOLD_WEEKS` 與 `GoldenAITWStrategyMonthly._run_core` 的出場偏移
# 都是從這裡導出的，別在那兩處另寫字面值。實盤那側的份數由 config 的 hold_weeks 決定
# （見 `order_executor.load_strategies` 把它當 num_tranches 傳給 `build_tranche_specs`），
# 與這裡各自獨立——回測要試幾份是回測自己的事。
NUM_TRANCHES = 4

# 各策略的錨點清單日（週日）。**從資料推導出來、算一次之後寫死**：若讓它每晚重算，
# 日後只要補進一份更早的清單、或清掉最早那筆，相位就會整組位移、整條歷史線回頭全變。
#
# 推導方式（2026-08-22 於 data_prod.db 算出）：
#     SELECT MIN(date) FROM recommendation_stocks WHERE frequency = '<weekly|monthly>'
#     -> 依 `_create_df` 的規則對齊到下一個週日 -> 該週日即錨點，隔天是第一個買入日
#
#   weekly  最早清單 2025-09-24（週三）-> 週日 2025-09-28 -> 首個買入日 2025-09-29
#   monthly 最早清單 2025-10-05（週日）-> 週日 2025-10-05 -> 首個買入日 2025-10-06
#
# 兩支策略各有各的錨點，因為它們的清單來源不同、開始時間差一週。
TRANCHE_ANCHOR_SUNDAYS = {
    'weekly_4w': pd.Timestamp('2025-09-28'),
    'monthly':   pd.Timestamp('2025-10-05'),
}


def anchor_sunday(strategy: str) -> pd.Timestamp:
    """該策略的錨點清單日。strategy 是 task_name（'weekly_4w' / 'monthly'）。"""
    try:
        return TRANCHE_ANCHOR_SUNDAYS[strategy]
    except KeyError:
        raise KeyError(
            f"{strategy} 沒有 tranche 錨點——只有持有多週的策略需要相位"
            f"（有錨點的是 {sorted(TRANCHE_ANCHOR_SUNDAYS)}）"
        ) from None


def tranche_of(strategy: str, list_date) -> int:
    """這份清單屬於第幾份 tranche（1~NUM_TRANCHES）。

    list_date 必須是對齊後的週日（`node_backtest.align_to_sunday` 的輸出、或
    `_create_df` 產出的 position index）；相位是按「距離錨點幾週」算的，
    傳非週日進來會算到隔壁的槽位。錨點之前的清單也算得出來（往前繞回）。
    """
    weeks = (pd.Timestamp(list_date).normalize() - anchor_sunday(strategy)).days // 7
    return weeks % NUM_TRANCHES + 1


def tranche_sundays(strategy: str, date_range, tranche: int) -> pd.DatetimeIndex:
    """第 `tranche` 份會進場的清單日（週日），涵蓋 `date_range` 的頭尾。

    取代舊的 `_get_nth_sundays(date_range, n)`：同樣是「這一輪要買哪幾個週日的清單」，
    只是改由錨點連續輪動決定。每個週日恰好屬於一份 tranche，所以四份加起來
    不重不漏地蓋滿整段期間，同一份 tranche 的相鄰進場間隔恆為 NUM_TRANCHES * 7 天。
    """
    if not 1 <= tranche <= NUM_TRANCHES:
        # 靜默回空 index 的話，呼叫端會拿到「這份 tranche 一次都沒進場」的空回測，
        # 看起來像資料不足而不像參數錯了。
        raise ValueError(f'tranche 要在 1~{NUM_TRANCHES} 之間，收到 {tranche}')

    idx = pd.DatetimeIndex(date_range)
    if len(idx) == 0:
        return pd.DatetimeIndex([])
    sundays = pd.date_range(start=idx.min(), end=idx.max(), freq='W-SUN')
    weeks = (sundays - anchor_sunday(strategy)).days // 7
    return sundays[weeks % NUM_TRANCHES == tranche - 1]
