"""節點回測的日期運算（純運算，不 import finlab，CI 可測）。

一個節點 = 一份推薦清單 × 一組 ranks 的一次獨立回測。跑 sim 的部分在
`research/golden_ai_tw_strategy/backfill_golden_ai_nodes.py`，這裡只負責
「哪天進、哪天出、視窗從哪天起跑、算不算已結算」。

為什麼要獨立回測而不是從連續回測的交易紀錄分組：連續回測時兩份清單重疊的股票會被
finlab 淨換倉、記成一筆橫跨兩週的交易。實測 2026-07-05 那份清單就因為 7/10 休市、
賣單順延到 7/13 撞上新一週的買進日，被拆成 3 檔 / 4 檔合併兩週 / 2 檔三塊。
單清單獨立回測不可能發生這件事。
"""

import pandas as pd

from core.backtest_window import snap_cutoff_to_flat_trading_day

# 每個策略一個節點持有幾週。weekly 是一週一輪；monthly 與 weekly_4w 行為相同
# （都是週日清單持有四週），只差清單來源。
HOLD_WEEKS = {'weekly': 1, 'monthly': 4, 'weekly_4w': 4}


def node_dates(list_date, buy_weekday: int, sell_weekday: int, hold_weeks: int):
    """回傳 (entry_date, exit_date)，皆為名目日期，休市順延交給 finlab 訊號對齊。

    list_date 是對齊後的週日。buy_weekday / sell_weekday 是 0-based（策略的
    `__init__` 已經把 config 的 1~5 減過 1），週一＝0、週五＝4。

    偏移量與正式策略一致：weekly 的出場是 hold_until 抓進場後第一個週五
    ＝ list_date + 1 + sell_weekday；monthly/4W 明寫成 list_date + 22 + sell_weekday。
    兩者都是 (hold_weeks - 1) * 7 + 1 + sell_weekday。
    """
    list_date = pd.Timestamp(list_date)
    entry_date = list_date + pd.Timedelta(days=1 + buy_weekday)
    exit_date = list_date + pd.Timedelta(days=(hold_weeks - 1) * 7 + 1 + sell_weekday)
    return entry_date, exit_date


def nth_sunday_of_month(list_date) -> int:
    """清單日是當月第幾個週日（1 起算）。

    4W／月策略的 Week1~4 就是這個維度：正式策略用 `_get_nth_sundays` 挑當月第 n 個
    週日當進場週，節點制不必為此跑四份回測，存下這個值之後篩即可。
    """
    d = pd.Timestamp(list_date)
    first = d.replace(day=1)
    first_sunday = first + pd.Timedelta(days=(6 - first.weekday()) % 7)
    return (d - first_sunday).days // 7 + 1


def node_window(position, entry_date, trading_days, exit_date, slack_days: int = 10):
    """節點回測的視窗 (start, end)。

    起點退到進場日之前最近的「空手交易日」——直接寫死「進場日減幾天」會錯：
    2026-07-10 週五休市那次，寫死的起點讓首筆交易延後一天、節點報酬從 -7.64%
    變成 -5.14%。理由與 [core.backtest_window] 相同，這裡重用同一支函式。

    終點放到出場日之後 slack_days 天，留給休市順延；出場後本來就空手，
    多幾天平盤不會產生交易。
    """
    entry_date = pd.Timestamp(entry_date)
    exit_date = pd.Timestamp(exit_date)

    start = snap_cutoff_to_flat_trading_day(position, entry_date, trading_days)
    end = min(exit_date + pd.Timedelta(days=slack_days),
              pd.Timestamp(position.index.max()))
    return start, end


def is_settled(exit_date, trading_days) -> bool:
    """名目出場日之後還有交易日才算結算完畢。

    只是回填時的便宜預檢：休市會讓實際出場順延，所以真正的判準是 sim 跑完後
    `trades` 的出場日都不是 NaT，呼叫端仍要檢查。
    """
    td = pd.DatetimeIndex(trading_days)
    return bool((td > pd.Timestamp(exit_date)).any())


def node_return(trades) -> float:
    """節點報酬＝各股報酬的單純平均。

    實測與該視窗的權益變化到小數第六位相同（節點內只有一次進出、無複利），
    所以不必再從 creturn 推導。
    """
    return float(trades['return'].mean())


def check_trades(trades, entry_date, exit_date, n_stocks: int):
    """驗證一次 sim 真的只產出「一個節點」。不符就回傳原因字串，正常回傳 None。

    三個條件對應三種曾經踩過或可能踩到的狀況：交易筆數與清單檔數不符（部位被合併
    或漏單）、進場日不只一個（視窗起點沒對齊到空手交易日）、出場日不只一個或還沒
    出場（節點還沒結算）。
    """
    if len(trades) == 0:
        return 'no trades'
    if len(trades) != n_stocks:
        return f'trade count {len(trades)} != list size {n_stocks}'
    if trades['entry_date'].nunique() != 1:
        return f'{trades["entry_date"].nunique()} distinct entry dates'
    if trades['exit_date'].isna().any():
        return 'position still open'
    if trades['exit_date'].nunique() != 1:
        return f'{trades["exit_date"].nunique()} distinct exit dates'
    return None
