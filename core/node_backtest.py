"""節點回測的日期運算（純運算，不 import finlab，CI 可測）。

一個節點 = 一份推薦清單 × 一組 ranks 的一次獨立回測。跑 sim 的部分在
`jobs/golden_ai_node_executor.py`，這裡只負責
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


def align_to_sunday(date) -> pd.Timestamp:
    """推薦清單的日期對齊規則，與 `_create_df` 相同：週中產出的清單算下一個週日，
    週日當天產出的留在當天。抽出來是為了讓「哪些週日真的有清單」能被獨立算出來。
    """
    d = pd.Timestamp(date)
    return d + pd.Timedelta(days=6 - d.weekday())


def nth_sunday_of_month(list_date) -> int:
    """清單日是當月第幾個週日（1 起算）。

    4W／月策略的 Week1~4 就是這個維度：正式策略用 `_get_nth_sundays` 挑當月第 n 個
    週日當進場週，節點制不必為此跑四份回測，存下這個值之後篩即可。
    """
    d = pd.Timestamp(list_date)
    first = d.replace(day=1)
    first_sunday = first + pd.Timedelta(days=(6 - first.weekday()) % 7)
    return (d - first_sunday).days // 7 + 1


def is_tradable_list_date(list_date, hold_weeks: int) -> bool:
    """這份清單在正式策略裡真的會被買進嗎？

    4 週／月策略的進場週由 `GoldenAITWStrategyMonthly._get_nth_sundays` 決定，而那支
    只跑 n=1~4——**當月第 5 個週日的清單從來不會進場**。節點制若照跑，會生出策略
    根本沒持有過的部位（實測 2025-11-30、2026-03-29、2026-05-31 三份清單就是）。

    weekly 每週都進場，不受這個限制。
    """
    return hold_weeks == 1 or nth_sunday_of_month(list_date) <= 4


# 視窗終點在結算日之後還要留幾個交易日。**0 ＝ 賣出當天就收工**：部位在結算日已經出清，
# 之後的日子只是平盤，對節點沒有任何資訊，卻會讓結果晚好幾天才看得到——週五賣掉要等到
# 下週三，而這個檢視存在的意義就是「週五收盤就知道上週結果」。
#
# 也不必為「價格資料當下還沒齊」預留餘裕：那種情況 sim 收不掉部位，`check_trades` 會判
# position still open、節點不寫入、隔晚重試，而重試時 `window_end` 算出來的視窗完全一樣。
# 失敗模式是安全的，不值得為它讓每一個節點都延後。
SLACK_TRADING_DAYS = 0


def settle_day(exit_date, trading_days):
    """名目出場日實際成交的那天＝第一個 >= exit_date 的交易日。資料還沒走到就回 None。"""
    td = pd.DatetimeIndex(trading_days).sort_values()
    later = td[td >= pd.Timestamp(exit_date)]
    return later[0] if len(later) else None


def window_end(exit_date, trading_days, slack_days: int = SLACK_TRADING_DAYS):
    """視窗終點＝結算日再往後 slack_days 個交易日（預設 0 ＝ 結算日當天）。
    資料還沒走到就回 None。

    **終點必須是節點自己的函數，不能是「跑的當下資料到哪天」。** finlab 的
    sharpe / sortino / annualReturn 是對整段視窗的日報酬算的，視窗一長一短，同一個
    節點就會拿到不同的數字：實測同一條 +4.12% 的權益曲線，結算日後補 0 個交易日
    得到 sharpe=15.13、annualReturn=+722.9%，補 7 個交易日得到 sharpe=7.97、
    annualReturn=+127.0%（maxDrawdown 與勝率不受影響）。DAO 又是 INSERT OR IGNORE，
    先寫進去的那份會被凍住，於是「當晚排程算的」與「事後回填算的」永遠對不起來。

    正式策略的滾動回測沒有這個問題，是因為它用 `backtest_date` 把資料尾端整條釘死
    （`GoldenAITWStrategyBase._run_core` 的 data.truncate_end 與 universe 裁切、
    `_apply_cutoff` 的 ref、`TargetWeekdayTWMarket._truncate`）。節點沒有 backtest_date
    可釘，所以改從出場日推算終點，達到同一件事。
    """
    td = pd.DatetimeIndex(trading_days).sort_values()
    later = td[td >= pd.Timestamp(exit_date)]
    return later[slack_days] if len(later) > slack_days else None


def node_window(position, entry_date, trading_days, exit_date,
                slack_days: int = SLACK_TRADING_DAYS):
    """節點回測的視窗 (start, end)。呼叫前必須先確認 `is_settled` 為真。

    起點退到進場日之前最近的「空手交易日」——直接寫死「進場日減幾天」會錯：
    2026-07-10 週五休市那次，寫死的起點讓首筆交易延後一天、節點報酬從 -7.64%
    變成 -5.14%。理由與 [core.backtest_window] 相同，這裡重用同一支函式。

    終點見 `window_end`：兩端都只跟節點與交易日曆有關，跟哪天跑無關。
    """
    start = snap_cutoff_to_flat_trading_day(
        position, pd.Timestamp(entry_date), trading_days)
    end = window_end(exit_date, trading_days, slack_days)
    if end is None:
        raise ValueError(
            f'node not settled: {pd.Timestamp(exit_date).date()} '
            f'+ {slack_days} trading days is beyond the data')
    return start, end


def is_settled(exit_date, trading_days, slack_days: int = SLACK_TRADING_DAYS) -> bool:
    """行情資料是否已經長到足以框出一個「與計算時間無關」的視窗。

    判準就是 `window_end` 算不算得出來。休市不必特別處理：出場日開市時當天成交，
    休市時順延到下一個交易日，兩種都由「第一個 >= 出場日的交易日」涵蓋。

    **不預留餘裕**：sim 用的價格 frame 是 adj_open 與 adj_close 的交集（見
    `core.price_frames.mix_open_close`），兩者收盤後非同步更新，偶爾比 price:收盤價 短一天。
    那種情況部位收不掉、`check_trades` 會擋、隔晚重試，視窗仍是同一個——與其讓每個節點
    都晚幾天，不如讓偶發的那一個重試。

    只是回填時的便宜預檢——真正的判準是 sim 跑完後 `trades` 的出場日都不是 NaT，
    `check_trades` 仍會擋。
    """
    return window_end(exit_date, trading_days, slack_days) is not None


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

    # 休市會讓成交往後順延，往前則永遠是錯的——那代表視窗起點沒對齊，
    # finlab 把框裡第一列當成訊號日了。
    actual_entry = pd.Timestamp(trades['entry_date'].iloc[0])
    actual_exit = pd.Timestamp(trades['exit_date'].iloc[0])
    if actual_entry < pd.Timestamp(entry_date):
        return f'entry {actual_entry.date()} precedes signal {pd.Timestamp(entry_date).date()}'
    if actual_exit < pd.Timestamp(exit_date):
        return f'exit {actual_exit.date()} precedes signal {pd.Timestamp(exit_date).date()}'
    return None
