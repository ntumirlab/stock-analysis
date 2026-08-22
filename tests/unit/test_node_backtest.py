"""節點日期運算測試（純運算，不需要 finlab）。

實際日期取自 logs/_node_check.py 已驗證過的四份清單。
"""

import numpy as np
import pandas as pd
import pytest

from core.node_backtest import (
    HOLD_WEEKS,
    align_to_sunday,
    check_trades,
    is_settled,
    is_tradable_list_date,
    node_dates,
    node_return,
    node_window,
    nth_sunday_of_month,
    settle_day,
    window_end,
)

MON, FRI = 0, 4  # config 的 buy_weekday=1 / sell_weekday=5 減 1 之後


def _calendar(start, end):
    """全日曆索引（position 就是 resample('D') 出來的）。"""
    return pd.date_range(start, end, freq='D')


def _trading_days(start, end, holidays=()):
    idx = pd.date_range(start, end, freq='B')
    return idx[~idx.isin(pd.DatetimeIndex(holidays))]


def _position(index, held_range=None):
    """單欄 position；held_range 給 (start, end) 表示那段期間持倉。"""
    df = pd.DataFrame(False, index=index, columns=['2330'])
    if held_range:
        lo, hi = (pd.Timestamp(d) for d in held_range)
        df.loc[(df.index >= lo) & (df.index <= hi), '2330'] = True
    return df


class TestNodeDates:
    @pytest.mark.parametrize('list_date, entry, exit_', [
        ('2026-06-14', '2026-06-15', '2026-06-19'),
        ('2026-07-05', '2026-07-06', '2026-07-10'),
        ('2026-07-12', '2026-07-13', '2026-07-17'),
        ('2026-08-09', '2026-08-10', '2026-08-14'),
    ])
    def test_weekly_enters_monday_and_exits_that_friday(self, list_date, entry, exit_):
        e, x = node_dates(list_date, MON, FRI, HOLD_WEEKS['weekly'])
        assert e == pd.Timestamp(entry)
        assert x == pd.Timestamp(exit_)

    def test_four_week_exit_matches_the_production_offset(self):
        # 正式策略寫的是 list_date + 22 + sell_weekday
        list_date = pd.Timestamp('2026-07-05')
        _, x = node_dates(list_date, MON, FRI, HOLD_WEEKS['monthly'])
        assert x == list_date + pd.Timedelta(days=22 + FRI)
        assert x == pd.Timestamp('2026-07-31')

    def test_monthly_and_weekly_4w_hold_the_same_length(self):
        assert HOLD_WEEKS['monthly'] == HOLD_WEEKS['weekly_4w'] == 4
        assert node_dates('2026-07-05', MON, FRI, 4) == node_dates('2026-07-05', MON, FRI, 4)

    def test_entry_follows_a_different_buy_weekday(self):
        e, x = node_dates('2026-07-05', 2, FRI, 1)  # 週三買
        assert e == pd.Timestamp('2026-07-08')
        assert x == pd.Timestamp('2026-07-10')


class TestNthSundayOfMonth:
    @pytest.mark.parametrize('list_date, nth', [
        ('2026-07-05', 1),
        ('2026-07-12', 2),
        ('2026-07-19', 3),
        ('2026-07-26', 4),
        ('2026-08-02', 1),
        ('2026-08-09', 2),
    ])
    def test_counts_sundays_within_the_month(self, list_date, nth):
        assert nth_sunday_of_month(list_date) == nth

    def test_month_starting_on_a_sunday_counts_from_day_one(self):
        assert pd.Timestamp('2026-03-01').weekday() == 6
        assert nth_sunday_of_month('2026-03-01') == 1
        assert nth_sunday_of_month('2026-03-08') == 2


class TestNodeWindow:
    def test_start_is_the_last_flat_trading_day_before_entry(self):
        index = _calendar('2026-06-25', '2026-07-20')
        # 進場 7/06（週一），持倉到 7/10；shift(-1) 後訊號落在 7/05
        pos = _position(index, ('2026-07-05', '2026-07-09'))
        td = _trading_days('2026-06-25', '2026-07-20')

        start, end = node_window(pos, '2026-07-06', td, '2026-07-10')

        assert start == pd.Timestamp('2026-07-03')  # 前一個週五
        assert not pos.loc[start].any()
        assert end == pd.Timestamp('2026-07-14')  # 結算日 7/10 再兩個交易日

    def test_start_retreats_further_when_that_friday_is_a_holiday(self):
        """7/10 休市那次：寫死「進場日減三天」會落在非交易日，節點報酬因此算錯。"""
        index = _calendar('2026-07-01', '2026-07-31')
        pos = _position(index, ('2026-07-12', '2026-07-16'))
        td = _trading_days('2026-07-01', '2026-07-31', holidays=['2026-07-10'])

        start, _ = node_window(pos, '2026-07-13', td, '2026-07-17')

        assert start == pd.Timestamp('2026-07-09')  # 退到週四，不是休市的週五
        assert not pos.loc[start].any()

    def test_end_leaves_slack_trading_days_after_the_settlement(self):
        index = _calendar('2026-07-01', '2026-08-31')
        pos = _position(index, ('2026-07-05', '2026-07-09'))
        td = _trading_days('2026-07-01', '2026-08-31')

        _, end = node_window(pos, '2026-07-06', td, '2026-07-10', slack_days=3)

        assert end == pd.Timestamp('2026-07-15')

    def test_slack_counts_trading_days_so_a_holiday_shifts_the_end(self):
        """7/10 休市那次：成交順延到 7/13，終點跟著往後，不是照日曆天硬加。"""
        index = _calendar('2026-07-01', '2026-07-31')
        pos = _position(index, ('2026-07-12', '2026-07-16'))
        td = _trading_days('2026-07-01', '2026-07-31', holidays=['2026-07-10'])

        _, end = node_window(pos, '2026-07-13', td, '2026-07-17')

        assert settle_day('2026-07-17', td) == pd.Timestamp('2026-07-17')
        assert end == pd.Timestamp('2026-07-21')

    def test_end_does_not_depend_on_how_much_data_follows(self):
        """回歸：終點原本是 min(出場日 + 10 天, 資料最後一天)，於是「排程當晚算的」
        與「事後回填算的」視窗長度不同。finlab 的 sharpe / sortino / annualReturn 是
        對整段視窗算的，同一條 +4.12% 的權益曲線，結算日後 0 個交易日得到
        sharpe=15.13、7 個交易日得到 sharpe=7.97（差 1.9 倍），而 DAO 是
        INSERT OR IGNORE，先寫進去的就被凍住。終點必須只跟節點與交易日曆有關。
        """
        pos = _position(_calendar('2026-07-01', '2026-08-31'),
                        ('2026-07-05', '2026-07-09'))

        ends = {node_window(pos, '2026-07-06',
                            _trading_days('2026-07-01', last), '2026-07-10')[1]
                for last in ('2026-07-14', '2026-07-20', '2026-08-31')}

        assert ends == {pd.Timestamp('2026-07-14')}

    def test_refuses_a_node_whose_slack_is_not_in_the_data_yet(self):
        pos = _position(_calendar('2026-07-01', '2026-07-10'),
                        ('2026-07-05', '2026-07-09'))
        td = _trading_days('2026-07-01', '2026-07-10')

        with pytest.raises(ValueError):
            node_window(pos, '2026-07-06', td, '2026-07-10')


class TestSettleDay:
    def test_an_open_exit_day_settles_that_day(self):
        td = _trading_days('2026-07-01', '2026-07-31')
        assert settle_day('2026-07-17', td) == pd.Timestamp('2026-07-17')

    def test_a_closed_exit_day_defers_to_the_next_trading_day(self):
        """7/10 休市，賣單順延到 7/13。"""
        td = _trading_days('2026-07-01', '2026-07-31', holidays=['2026-07-10'])
        assert settle_day('2026-07-10', td) == pd.Timestamp('2026-07-13')

    def test_none_when_the_data_stops_before_the_exit(self):
        td = _trading_days('2026-07-01', '2026-07-09')
        assert settle_day('2026-07-10', td) is None
        assert window_end('2026-07-10', td) is None


class TestIsSettled:
    """結算＝資料已經長到能框出一個「與計算時間無關」的視窗，也就是結算日之後還要
    有 SLACK_TRADING_DAYS 個交易日。只到出場日當天是不夠的——那個視窗會被截短。"""

    def test_settled_once_the_window_plus_a_spare_day_is_in(self):
        # 出場 7/10，視窗到 7/14，再多要一個交易日 7/15
        td = _trading_days('2026-07-01', '2026-07-15')
        assert is_settled('2026-07-10', td) is True

    def test_not_settled_while_the_data_stops_at_the_exit_day(self):
        """排程當晚跑到的就是這種：資料只到出場日，視窗會比事後回填短一截。"""
        td = _trading_days('2026-07-01', '2026-07-10')
        assert is_settled('2026-07-10', td) is False

    def test_not_settled_with_only_part_of_the_slack(self):
        td = _trading_days('2026-07-01', '2026-07-13')
        assert is_settled('2026-07-10', td) is False

    def test_not_settled_while_the_window_end_is_the_last_row_of_the_data(self):
        """視窗終點壓在資料最後一列時也還不算數：sim 的價格 frame 是 adj_open 與
        adj_close 的交集，可能比這裡的 price:收盤價 短一天。"""
        td = _trading_days('2026-07-01', '2026-07-14')
        assert window_end('2026-07-10', td) == pd.Timestamp('2026-07-14')
        assert is_settled('2026-07-10', td) is False

    def test_a_closed_exit_day_counts_slack_from_the_deferred_settlement(self):
        """7/10 休市成交落在 7/13，slack 要從 7/13 起算，不是從 7/10。"""
        td = _trading_days('2026-07-01', '2026-07-15', holidays=['2026-07-10'])
        assert is_settled('2026-07-10', td) is False

        td = _trading_days('2026-07-01', '2026-07-16', holidays=['2026-07-10'])
        assert is_settled('2026-07-10', td) is True

    def test_not_settled_when_the_data_stops_before_the_exit(self):
        td = _trading_days('2026-07-01', '2026-07-09')
        assert is_settled('2026-07-10', td) is False

    def test_not_settled_for_a_future_exit(self):
        td = _trading_days('2026-07-01', '2026-07-10')
        assert is_settled('2026-08-14', td) is False


class TestIsTradableListDate:
    """4 週策略的進場週來自 `_get_nth_sundays`，而那支只跑 n=1~4。"""

    FIFTH_SUNDAYS = ['2025-11-30', '2026-03-29', '2026-05-31']

    @pytest.mark.parametrize('list_date', FIFTH_SUNDAYS)
    def test_a_four_week_strategy_never_enters_on_a_fifth_sunday(self, list_date):
        assert nth_sunday_of_month(list_date) == 5
        assert is_tradable_list_date(list_date, HOLD_WEEKS['monthly']) is False
        assert is_tradable_list_date(list_date, HOLD_WEEKS['weekly_4w']) is False

    @pytest.mark.parametrize('list_date', FIFTH_SUNDAYS)
    def test_weekly_enters_on_every_sunday(self, list_date):
        assert is_tradable_list_date(list_date, HOLD_WEEKS['weekly']) is True

    @pytest.mark.parametrize('list_date', [
        '2026-07-05', '2026-07-12', '2026-07-19', '2026-07-26'])
    def test_the_first_four_sundays_are_tradable_either_way(self, list_date):
        assert is_tradable_list_date(list_date, 1) is True
        assert is_tradable_list_date(list_date, 4) is True


class TestNodeReturn:
    def test_plain_average_of_the_per_stock_returns(self):
        trades = pd.DataFrame({'return': [0.10, -0.04, 0.03]})
        assert node_return(trades) == pytest.approx(0.03)

    def test_single_stock_node(self):
        assert node_return(pd.DataFrame({'return': [-0.0764]})) == pytest.approx(-0.0764)


class TestCheckTrades:
    def _trades(self, n=3, entry='2026-07-06', exit_='2026-07-10'):
        return pd.DataFrame({
            'return': [0.01] * n,
            'entry_date': [pd.Timestamp(entry)] * n,
            'exit_date': [pd.Timestamp(exit_)] * n,
        })

    def test_a_clean_node_passes(self):
        assert check_trades(self._trades(3), '2026-07-06', '2026-07-10', 3) is None

    def test_rejects_an_empty_result(self):
        empty = pd.DataFrame(columns=['return', 'entry_date', 'exit_date'])
        assert check_trades(empty, '2026-07-06', '2026-07-10', 3) == 'no trades'

    def test_rejects_a_trade_count_that_misses_the_list(self):
        msg = check_trades(self._trades(2), '2026-07-06', '2026-07-10', 3)
        assert msg == 'trade count 2 != list size 3'

    def test_rejects_more_than_one_entry_date(self):
        trades = self._trades(2)
        trades.loc[1, 'entry_date'] = pd.Timestamp('2026-07-13')
        assert check_trades(trades, '2026-07-06', '2026-07-10', 2) == '2 distinct entry dates'

    def test_rejects_a_position_that_never_closed(self):
        trades = self._trades(2)
        trades.loc[1, 'exit_date'] = pd.NaT
        assert check_trades(trades, '2026-07-06', '2026-07-10', 2) == 'position still open'

    def test_rejects_a_merged_two_week_trade(self):
        """連續回測時重疊持股被淨換倉的那個症狀：同一批股票出場日不一致。"""
        trades = self._trades(2)
        trades.loc[1, 'exit_date'] = pd.Timestamp('2026-07-17')
        assert check_trades(trades, '2026-07-06', '2026-07-10', 2) == '2 distinct exit dates'


class TestAlignToSunday:
    """對齊規則要跟 _create_df 一致，否則「哪些週日真的有清單」會算錯。"""

    @pytest.mark.parametrize('raw, aligned', [
        ('2026-08-09', '2026-08-09'),   # 週日當天產出 → 留在當天
        ('2026-08-10', '2026-08-16'),   # 週一 → 下一個週日
        ('2026-08-14', '2026-08-16'),   # 週五 → 下一個週日
        ('2026-08-15', '2026-08-16'),   # 週六 → 隔天
    ])
    def test_matches_the_create_df_rule(self, raw, aligned):
        assert align_to_sunday(raw) == pd.Timestamp(aligned)

    def test_accepts_a_timestamp(self):
        assert align_to_sunday(pd.Timestamp('2026-08-10')) == pd.Timestamp('2026-08-16')


class TestCheckTradesAgainstSignalDates:
    """休市會讓成交往後順延，往前則代表視窗起點沒對齊。"""

    def _trades(self, entry, exit_, n=2):
        return pd.DataFrame({
            'return': [0.01] * n,
            'entry_date': [pd.Timestamp(entry)] * n,
            'exit_date': [pd.Timestamp(exit_)] * n,
        })

    def test_deferred_dates_are_fine(self):
        # 7/10 休市，賣單順延到 7/13
        trades = self._trades('2026-07-06', '2026-07-13')
        assert check_trades(trades, '2026-07-06', '2026-07-10', 2) is None

    def test_rejects_an_entry_before_its_signal(self):
        trades = self._trades('2026-07-03', '2026-07-10')
        msg = check_trades(trades, '2026-07-06', '2026-07-10', 2)
        assert msg == 'entry 2026-07-03 precedes signal 2026-07-06'

    def test_rejects_an_exit_before_its_signal(self):
        trades = self._trades('2026-07-06', '2026-07-09')
        msg = check_trades(trades, '2026-07-06', '2026-07-10', 2)
        assert msg == 'exit 2026-07-09 precedes signal 2026-07-10'
